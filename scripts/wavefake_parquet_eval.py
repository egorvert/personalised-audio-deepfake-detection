# Faster WaveFake (HF parquet) eval client. Avoids the HF streaming +
# Audio-decoding bottleneck by downloading a small set of parquet partitions of
# ajaykarthick/wavefake-audio with hf_hub_download, reading them with pyarrow,
# sampling balanced bona-fide / spoof, decoding the embedded audio bytes, and
# POSTing each clip to the VDetect API.
#
# Usage:
#     python wavefake_parquet_eval.py \
#         --api-url http://127.0.0.1:8000 \
#         --partitions 0,1,2,60,61,62,120,121,122 \
#         --max-per-class 2000 \
#         --output ~/datasets/wavefake/results_wavefake_parquet.json

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# Label scheme used in ajaykarthick/wavefake-audio:
#   "R"     -> bonafide LJSpeech recording
#   "WF1"   -> spoofed audio from WaveFake architecture 1
#   ...
#   "WF7"   -> spoofed audio from WaveFake architecture 7
WAVEFAKE_REAL = "R"


def normalise_label(raw: str) -> Optional[int]:
    s = (raw or "").strip()
    if not s:
        return None
    if s == WAVEFAKE_REAL:
        return 0
    if s.startswith("WF") and s[2:].isdigit():
        return 1
    # Be permissive about other public WaveFake re-uploads.
    low = s.lower()
    if low in {"spoof", "fake", "deepfake"}:
        return 1
    if low in {"bona-fide", "bonafide", "real", "genuine"}:
        return 0
    return None


def architecture_tag(raw: str) -> str:
    """Return a short architecture tag for breakdown reporting."""
    s = (raw or "").strip()
    if s == WAVEFAKE_REAL:
        return "real"
    if s.startswith("WF") and s[2:].isdigit():
        return s
    return "other"


def encode_wav_bytes(array: np.ndarray, sampling_rate: int) -> bytes:
    if array.ndim > 1:
        array = array.mean(axis=-1)
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_audio_bytes(audio_blob: bytes) -> Tuple[np.ndarray, int]:
    """Decode an audio bytes blob (whatever container/codec) to (array, sr)."""
    array, sr = sf.read(io.BytesIO(audio_blob), dtype="float32", always_2d=False)
    return array, int(sr)


def send_bytes(
    api_url: str,
    audio_bytes: bytes,
    audio_id: str,
    threshold: float = 0.5,
    timeout_s: int = 30,
) -> dict:
    url = f"{api_url.rstrip('/')}/detect"
    resp = requests.post(
        url,
        files={"audio": (f"{audio_id}.wav", audio_bytes)},
        data={"threshold": str(threshold)},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def compute_metrics(labels: np.ndarray, scores: np.ndarray,
                    threshold: float = 0.5) -> dict:
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return {
            "eer": float("nan"),
            "auc": float("nan"),
            "n_pos": int((labels == 1).sum()) if len(labels) else 0,
            "n_neg": int((labels == 0).sum()) if len(labels) else 0,
        }
    fpr, tpr, thresh = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    idx = int(np.argmin(abs_diff))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresh[idx]) if idx < len(thresh) else 0.5
    auc = float(roc_auc_score(labels, scores))
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "score_mean_real": (float(scores[labels == 0].mean())
                            if (labels == 0).any() else float("nan")),
        "score_mean_fake": (float(scores[labels == 1].mean())
                            if (labels == 1).any() else float("nan")),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def parse_partition_arg(arg: str) -> List[int]:
    """Parse '0,1,2-5,10' into [0,1,2,3,4,5,10]."""
    out: List[int] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(piece))
    return sorted(set(out))


def download_partition(
    repo_id: str, partition: int, cache_dir: Optional[str]
) -> Path:
    rfilename = f"data/partition{partition}-00000-of-00001.parquet"
    print(f"  downloading {rfilename} ...")
    t0 = time.time()
    local = hf_hub_download(
        repo_id=repo_id,
        filename=rfilename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    elapsed = time.time() - t0
    size_mb = Path(local).stat().st_size / 1e6
    print(f"    -> {local} ({size_mb:.1f} MB in {elapsed:.1f}s, "
          f"{size_mb/elapsed:.1f} MB/s)")
    return Path(local)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--repo-id", default="ajaykarthick/wavefake-audio")
    parser.add_argument(
        "--partitions", default="0,1,2,30,60,90,120,125,128,130",
        help="Comma list / ranges of partition indices to download",
    )
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--probe-only", action="store_true",
                        help="Just inspect label distribution per partition")
    args = parser.parse_args()

    random.seed(args.seed)

    # API health
    print(f"Checking API at {args.api_url} ...")
    try:
        health = requests.get(
            f"{args.api_url.rstrip('/')}/health", timeout=10
        ).json()
        print(f"  Model loaded: {health['model_loaded']} "
              f"({health.get('model_type', '?')}) on {health.get('device', '?')}")
    except Exception as e:
        print(f"ERROR: API unreachable: {e}", file=sys.stderr)
        sys.exit(1)

    parts = parse_partition_arg(args.partitions)
    print(f"\nDownloading {len(parts)} partition(s): {parts}")

    paths: Dict[int, Path] = {}
    for p in parts:
        try:
            paths[p] = download_partition(args.repo_id, p, args.cache_dir)
        except Exception as e:
            print(f"  WARN: partition {p} failed: {e}", file=sys.stderr)

    # Per-partition label probe
    # Track row indices keyed by partition AND architecture tag, so we can
    # sample balanced both across the real/fake axis and across the seven
    # WaveFake vocoder architectures (WF1..WF7).
    print("\nProbing label distributions ...")
    # partition_arch_rows[partition][arch_tag] = [row_idx, ...]
    partition_arch_rows: Dict[int, Dict[str, List[int]]] = {}
    arch_counts_total: Dict[str, int] = {}
    label_counts_total = {"real": 0, "fake": 0, "other": 0}

    for p, path in paths.items():
        table = pq.read_table(str(path), columns=["real_or_fake"])
        labels_col = table.column("real_or_fake").to_pylist()
        per_arch: Dict[str, List[int]] = {}
        for i, lab in enumerate(labels_col):
            norm = normalise_label(lab)
            tag = architecture_tag(lab)
            if norm == 0:
                label_counts_total["real"] += 1
            elif norm == 1:
                label_counts_total["fake"] += 1
            else:
                label_counts_total["other"] += 1
                continue
            per_arch.setdefault(tag, []).append(i)
            arch_counts_total[tag] = arch_counts_total.get(tag, 0) + 1
        partition_arch_rows[p] = per_arch
        breakdown = " ".join(
            f"{a}={len(v)}" for a, v in sorted(per_arch.items())
        )
        print(f"  partition {p}: {breakdown}")

    total_real = arch_counts_total.get("real", 0)
    total_fake = sum(v for k, v in arch_counts_total.items() if k != "real")
    print(
        f"\nTotal across downloaded partitions: real={total_real} "
        f"fake={total_fake}  by-arch={dict(sorted(arch_counts_total.items()))}"
    )

    if args.probe_only:
        return

    if total_real < args.max_per_class or total_fake < args.max_per_class:
        print(
            f"\nWARN: not enough examples for max_per_class={args.max_per_class}. "
            f"Will sample what is available.",
            file=sys.stderr,
        )

    # Sample balanced
    n_real = min(args.max_per_class, total_real)
    n_fake = min(args.max_per_class, total_fake)
    print(
        f"\nSampling {n_real} real and {n_fake} fake "
        f"(balanced across partitions and architectures)..."
    )

    def _round_robin_sample(
        per_partition_rows: Dict[int, List[int]], n_target: int,
    ) -> List[Tuple[int, int]]:
        avail = {p: list(rows) for p, rows in per_partition_rows.items() if rows}
        for p in avail:
            random.shuffle(avail[p])
        keys = list(avail.keys())
        random.shuffle(keys)
        picked: List[Tuple[int, int]] = []
        while len(picked) < n_target and any(avail.values()):
            for p in keys:
                if not avail[p]:
                    continue
                picked.append((p, avail[p].pop()))
                if len(picked) >= n_target:
                    break
        return picked

    # Real picks: just balanced across partitions.
    real_partition_rows: Dict[int, List[int]] = {
        p: per.get("real", []) for p, per in partition_arch_rows.items()
    }
    real_picks_pl: List[Tuple[int, int]] = _round_robin_sample(
        real_partition_rows, n_real,
    )
    real_picks: List[Tuple[int, int, str]] = [
        (p, i, "real") for (p, i) in real_picks_pl
    ]

    # Fake picks: balanced across architectures *and* partitions.
    fake_archs = sorted(
        a for a in arch_counts_total.keys() if a != "real"
    )
    n_per_arch = max(1, n_fake // len(fake_archs)) if fake_archs else 0
    fake_picks: List[Tuple[int, int, str]] = []
    for arch in fake_archs:
        per_partition_rows = {
            p: per.get(arch, []) for p, per in partition_arch_rows.items()
        }
        picks = _round_robin_sample(per_partition_rows, n_per_arch)
        fake_picks.extend((p, i, arch) for (p, i) in picks)

    # If rounding leaves us short of n_fake, top up with whichever
    # architectures still have rows.
    if len(fake_picks) < n_fake:
        already = {(p, i) for (p, i, _) in fake_picks}
        leftover_pool: List[Tuple[int, int, str]] = []
        for arch in fake_archs:
            for p, per in partition_arch_rows.items():
                for idx in per.get(arch, []):
                    if (p, idx) not in already:
                        leftover_pool.append((p, idx, arch))
        random.shuffle(leftover_pool)
        for tup in leftover_pool:
            fake_picks.append(tup)
            if len(fake_picks) >= n_fake:
                break

    # Group picks per partition for efficient parquet reads.
    # by_partition[p][row_idx] = (label_int, arch_tag)
    by_partition: Dict[int, Dict[int, Tuple[int, str]]] = {}
    for p, idx, _arch in real_picks:
        by_partition.setdefault(p, {})[idx] = (0, "real")
    for p, idx, arch in fake_picks:
        by_partition.setdefault(p, {})[idx] = (1, arch)

    # Iterate selected rows and send to API
    print(f"\nSending {len(real_picks) + len(fake_picks)} examples to API...")
    all_labels: List[int] = []
    all_scores: List[float] = []
    all_archs: List[str] = []
    audio_ids: List[str] = []
    errors: List[str] = []

    t0 = time.time()
    total_to_send = sum(len(v) for v in by_partition.values())
    sent = 0

    for p, row_to_meta in by_partition.items():
        path = paths[p]
        wanted_rows = sorted(row_to_meta.keys())
        table = pq.read_table(str(path))
        audio_col = table.column("audio")
        id_col = table.column("audio_id")

        for row_idx in wanted_rows:
            label, arch = row_to_meta[row_idx]
            try:
                audio_struct = audio_col[row_idx].as_py()
                if "bytes" in audio_struct and audio_struct["bytes"]:
                    array, sr = decode_audio_bytes(audio_struct["bytes"])
                elif "array" in audio_struct:
                    array = np.asarray(audio_struct["array"], dtype=np.float32)
                    sr = int(audio_struct.get("sampling_rate", 16000))
                else:
                    raise RuntimeError(
                        f"unknown audio struct keys: {list(audio_struct.keys())}"
                    )
                wav_bytes = encode_wav_bytes(array, sr)
                audio_id = id_col[row_idx].as_py()
                result = send_bytes(
                    args.api_url, wav_bytes, str(audio_id),
                    args.threshold, args.timeout,
                )
                all_labels.append(label)
                all_scores.append(result["score"])
                all_archs.append(arch)
                audio_ids.append(str(audio_id))
            except Exception as e:
                errors.append(f"p{p}r{row_idx}/{arch}: {e}")
            sent += 1
            if sent % 50 == 0 or sent == total_to_send:
                elapsed = time.time() - t0
                rate = sent / elapsed if elapsed > 0 else 0
                eta = (total_to_send - sent) / rate if rate > 0 else 0
                print(
                    f"  [{sent}/{total_to_send}] {rate:.1f} files/s "
                    f"eta={eta/60:.1f} min", end="\r",
                )
    print()
    elapsed = time.time() - t0
    print(f"\nDone. Processed {len(all_labels)}/{total_to_send} files in "
          f"{elapsed:.1f}s ({len(all_labels)/elapsed:.1f} files/s)")
    if errors:
        print(f"  {len(errors)} errors")

    # Metrics
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)
    archs_arr = np.array(all_archs)
    overall = compute_metrics(labels_arr, scores_arr, args.threshold)

    # Per-architecture metrics: each architecture's fake examples vs all real.
    real_mask = labels_arr == 0
    real_labels = labels_arr[real_mask]
    real_scores = scores_arr[real_mask]
    per_arch_metrics: Dict[str, dict] = {}
    fake_arch_set = sorted({a for a in all_archs if a != "real"})
    for arch in fake_arch_set:
        fake_mask = (archs_arr == arch) & (labels_arr == 1)
        if not fake_mask.any():
            continue
        comb_labels = np.concatenate([real_labels, labels_arr[fake_mask]])
        comb_scores = np.concatenate([real_scores, scores_arr[fake_mask]])
        m = compute_metrics(comb_labels, comb_scores, args.threshold)
        m["n_arch_fake"] = int(fake_mask.sum())
        per_arch_metrics[arch] = m

    print("\n" + "=" * 78)
    print("WAVEFAKE (HF parquet) EVALUATION RESULTS")
    print("=" * 78)
    print(
        f"\n{'Overall':>20}  EER={overall['eer']*100:6.2f}%  "
        f"AUC={overall['auc']:.4f}  Acc={overall.get('accuracy', 0)*100:.1f}%  "
        f"F1={overall.get('f1', 0):.4f}"
    )
    print(
        f"  mean(score|real)={overall.get('score_mean_real', float('nan')):.4f}"
        f"   mean(score|fake)={overall.get('score_mean_fake', float('nan')):.4f}"
    )
    print(
        f"  n_real={overall.get('n_neg', 0)}  "
        f"n_fake={overall.get('n_pos', 0)}"
    )

    if per_arch_metrics:
        print(f"\n{'Architecture':<14} {'EER':>8} {'AUC':>8} {'fake n':>8} "
              f"{'mean(real)':>12} {'mean(fake)':>12}")
        print("-" * 78)
        for arch in sorted(per_arch_metrics.keys(),
                           key=lambda a: per_arch_metrics[a].get("eer", 1.0)):
            m = per_arch_metrics[arch]
            eer = m.get("eer", float("nan"))
            auc = m.get("auc", float("nan"))
            print(
                f"{arch:<14} "
                f"{eer*100 if not np.isnan(eer) else float('nan'):>7.2f}% "
                f"{auc:>8.4f} {m['n_arch_fake']:>8} "
                f"{m.get('score_mean_real', float('nan')):>12.4f} "
                f"{m.get('score_mean_fake', float('nan')):>12.4f}"
            )
    print("=" * 78)

    if args.output:
        out = {
            "api_url": args.api_url,
            "model_type": health.get("model_type"),
            "dataset": args.repo_id,
            "partitions": parts,
            "max_per_class": args.max_per_class,
            "threshold": args.threshold,
            "label_counts_total": label_counts_total,
            "arch_counts_total": arch_counts_total,
            "total_processed": len(all_labels),
            "total_errors": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "overall_metrics": overall,
            "per_architecture_metrics": per_arch_metrics,
            "errors": errors[:50],
        }
        Path(args.output).expanduser().write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
