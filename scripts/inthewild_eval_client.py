# Send audio files from the In-the-Wild dataset (Müller et al., 2022) to a
# running VDetect API and compute EER / AUC / accuracy / F1 plus a per-speaker
# breakdown. Dataset: https://deepfake-total.com/in_the_wild
#
# Expected layout:
#     release_in_the_wild/
#         meta.csv             # columns: file,speaker,label
#         0.wav 1.wav 2.wav    # flat 16 kHz mono WAVs
# Labels are 'spoof' (1) / 'bona-fide' (0).
#
# Usage:
#     python inthewild_eval_client.py \
#         --api-url http://127.0.0.1:8000 \
#         --dataset-dir ./release_in_the_wild \
#         --max-per-class 2000 \
#         --output results_inthewild.json

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


SPOOF_LABELS = {"spoof", "fake", "deepfake"}
BONAFIDE_LABELS = {"bona-fide", "bonafide", "real", "genuine"}


def load_metadata(meta_csv: Path) -> List[Dict[str, str]]:
    """Read meta.csv into a list of {file, speaker, label} dicts."""
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta.csv not found at {meta_csv}")

    rows: List[Dict[str, str]] = []
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise the key set; the dataset ships with 'file,speaker,label'.
            rows.append({
                "file": row.get("file") or row.get("filename") or "",
                "speaker": row.get("speaker") or "unknown",
                "label": (row.get("label") or "").strip().lower(),
            })
    return rows


def split_by_label(
    rows: List[Dict[str, str]],
    dataset_dir: Path,
) -> Tuple[List[Dict], List[Dict]]:
    """Split rows into (bonafide, spoof). Verifies each audio file exists."""
    bonafide, spoof = [], []
    missing = 0
    for row in rows:
        audio_path = dataset_dir / row["file"]
        if not audio_path.exists():
            missing += 1
            continue
        row = {**row, "path": audio_path}
        if row["label"] in SPOOF_LABELS:
            spoof.append(row)
        elif row["label"] in BONAFIDE_LABELS:
            bonafide.append(row)
        else:
            print(f"warn: unknown label '{row['label']}' for {row['file']}",
                  file=sys.stderr)
    if missing:
        print(f"warn: {missing} audio files referenced in meta.csv not found",
              file=sys.stderr)
    return bonafide, spoof


def sample_balanced(
    rows: List[Dict],
    max_count: int,
    by_speaker: bool = True,
) -> List[Dict]:
    """Random sample at most max_count rows.

    When by_speaker=True, sample roughly equally per speaker so a few
    over-represented voices do not dominate the eval set.
    """
    if max_count <= 0 or len(rows) <= max_count:
        return rows
    if not by_speaker:
        return random.sample(rows, max_count)

    by_spk: Dict[str, List[Dict]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)

    speakers = list(by_spk.keys())
    random.shuffle(speakers)

    # Round-robin draw.
    picked: List[Dict] = []
    while len(picked) < max_count and any(by_spk.values()):
        for s in speakers:
            if not by_spk[s]:
                continue
            picked.append(by_spk[s].pop(random.randrange(len(by_spk[s]))))
            if len(picked) >= max_count:
                break
    return picked


def send_file(api_url: str, file_path: Path, threshold: float = 0.5,
              timeout_s: int = 30) -> dict:
    url = f"{api_url.rstrip('/')}/detect"
    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            files={"audio": (file_path.name, f)},
            data={"threshold": str(threshold)},
            timeout=timeout_s,
        )
    resp.raise_for_status()
    return resp.json()


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    if len(labels) == 0:
        return {"eer": float("nan"), "auc": float("nan"), "n": 0}

    predictions = (scores >= threshold).astype(int)

    if len(set(labels.tolist())) < 2:
        # Cannot compute EER/AUC without both classes present.
        return {
            "eer": float("nan"),
            "auc": float("nan"),
            "accuracy": float((predictions == labels).mean()),
            "n_pos": int(labels.sum()),
            "n_neg": int(len(labels) - labels.sum()),
        }

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    min_idx = int(np.argmin(abs_diff))
    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    eer_threshold = (
        float(thresholds[min_idx]) if min_idx < len(thresholds) else 0.5
    )

    auc = float(roc_auc_score(labels, scores))

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "score_mean_real": float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan"),
        "score_mean_fake": float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def print_results(overall: Dict, per_speaker: Dict[str, Dict]) -> None:
    print("\n" + "=" * 78)
    print("IN-THE-WILD CROSS-DATASET EVALUATION RESULTS")
    print("=" * 78)
    print(
        f"\n{'Overall':>20}  EER={overall['eer']*100:6.2f}%  "
        f"AUC={overall['auc']:.4f}  Acc={overall.get('accuracy', 0)*100:.1f}%  "
        f"F1={overall.get('f1', 0):.4f}"
    )
    print(
        f"  mean(score|real)={overall.get('score_mean_real', float('nan')):.4f}   "
        f"mean(score|fake)={overall.get('score_mean_fake', float('nan')):.4f}"
    )

    if per_speaker:
        print(f"\n{'Speaker':<35} {'EER':>8} {'AUC':>8} {'real n':>7} {'fake n':>7}")
        print("-" * 78)
        rows = sorted(
            per_speaker.items(),
            key=lambda x: (np.isnan(x[1].get("eer", float("nan"))),
                           x[1].get("eer", 1.0)),
        )
        for spk, m in rows[:30]:
            eer_disp = (
                f"{m['eer']*100:>7.2f}%"
                if not np.isnan(m.get("eer", float("nan"))) else "    n/a "
            )
            auc_disp = (
                f"{m['auc']:>8.4f}"
                if not np.isnan(m.get("auc", float("nan"))) else "     n/a"
            )
            print(
                f"{spk[:34]:<35} {eer_disp} {auc_disp} {m['n_neg']:>7} {m['n_pos']:>7}"
            )
        if len(rows) > 30:
            print(f"  ... ({len(rows) - 30} more speakers in JSON)")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote In-the-Wild evaluation via VDetect API",
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path,
                        help="Path to release_in_the_wild/ directory")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Max files per class (0 = use all)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-request timeout in seconds")
    args = parser.parse_args()

    random.seed(args.seed)

    # Health check
    print(f"Checking API at {args.api_url} ...")
    try:
        health = requests.get(
            f"{args.api_url.rstrip('/')}/health", timeout=10
        ).json()
        print(
            f"  Model loaded: {health['model_loaded']} "
            f"({health.get('model_type', '?')}) on {health.get('device', '?')}"
        )
    except Exception as e:
        print(f"ERROR: Cannot reach API — {e}", file=sys.stderr)
        sys.exit(1)

    # Discover
    dataset_dir = args.dataset_dir
    meta_csv = dataset_dir / "meta.csv"
    print(f"\nLoading metadata from {meta_csv} ...")
    rows = load_metadata(meta_csv)
    print(f"  {len(rows)} rows in meta.csv")

    bonafide, spoof = split_by_label(rows, dataset_dir)
    print(
        f"  bonafide: {len(bonafide)} files  "
        f"spoof: {len(spoof)} files  "
        f"speakers: {len({r['speaker'] for r in rows})}"
    )

    if args.max_per_class > 0:
        print(f"\nSampling up to {args.max_per_class} per class "
              f"(balanced across speakers) ...")
        bonafide = sample_balanced(bonafide, args.max_per_class)
        spoof = sample_balanced(spoof, args.max_per_class)
        print(f"  after sampling: bonafide={len(bonafide)} spoof={len(spoof)}")

    total = len(bonafide) + len(spoof)
    if total == 0:
        print("ERROR: nothing to evaluate.", file=sys.stderr)
        sys.exit(1)

    # Send
    all_labels: List[int] = []
    all_scores: List[float] = []
    speakers: List[str] = []
    errors: List[str] = []
    processed = 0
    t0 = time.time()

    def _process(group: List[Dict], label: int, name: str) -> None:
        nonlocal processed
        print(f"\nProcessing {name} ({len(group)} files) ...")
        for i, row in enumerate(group, 1):
            try:
                result = send_file(
                    args.api_url, row["path"], args.threshold, args.timeout
                )
                all_labels.append(label)
                all_scores.append(result["score"])
                speakers.append(row["speaker"])
                processed += 1
            except Exception as e:
                errors.append(f"{name}/{row['file']}: {e}")
            if i % 50 == 0 or i == len(group):
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{len(group)}]  {rate:.1f} files/s  "
                    f"eta={eta/60:.1f} min", end="\r",
                )
        print()

    _process(bonafide, label=0, name="bonafide")
    _process(spoof, label=1, name="spoof")

    elapsed = time.time() - t0
    print(
        f"\nDone. Processed {processed}/{total} files in {elapsed:.1f}s "
        f"({processed/elapsed:.1f} files/s)"
    )
    if errors:
        print(f"  {len(errors)} errors (first few in output JSON)")

    # Metrics
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)
    speakers_arr = np.array(speakers)

    overall = compute_metrics(labels_arr, scores_arr, args.threshold)

    per_speaker: Dict[str, Dict] = {}
    for spk in sorted(set(speakers)):
        mask = speakers_arr == spk
        m = compute_metrics(
            labels_arr[mask], scores_arr[mask], args.threshold,
        )
        per_speaker[spk] = m

    print_results(overall, per_speaker)

    # Save
    if args.output:
        out = {
            "api_url": args.api_url,
            "model_type": health.get("model_type"),
            "dataset": "in_the_wild",
            "dataset_dir": str(dataset_dir),
            "max_per_class": args.max_per_class,
            "threshold": args.threshold,
            "total_processed": processed,
            "total_errors": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "overall_metrics": overall,
            "per_speaker_metrics": per_speaker,
            "errors": errors[:50],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
