# Personalised (FiLM) variant of inthewild_eval_client.py. Before scoring,
# each speaker is enrolled via /enroll with --enroll-k bona-fide samples; every
# subsequent /detect call includes their speaker_id so the FiLM path of the
# fusion detector is active. Lets us A/B against the non-personalised baseline.
# Enrolment clips are excluded from the eval pool; speakers with fewer than
# --enroll-k real samples are skipped.
#
# Usage:
#     python inthewild_personalised_eval.py \
#         --api-url http://127.0.0.1:8000 \
#         --dataset-dir ~/datasets/inthewild/release_in_the_wild \
#         --max-per-class 2000 \
#         --enroll-k 3 --speaker-prefix itw_ \
#         --output ~/datasets/inthewild/results_inthewild_personalised_2k.json

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
    rows: List[Dict[str, str]] = []
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "file": row.get("file") or row.get("filename") or "",
                "speaker": row.get("speaker") or "unknown",
                "label": (row.get("label") or "").strip().lower(),
            })
    return rows


def is_real(row: Dict[str, str]) -> bool:
    return row["label"] in BONAFIDE_LABELS


def is_fake(row: Dict[str, str]) -> bool:
    return row["label"] in SPOOF_LABELS


def safe_speaker_id(prefix: str, name: str) -> str:
    """Sanitise a speaker name into a stable ID for the prototype DB."""
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return prefix + "".join(keep)


def enroll_speaker(
    api_url: str,
    speaker_id: str,
    audio_paths: List[Path],
    timeout_s: int = 120,
) -> dict:
    url = f"{api_url.rstrip('/')}/enroll"
    files = []
    handles = []
    try:
        for p in audio_paths:
            fh = open(p, "rb")
            handles.append(fh)
            files.append(("files", (p.name, fh, "audio/wav")))
        data = {"speaker_id": speaker_id, "normalize": "true"}
        resp = requests.post(url, files=files, data=data, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    finally:
        for fh in handles:
            try:
                fh.close()
            except Exception:
                pass


def send_file(
    api_url: str,
    file_path: Path,
    speaker_id: str,
    threshold: float = 0.5,
    timeout_s: int = 30,
) -> dict:
    url = f"{api_url.rstrip('/')}/detect"
    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            files={"audio": (file_path.name, f)},
            data={"threshold": str(threshold), "speaker_id": speaker_id},
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
    if len(set(labels.tolist())) < 2:
        return {
            "eer": float("nan"),
            "auc": float("nan"),
            "accuracy": float((scores >= threshold).astype(int).mean()),
            "n_pos": int(labels.sum()),
            "n_neg": int(len(labels) - labels.sum()),
        }
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    idx = int(np.argmin(abs_diff))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = (
        float(thresholds[idx]) if idx < len(thresholds) else 0.5
    )
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


def round_robin_sample(
    rows_by_speaker: Dict[str, List[Dict]],
    n_target: int,
    rng: random.Random,
) -> List[Dict]:
    """Sample n_target rows balanced across speakers."""
    if n_target <= 0:
        return []
    avail = {s: list(rs) for s, rs in rows_by_speaker.items() if rs}
    for s in avail:
        rng.shuffle(avail[s])
    keys = list(avail.keys())
    rng.shuffle(keys)
    picked: List[Dict] = []
    while len(picked) < n_target and any(avail.values()):
        for s in keys:
            if not avail[s]:
                continue
            picked.append(avail[s].pop())
            if len(picked) >= n_target:
                break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personalised In-the-Wild eval (FiLM) via VDetect API",
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--enroll-k", type=int, default=3,
                        help="Number of bonafide samples to use per speaker "
                             "for enrolment (3-5 supported by API)")
    parser.add_argument("--speaker-prefix", default="itw_")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--enroll-only", action="store_true")
    parser.add_argument("--skip-enroll", action="store_true",
                        help="Assume speakers are already enrolled.")
    args = parser.parse_args()

    if args.enroll_k < 3 or args.enroll_k > 5:
        print("ERROR: --enroll-k must be in [3,5] (API constraint).",
              file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)

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

    # Read meta + group
    dataset_dir: Path = args.dataset_dir
    meta_csv = dataset_dir / "meta.csv"
    print(f"\nLoading metadata from {meta_csv} ...")
    rows = load_metadata(meta_csv)
    print(f"  {len(rows)} rows in meta.csv")

    # Verify audio files exist; attach path.
    rows_with_path: List[Dict] = []
    missing = 0
    for r in rows:
        p = dataset_dir / r["file"]
        if not p.exists():
            missing += 1
            continue
        rows_with_path.append({**r, "path": p})
    if missing:
        print(f"WARN: {missing} files referenced in meta.csv missing on disk",
              file=sys.stderr)

    # Group by speaker, sorted by file id (stable for reproducible enrolment)
    by_speaker: Dict[str, List[Dict]] = {}
    for r in sorted(rows_with_path, key=lambda x: x["file"]):
        by_speaker.setdefault(r["speaker"], []).append(r)

    # Choose enrolment + eval pools per speaker
    enrolment_per_speaker: Dict[str, List[Dict]] = {}
    eligible_real: Dict[str, List[Dict]] = {}
    eligible_fake: Dict[str, List[Dict]] = {}
    skipped: Dict[str, str] = {}

    for spk, rs in by_speaker.items():
        reals = [r for r in rs if is_real(r)]
        fakes = [r for r in rs if is_fake(r)]
        if len(reals) < args.enroll_k:
            skipped[spk] = (
                f"only {len(reals)} bonafide samples (<{args.enroll_k})"
            )
            continue
        # Deterministic: take the first K real (by sorted file id) for enrolment.
        enrolment_per_speaker[spk] = reals[:args.enroll_k]
        remaining_real = reals[args.enroll_k:]
        if not remaining_real and not fakes:
            skipped[spk] = "no eval material after enrolment"
            continue
        if remaining_real:
            eligible_real[spk] = remaining_real
        if fakes:
            eligible_fake[spk] = fakes

    print(
        f"  speakers in dataset: {len(by_speaker)}; "
        f"enrolled: {len(enrolment_per_speaker)}; "
        f"skipped: {len(skipped)}"
    )
    if skipped:
        for spk, why in list(skipped.items())[:5]:
            print(f"    skip {spk!r}: {why}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    # Enrol speakers
    enroll_results: Dict[str, dict] = {}
    enroll_errors: Dict[str, str] = {}

    if not args.skip_enroll:
        print(f"\nEnrolling {len(enrolment_per_speaker)} speakers "
              f"({args.enroll_k} samples each) ...")
        t0 = time.time()
        for i, (spk, enrol_rows) in enumerate(
            sorted(enrolment_per_speaker.items()), 1
        ):
            sid = safe_speaker_id(args.speaker_prefix, spk)
            paths = [r["path"] for r in enrol_rows]
            try:
                res = enroll_speaker(args.api_url, sid, paths, timeout_s=120)
                enroll_results[spk] = {"speaker_id": sid, "result": res}
            except Exception as e:
                enroll_errors[spk] = str(e)
            if i % 5 == 0 or i == len(enrolment_per_speaker):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i}/{len(enrolment_per_speaker)}] "
                    f"{rate:.1f} speakers/s", end="\r",
                )
        print()
        if enroll_errors:
            print(f"  WARN: {len(enroll_errors)} enrolment errors")
            for spk, err in list(enroll_errors.items())[:5]:
                print(f"    {spk}: {err}")

    if args.enroll_only:
        return

    # Sample balanced eval pool
    # Drop speakers with no successful enrolment if we tried to enrol them.
    if not args.skip_enroll:
        good_speakers = set(enroll_results.keys())
        eligible_real = {
            s: rs for s, rs in eligible_real.items() if s in good_speakers
        }
        eligible_fake = {
            s: rs for s, rs in eligible_fake.items() if s in good_speakers
        }

    total_real = sum(len(v) for v in eligible_real.values())
    total_fake = sum(len(v) for v in eligible_fake.values())
    n_real = min(args.max_per_class, total_real)
    n_fake = min(args.max_per_class, total_fake)
    print(
        f"\nEval pool: {total_real} real (across {len(eligible_real)} speakers), "
        f"{total_fake} fake (across {len(eligible_fake)} speakers)"
    )
    print(
        f"Sampling {n_real} real and {n_fake} fake (balanced across speakers)..."
    )

    real_samples = round_robin_sample(eligible_real, n_real, rng)
    fake_samples = round_robin_sample(eligible_fake, n_fake, rng)

    eval_set: List[Tuple[int, Dict]] = (
        [(0, r) for r in real_samples] + [(1, r) for r in fake_samples]
    )
    rng.shuffle(eval_set)

    # Detect with FiLM conditioning
    print(f"\nSending {len(eval_set)} examples to /detect (with speaker_id) ...")
    all_labels: List[int] = []
    all_scores: List[float] = []
    speakers_used: List[str] = []
    errors: List[str] = []

    t0 = time.time()
    for i, (label, row) in enumerate(eval_set, 1):
        sid = safe_speaker_id(args.speaker_prefix, row["speaker"])
        try:
            res = send_file(
                args.api_url, row["path"], sid, args.threshold, args.timeout,
            )
            all_labels.append(label)
            all_scores.append(res["score"])
            speakers_used.append(row["speaker"])
        except Exception as e:
            errors.append(f"{row['speaker']}/{row['file']}: {e}")
        if i % 50 == 0 or i == len(eval_set):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(eval_set) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(eval_set)}]  {rate:.1f} files/s  "
                f"eta={eta/60:.1f} min", end="\r",
            )
    print()
    elapsed = time.time() - t0
    print(
        f"\nDone. Processed {len(all_labels)}/{len(eval_set)} files in "
        f"{elapsed:.1f}s ({len(all_labels)/elapsed:.1f} files/s)"
    )
    if errors:
        print(f"  {len(errors)} errors")

    # Metrics
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)
    speakers_arr = np.array(speakers_used)

    overall = compute_metrics(labels_arr, scores_arr, args.threshold)
    per_speaker: Dict[str, dict] = {}
    for spk in sorted(set(speakers_used)):
        mask = speakers_arr == spk
        per_speaker[spk] = compute_metrics(
            labels_arr[mask], scores_arr[mask], args.threshold,
        )

    # Print
    print("\n" + "=" * 78)
    print("IN-THE-WILD PERSONALISED (FiLM) EVALUATION RESULTS")
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

    print(f"\n{'Speaker':<35} {'EER':>8} {'AUC':>8} {'real n':>7} {'fake n':>7}")
    print("-" * 78)
    rows_sorted = sorted(
        per_speaker.items(),
        key=lambda x: (np.isnan(x[1].get("eer", float("nan"))),
                       x[1].get("eer", 1.0)),
    )
    for spk, m in rows_sorted[:30]:
        eer = m.get("eer", float("nan"))
        auc = m.get("auc", float("nan"))
        eer_disp = f"{eer*100:>7.2f}%" if not np.isnan(eer) else "    n/a "
        auc_disp = f"{auc:>8.4f}" if not np.isnan(auc) else "     n/a"
        print(
            f"{spk[:34]:<35} {eer_disp} {auc_disp} "
            f"{m['n_neg']:>7} {m['n_pos']:>7}"
        )
    if len(rows_sorted) > 30:
        print(f"  ... ({len(rows_sorted) - 30} more speakers in JSON)")
    print("=" * 78)

    # Save
    if args.output:
        out = {
            "api_url": args.api_url,
            "model_type": health.get("model_type"),
            "dataset": "in_the_wild",
            "dataset_dir": str(dataset_dir),
            "personalised": True,
            "enroll_k": args.enroll_k,
            "speaker_prefix": args.speaker_prefix,
            "max_per_class": args.max_per_class,
            "threshold": args.threshold,
            "speakers_in_dataset": len(by_speaker),
            "speakers_enrolled": len(enroll_results),
            "speakers_skipped": skipped,
            "enroll_errors": enroll_errors,
            "total_processed": len(all_labels),
            "total_errors": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "overall_metrics": overall,
            "per_speaker_metrics": per_speaker,
            "errors": errors[:50],
        }
        Path(args.output).expanduser().write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
