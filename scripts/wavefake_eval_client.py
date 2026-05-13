# Send WaveFake clips to a running VDetect API and compute EER / AUC / accuracy / F1.
#
# Expects:
#   --wavefake-dir: root of the WaveFake generated_audio folder, with one
#                   subfolder per architecture (ljspeech_melgan, ljspeech_waveglow, ...)
#   --bonafide-dir: directory of real audio (e.g. the LJSpeech WAVs)
#
# Usage:
#     python wavefake_eval_client.py \
#         --api-url http://<tailscale-ip>:8000 \
#         --wavefake-dir ./wavefake_dataset \
#         --bonafide-dir ./LJSpeech-1.1/wavs \
#         --max-per-class 1000 \
#         --output results_wavefake.json

import argparse
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


def discover_audio_files(
    directory: Path,
    extensions: tuple = (".wav", ".flac", ".mp3"),
) -> List[Path]:
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def sample_files(files: List[Path], max_count: int) -> List[Path]:
    if max_count <= 0 or len(files) <= max_count:
        return files
    return sorted(random.sample(files, max_count))


def discover_wavefake_classes(wavefake_dir: Path) -> Dict[str, List[Path]]:
    """Map each architecture subfolder to its audio files."""
    classes = {}
    for subdir in sorted(wavefake_dir.iterdir()):
        if not subdir.is_dir():
            continue
        files = discover_audio_files(subdir)
        if files:
            classes[subdir.name] = files
    return classes


def send_file(api_url: str, file_path: Path, threshold: float = 0.5) -> dict:
    url = f"{api_url.rstrip('/')}/detect"
    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            files={"audio": (file_path.name, f)},
            data={"threshold": str(threshold)},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    predictions = (scores >= threshold).astype(int)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    abs_diff = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diff)
    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    eer_threshold = float(thresholds[min_idx]) if min_idx < len(thresholds) else 0.5

    auc = float(roc_auc_score(labels, scores))

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def print_results(overall: Dict, per_arch: Dict[str, Dict]):
    print("\n" + "=" * 72)
    print("WAVEFAKE CROSS-DATASET EVALUATION RESULTS")
    print("=" * 72)

    print(f"\n{'Overall':>20}  EER={overall['eer']*100:6.2f}%  "
          f"AUC={overall['auc']:.4f}  Acc={overall['accuracy']*100:.1f}%  "
          f"F1={overall['f1']:.4f}")

    if per_arch:
        print(f"\n{'Architecture':<35} {'EER':>8} {'AUC':>8} {'Samples':>8}")
        print("-" * 72)
        for arch, m in sorted(per_arch.items(), key=lambda x: x[1]["eer"]):
            print(f"{arch:<35} {m['eer']*100:>7.2f}% {m['auc']:>8.4f} {m['n_spoof']:>8}")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Remote WaveFake evaluation via VDetect API",
    )
    parser.add_argument(
        "--api-url", required=True,
        help="Base URL of the VDetect API (e.g. http://100.x.y.z:8000)",
    )
    parser.add_argument(
        "--wavefake-dir", required=True, type=Path,
        help="Root of the WaveFake generated_audio directory",
    )
    parser.add_argument(
        "--bonafide-dir", required=True, type=Path,
        help="Directory of bonafide/real audio files (e.g. LJSpeech wavs)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Max files to sample per architecture (0 = use all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold for accuracy/F1 metrics",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Health check -------------------------------------------------------
    print(f"Checking API at {args.api_url} ...")
    try:
        health = requests.get(f"{args.api_url.rstrip('/')}/health", timeout=10).json()
        print(f"  Model loaded: {health['model_loaded']} ({health.get('model_type', '?')})"
              f" on {health.get('device', '?')}")
    except Exception as e:
        print(f"ERROR: Cannot reach API — {e}", file=sys.stderr)
        sys.exit(1)

    # --- Discover files -----------------------------------------------------
    print(f"\nScanning WaveFake directory: {args.wavefake_dir}")
    arch_classes = discover_wavefake_classes(args.wavefake_dir)
    if not arch_classes:
        print("ERROR: No architecture subfolders found.", file=sys.stderr)
        sys.exit(1)

    for name, files in arch_classes.items():
        print(f"  {name}: {len(files)} files")

    print(f"\nScanning bonafide directory: {args.bonafide_dir}")
    bonafide_files = discover_audio_files(args.bonafide_dir)
    print(f"  bonafide: {len(bonafide_files)} files")

    # --- Sample if requested ------------------------------------------------
    if args.max_per_class > 0:
        print(f"\nSampling up to {args.max_per_class} per class ...")
        bonafide_files = sample_files(bonafide_files, args.max_per_class)
        for name in arch_classes:
            arch_classes[name] = sample_files(arch_classes[name], args.max_per_class)

    total_spoof = sum(len(f) for f in arch_classes.values())
    total = len(bonafide_files) + total_spoof
    print(f"\nTotal files to evaluate: {total} "
          f"({len(bonafide_files)} bonafide, {total_spoof} spoof)")

    # --- Send files to API --------------------------------------------------
    all_labels: List[int] = []
    all_scores: List[float] = []
    arch_labels: Dict[str, List[int]] = {}
    arch_scores: Dict[str, List[float]] = {}
    errors: List[str] = []

    processed = 0
    t0 = time.time()

    # Bonafide
    print("\nProcessing bonafide files ...")
    for i, fp in enumerate(bonafide_files, 1):
        try:
            result = send_file(args.api_url, fp, args.threshold)
            all_labels.append(0)
            all_scores.append(result["score"])
            processed += 1
        except Exception as e:
            errors.append(f"bonafide/{fp.name}: {e}")
        if i % 100 == 0 or i == len(bonafide_files):
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  [{i}/{len(bonafide_files)}]  {rate:.1f} files/s", end="\r")
    print()

    # Spoof (per architecture)
    for arch_name, files in arch_classes.items():
        print(f"Processing {arch_name} ({len(files)} files) ...")
        arch_labels[arch_name] = []
        arch_scores[arch_name] = []

        for i, fp in enumerate(files, 1):
            try:
                result = send_file(args.api_url, fp, args.threshold)
                all_labels.append(1)
                all_scores.append(result["score"])
                arch_labels[arch_name].append(1)
                arch_scores[arch_name].append(result["score"])
                processed += 1
            except Exception as e:
                errors.append(f"{arch_name}/{fp.name}: {e}")
            if i % 100 == 0 or i == len(files):
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(files)}]  {rate:.1f} files/s", end="\r")
        print()

    elapsed = time.time() - t0
    print(f"\nDone. Processed {processed}/{total} files in {elapsed:.1f}s "
          f"({processed/elapsed:.1f} files/s)")
    if errors:
        print(f"  {len(errors)} errors (see output JSON for details)")

    # --- Compute metrics ----------------------------------------------------
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)

    overall = compute_metrics(labels_arr, scores_arr, args.threshold)

    per_arch = {}
    for arch_name in arch_classes:
        if not arch_scores.get(arch_name):
            continue
        bonafide_labels = np.zeros(len(bonafide_files))
        bonafide_sc = scores_arr[:len(bonafide_files)]
        arch_l = np.array(arch_labels[arch_name])
        arch_s = np.array(arch_scores[arch_name])
        combined_labels = np.concatenate([bonafide_labels, arch_l])
        combined_scores = np.concatenate([bonafide_sc, arch_s])
        m = compute_metrics(combined_labels, combined_scores, args.threshold)
        m["n_spoof"] = len(arch_scores[arch_name])
        per_arch[arch_name] = m

    print_results(overall, per_arch)

    # --- Save ---------------------------------------------------------------
    if args.output:
        out = {
            "api_url": args.api_url,
            "model_type": health.get("model_type"),
            "wavefake_dir": str(args.wavefake_dir),
            "bonafide_dir": str(args.bonafide_dir),
            "max_per_class": args.max_per_class,
            "threshold": args.threshold,
            "total_processed": processed,
            "total_errors": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "overall_metrics": overall,
            "per_architecture_metrics": per_arch,
            "errors": errors[:50],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
