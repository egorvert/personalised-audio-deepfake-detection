# Stream the ajaykarthick/wavefake-audio HuggingFace dataset (a re-upload of
# WaveFake by Frank & Schönherr, 2021), sample a balanced bona-fide/spoof
# subset, encode each clip as WAV bytes, and POST it to the VDetect API.
#
# The HF re-upload only carries {audio, audio_id, real_or_fake}, so per-vocoder
# breakdown isn't available — we report overall EER/AUC only.
#
# Usage:
#     python wavefake_hf_eval_client.py \
#         --api-url http://127.0.0.1:8000 \
#         --max-per-class 2000 \
#         --output results_wavefake_2k.json

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import soundfile as sf
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


SPOOF_LABELS = {"spoof", "fake", "deepfake"}
BONAFIDE_LABELS = {"bona-fide", "bonafide", "real", "genuine"}


def normalise_label(raw: str) -> Optional[int]:
    s = (raw or "").strip().lower()
    if s in SPOOF_LABELS:
        return 1
    if s in BONAFIDE_LABELS:
        return 0
    return None


def encode_wav_bytes(array: np.ndarray, sampling_rate: int) -> bytes:
    """Encode a float audio array to 16-bit PCM WAV bytes."""
    if array.ndim > 1:
        # downmix to mono
        array = array.mean(axis=-1)
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


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
        "score_mean_real": (
            float(scores[labels == 0].mean())
            if (labels == 0).any() else float("nan")
        ),
        "score_mean_fake": (
            float(scores[labels == 1].mean())
            if (labels == 1).any() else float("nan")
        ),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WaveFake (HF stream) evaluation via VDetect API",
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dataset", default="ajaykarthick/wavefake-audio")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--shuffle-buffer", type=int, default=10000,
        help="HF streaming shuffle buffer size",
    )
    parser.add_argument(
        "--max-stream-pulls", type=int, default=200000,
        help="Safety cap on examples pulled while sampling",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # --- API health ---------------------------------------------------------
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
        print(f"ERROR: API unreachable: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Streaming load -----------------------------------------------------
    print(f"\nStreaming {args.dataset} (split={args.split}) ...")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    # --- Sample balanced ----------------------------------------------------
    sampled_real: List[dict] = []
    sampled_fake: List[dict] = []
    label_counts = {"real": 0, "fake": 0, "other": 0}
    print(f"Sampling {args.max_per_class} per class (balanced)...")
    pulled = 0
    for example in ds:
        pulled += 1
        if pulled > args.max_stream_pulls:
            print(
                f"\nWARN: stream cap {args.max_stream_pulls} reached",
                file=sys.stderr,
            )
            break
        label = normalise_label(example.get("real_or_fake", ""))
        if label is None:
            label_counts["other"] += 1
            continue
        if label == 0 and len(sampled_real) < args.max_per_class:
            sampled_real.append(example)
            label_counts["real"] += 1
        elif label == 1 and len(sampled_fake) < args.max_per_class:
            sampled_fake.append(example)
            label_counts["fake"] += 1
        if (
            len(sampled_real) >= args.max_per_class
            and len(sampled_fake) >= args.max_per_class
        ):
            break
        if pulled % 200 == 0:
            print(
                f"  pulled={pulled}  real={len(sampled_real)}  "
                f"fake={len(sampled_fake)}",
                end="\r",
            )
    print()
    print(
        f"Stream pulled {pulled} examples. "
        f"Selected: real={len(sampled_real)}, fake={len(sampled_fake)}, "
        f"unrecognised_label={label_counts['other']}"
    )

    examples_to_eval = (
        [(0, e) for e in sampled_real] + [(1, e) for e in sampled_fake]
    )
    random.shuffle(examples_to_eval)

    if not examples_to_eval:
        print("ERROR: no usable examples sampled.", file=sys.stderr)
        sys.exit(1)

    # --- Send to API --------------------------------------------------------
    all_labels: List[int] = []
    all_scores: List[float] = []
    audio_ids: List[str] = []
    errors: List[str] = []

    t0 = time.time()
    print(f"\nSending {len(examples_to_eval)} examples to API...")
    for i, (label, example) in enumerate(examples_to_eval, 1):
        try:
            audio = example["audio"]
            array = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            wav_bytes = encode_wav_bytes(array, sr)
            audio_id = example.get("audio_id", f"sample{i}")
            result = send_bytes(
                args.api_url, wav_bytes, audio_id,
                args.threshold, args.timeout,
            )
            all_labels.append(label)
            all_scores.append(result["score"])
            audio_ids.append(audio_id)
        except Exception as e:
            errors.append(f"{example.get('audio_id', '?')}/{label}: {e}")
        if i % 50 == 0 or i == len(examples_to_eval):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(examples_to_eval) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(examples_to_eval)}]  {rate:.1f} files/s  "
                f"eta={eta/60:.1f} min", end="\r",
            )
    print()

    elapsed = time.time() - t0
    print(
        f"\nDone. Processed {len(all_labels)}/{len(examples_to_eval)} files "
        f"in {elapsed:.1f}s ({len(all_labels)/elapsed:.1f} files/s)"
    )
    if errors:
        print(f"  {len(errors)} errors (first few in output JSON)")

    # --- Metrics ------------------------------------------------------------
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)
    overall = compute_metrics(labels_arr, scores_arr, args.threshold)

    print("\n" + "=" * 78)
    print("WAVEFAKE (HF stream) EVALUATION RESULTS")
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
    print("=" * 78)

    if args.output:
        out = {
            "api_url": args.api_url,
            "model_type": health.get("model_type"),
            "dataset": args.dataset,
            "split": args.split,
            "max_per_class": args.max_per_class,
            "threshold": args.threshold,
            "label_counts_during_sampling": label_counts,
            "stream_pulled": pulled,
            "total_processed": len(all_labels),
            "total_errors": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "overall_metrics": overall,
            "errors": errors[:50],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
