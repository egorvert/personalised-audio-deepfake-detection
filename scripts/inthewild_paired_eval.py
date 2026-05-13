# Paired In-the-Wild evaluation. Each sampled file is scored twice via the
# VDetect API — once without speaker_id (FiLM bypassed) and once with it (FiLM
# active). Call order is randomised per file so any temporal drift in API
# state cannot systematically bias one mode.
#
# Pairing removes the sampling-noise confound between separate non-personalised
# and personalised runs (which used different clips). With the same clips
# scored both ways, the delta is attributable to FiLM alone.
#
# Per-speaker bootstrap 95% CIs on the EER delta are produced by resampling
# the speaker's paired records with replacement.
#
# Usage:
#     python inthewild_paired_eval.py \
#         --api-url http://127.0.0.1:8000 \
#         --dataset-dir ~/datasets/inthewild/release_in_the_wild \
#         --max-per-class 2000 \
#         --enroll-k 3 --speaker-prefix itw_ \
#         --output ~/datasets/inthewild/results_inthewild_paired_2k.json

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


SPOOF_LABELS = {"spoof", "fake", "deepfake"}
BONAFIDE_LABELS = {"bona-fide", "bonafide", "real", "genuine"}


# Metadata + grouping


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
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return prefix + "".join(keep)


# API helpers


def enroll_speaker(
    api_url: str,
    speaker_id: str,
    audio_paths: List[Path],
    timeout_s: int = 120,
) -> dict:
    url = f"{api_url.rstrip('/')}/enroll"
    files: List[Tuple[str, Tuple[str, object, str]]] = []
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


def detect_once(
    api_url: str,
    file_path: Path,
    speaker_id: Optional[str],
    threshold: float,
    timeout_s: int,
) -> dict:
    """Single /detect call, with or without speaker_id."""
    url = f"{api_url.rstrip('/')}/detect"
    data = {"threshold": str(threshold)}
    if speaker_id is not None:
        data["speaker_id"] = speaker_id
    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            files={"audio": (file_path.name, f)},
            data=data,
            timeout=timeout_s,
        )
    resp.raise_for_status()
    return resp.json()


# Metrics


def eer_only(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


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
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
        }
    fpr, tpr, thresh = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresh[idx]) if idx < len(thresh) else 0.5
    auc = float(roc_auc_score(labels, scores))
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "score_mean_real": (float(scores[labels == 0].mean())
                            if (labels == 0).any() else float("nan")),
        "score_mean_fake": (float(scores[labels == 1].mean())
                            if (labels == 1).any() else float("nan")),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def bootstrap_delta_eer_ci(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_iter: int = 1000,
    seed: int = 1234,
) -> Tuple[float, float, float, float, float]:
    """Bootstrap 95% CI on (EER_b - EER_a). Returns
    (mean_delta, lo, hi, mean_eer_a, mean_eer_b)."""
    n = len(labels)
    if n == 0 or len(set(labels.tolist())) < 2:
        return (float("nan"),) * 5
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_iter, dtype=np.float64)
    eer_a_arr = np.empty(n_iter, dtype=np.float64)
    eer_b_arr = np.empty(n_iter, dtype=np.float64)
    valid = 0
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        l = labels[idx]
        if len(set(l.tolist())) < 2:
            continue
        ea = eer_only(l, scores_a[idx])
        eb = eer_only(l, scores_b[idx])
        if math.isnan(ea) or math.isnan(eb):
            continue
        eer_a_arr[valid] = ea
        eer_b_arr[valid] = eb
        deltas[valid] = eb - ea
        valid += 1
    if valid < 50:
        return (float("nan"),) * 5
    deltas = deltas[:valid]
    eer_a_arr = eer_a_arr[:valid]
    eer_b_arr = eer_b_arr[:valid]
    return (
        float(deltas.mean()),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
        float(eer_a_arr.mean()),
        float(eer_b_arr.mean()),
    )


# Sampling


def round_robin_sample(
    rows_by_speaker: Dict[str, List[Dict]],
    n_target: int,
    rng: random.Random,
) -> List[Dict]:
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


# Main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired In-the-Wild eval (FiLM vs no-FiLM, same files)",
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--enroll-k", type=int, default=3)
    parser.add_argument("--speaker-prefix", default="itw_")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--skip-enroll", action="store_true",
                        help="Assume speakers are already enrolled.")
    args = parser.parse_args()

    if args.enroll_k < 3 or args.enroll_k > 5:
        print("ERROR: --enroll-k must be in [3,5]", file=sys.stderr)
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
    rows_with_path: List[Dict] = []
    missing = 0
    for r in rows:
        p = dataset_dir / r["file"]
        if not p.exists():
            missing += 1
            continue
        rows_with_path.append({**r, "path": p})
    if missing:
        print(f"WARN: {missing} files missing on disk", file=sys.stderr)

    by_speaker: Dict[str, List[Dict]] = {}
    for r in sorted(rows_with_path, key=lambda x: x["file"]):
        by_speaker.setdefault(r["speaker"], []).append(r)

    # Choose enrolment + eval pools
    enrolment_per_speaker: Dict[str, List[Dict]] = {}
    eligible_real: Dict[str, List[Dict]] = {}
    eligible_fake: Dict[str, List[Dict]] = {}
    skipped: Dict[str, str] = {}
    for spk, rs in by_speaker.items():
        reals = [r for r in rs if is_real(r)]
        fakes = [r for r in rs if is_fake(r)]
        if len(reals) < args.enroll_k:
            skipped[spk] = f"only {len(reals)} bonafide samples"
            continue
        enrolment_per_speaker[spk] = reals[:args.enroll_k]
        remaining = reals[args.enroll_k:]
        if not remaining and not fakes:
            skipped[spk] = "no eval material"
            continue
        if remaining:
            eligible_real[spk] = remaining
        if fakes:
            eligible_fake[spk] = fakes

    print(f"  speakers in dataset: {len(by_speaker)}; "
          f"will enrol: {len(enrolment_per_speaker)}; "
          f"skipped: {len(skipped)}")

    # Enrol
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
                res = enroll_speaker(args.api_url, sid, paths)
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

    # Drop speakers without successful enrolment from eval pools
    if not args.skip_enroll:
        good = set(enroll_results.keys())
        eligible_real = {s: rs for s, rs in eligible_real.items() if s in good}
        eligible_fake = {s: rs for s, rs in eligible_fake.items() if s in good}

    # Sample one balanced eval pool — fixed for both modes
    n_real = min(args.max_per_class,
                 sum(len(v) for v in eligible_real.values()))
    n_fake = min(args.max_per_class,
                 sum(len(v) for v in eligible_fake.values()))
    print(f"\nSampling {n_real} real and {n_fake} fake (balanced)...")
    real_samples = round_robin_sample(eligible_real, n_real, rng)
    fake_samples = round_robin_sample(eligible_fake, n_fake, rng)

    eval_set: List[Tuple[int, Dict]] = (
        [(0, r) for r in real_samples] + [(1, r) for r in fake_samples]
    )
    rng.shuffle(eval_set)

    # Per-pair coin flip: which mode to call first
    flips = [rng.choice([0, 1]) for _ in eval_set]

    # Paired detection
    print(f"\nPaired detection: {len(eval_set)} files × 2 modes = "
          f"{2 * len(eval_set)} API calls")
    paired_records: List[Dict] = []
    errors: List[str] = []
    t0 = time.time()
    for i, ((label, row), first_mode) in enumerate(zip(eval_set, flips), 1):
        sid = safe_speaker_id(args.speaker_prefix, row["speaker"])
        try:
            if first_mode == 0:
                # no-FiLM first, then with FiLM
                r0 = detect_once(args.api_url, row["path"], None,
                                 args.threshold, args.timeout)
                r1 = detect_once(args.api_url, row["path"], sid,
                                 args.threshold, args.timeout)
            else:
                r1 = detect_once(args.api_url, row["path"], sid,
                                 args.threshold, args.timeout)
                r0 = detect_once(args.api_url, row["path"], None,
                                 args.threshold, args.timeout)
            paired_records.append({
                "speaker": row["speaker"],
                "file": row["file"],
                "label": int(label),
                "score_nofilm": float(r0["score"]),
                "score_film": float(r1["score"]),
                "first_mode": int(first_mode),
            })
        except Exception as e:
            errors.append(f"{row['speaker']}/{row['file']}: {e}")
        if i % 25 == 0 or i == len(eval_set):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(eval_set) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(eval_set)}] {rate:.1f} pairs/s "
                f"eta={eta/60:.1f} min", end="\r",
            )
    print()
    elapsed = time.time() - t0
    print(f"\nDone. {len(paired_records)} pairs in {elapsed:.1f}s "
          f"({len(paired_records)/elapsed:.1f} pairs/s)")
    if errors:
        print(f"  {len(errors)} errors")

    # Aggregate metrics
    labels = np.array([p["label"] for p in paired_records], dtype=np.int64)
    scores_no = np.array(
        [p["score_nofilm"] for p in paired_records], dtype=np.float64,
    )
    scores_w = np.array(
        [p["score_film"] for p in paired_records], dtype=np.float64,
    )
    speakers = np.array([p["speaker"] for p in paired_records])

    overall_no = compute_metrics(labels, scores_no, args.threshold)
    overall_w = compute_metrics(labels, scores_w, args.threshold)
    delta_eer = overall_w["eer"] - overall_no["eer"]
    delta_auc = overall_w["auc"] - overall_no["auc"]

    # Bootstrap CI on overall ΔEER
    delta_mean, delta_lo, delta_hi, _, _ = bootstrap_delta_eer_ci(
        labels, scores_no, scores_w,
        n_iter=args.bootstrap_iters, seed=args.seed,
    )

    # Per-speaker
    per_speaker: Dict[str, dict] = {}
    for spk in sorted(set(speakers.tolist())):
        mask = speakers == spk
        if not mask.any():
            continue
        l = labels[mask]
        s_no = scores_no[mask]
        s_w = scores_w[mask]
        m_no = compute_metrics(l, s_no, args.threshold)
        m_w = compute_metrics(l, s_w, args.threshold)
        boot = bootstrap_delta_eer_ci(
            l, s_no, s_w, n_iter=args.bootstrap_iters, seed=args.seed,
        )
        per_speaker[spk] = {
            "no_film": m_no,
            "with_film": m_w,
            "delta_eer_point": (
                m_w["eer"] - m_no["eer"]
                if not (math.isnan(m_no.get("eer", float("nan")))
                        or math.isnan(m_w.get("eer", float("nan"))))
                else float("nan")
            ),
            "delta_eer_bootstrap_mean": boot[0],
            "delta_eer_bootstrap_lo95": boot[1],
            "delta_eer_bootstrap_hi95": boot[2],
            "delta_score_mean_real": (
                m_w.get("score_mean_real", float("nan"))
                - m_no.get("score_mean_real", float("nan"))
            ),
            "delta_score_mean_fake": (
                m_w.get("score_mean_fake", float("nan"))
                - m_no.get("score_mean_fake", float("nan"))
            ),
        }

    # Print summary
    print("\n" + "=" * 80)
    print("IN-THE-WILD PAIRED EVALUATION (no-FiLM vs FiLM, same files)")
    print("=" * 80)
    print(f"\n  no-FiLM   EER={overall_no['eer']*100:6.2f}%  "
          f"AUC={overall_no['auc']:.4f}  "
          f"mean(real)={overall_no.get('score_mean_real', 0):.4f}  "
          f"mean(fake)={overall_no.get('score_mean_fake', 0):.4f}")
    print(f"  with FiLM EER={overall_w['eer']*100:6.2f}%  "
          f"AUC={overall_w['auc']:.4f}  "
          f"mean(real)={overall_w.get('score_mean_real', 0):.4f}  "
          f"mean(fake)={overall_w.get('score_mean_fake', 0):.4f}")
    print(f"  Δ (FiLM − no): EER={delta_eer*100:+.2f}pp  AUC={delta_auc:+.4f}")
    print(f"  bootstrap ΔEER 95% CI: "
          f"[{delta_lo*100:+.2f}pp, {delta_hi*100:+.2f}pp]  "
          f"mean={delta_mean*100:+.2f}pp")
    print(f"  n_pairs={len(paired_records)}")

    # Per-speaker leaderboard
    rows_sorted = sorted(
        per_speaker.items(),
        key=lambda kv: (
            math.isnan(kv[1].get("delta_eer_point", float("nan"))),
            kv[1].get("delta_eer_point", 1.0),
        ),
    )
    print(f"\n{'Speaker':<28} {'EER_no':>7} {'EER_w':>7} "
          f"{'ΔEER':>8} {'CI95':>21} {'Δreal':>8} {'Δfake':>8} "
          f"{'n_r':>4} {'n_f':>4}")
    print("-" * 80)
    for spk, m in rows_sorted:
        eer_no = m["no_film"].get("eer", float("nan"))
        eer_w = m["with_film"].get("eer", float("nan"))
        d = m.get("delta_eer_point", float("nan"))
        lo = m.get("delta_eer_bootstrap_lo95", float("nan"))
        hi = m.get("delta_eer_bootstrap_hi95", float("nan"))
        dr = m.get("delta_score_mean_real", float("nan"))
        df = m.get("delta_score_mean_fake", float("nan"))
        n_r = m["no_film"].get("n_neg", 0)
        n_f = m["no_film"].get("n_pos", 0)

        def fmt_pct(x):
            return f"{x*100:>6.2f}%" if not math.isnan(x) else "    n/a"

        def fmt_pp(x):
            return f"{x*100:>+6.2f}pp" if not math.isnan(x) else "      n/a"

        ci_str = (
            f"[{lo*100:>+6.2f},{hi*100:>+6.2f}]"
            if not (math.isnan(lo) or math.isnan(hi)) else "         n/a"
        )
        print(
            f"{spk[:27]:<28} {fmt_pct(eer_no):>7} {fmt_pct(eer_w):>7} "
            f"{fmt_pp(d):>8} {ci_str:>21} "
            f"{(f'{dr:+.4f}' if not math.isnan(dr) else 'n/a'):>8} "
            f"{(f'{df:+.4f}' if not math.isnan(df) else 'n/a'):>8} "
            f"{n_r:>4} {n_f:>4}"
        )
    print("=" * 80)

    # Save
    if args.output:
        out = {
            "api_url": args.api_url,
            "model_type": health.get("model_type"),
            "dataset": "in_the_wild",
            "dataset_dir": str(dataset_dir),
            "design": "paired (same files, both modes, randomised order)",
            "enroll_k": args.enroll_k,
            "speaker_prefix": args.speaker_prefix,
            "max_per_class": args.max_per_class,
            "threshold": args.threshold,
            "bootstrap_iters": args.bootstrap_iters,
            "speakers_enrolled": len(enroll_results),
            "speakers_skipped": skipped,
            "enroll_errors": enroll_errors,
            "n_pairs": len(paired_records),
            "elapsed_seconds": round(elapsed, 1),
            "overall_no_film": overall_no,
            "overall_with_film": overall_w,
            "delta_eer_point": delta_eer,
            "delta_auc_point": delta_auc,
            "delta_eer_bootstrap_mean": delta_mean,
            "delta_eer_bootstrap_lo95": delta_lo,
            "delta_eer_bootstrap_hi95": delta_hi,
            "per_speaker": per_speaker,
            "paired_records": paired_records,
            "errors": errors[:50],
        }
        Path(args.output).expanduser().write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
