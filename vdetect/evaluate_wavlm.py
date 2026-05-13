import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from vdetect.models.wavlm_baseline import WavLMDetector
from vdetect.data.asvspoof2019_la import ASVspoofLADataset


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_metrics(labels: list, scores: list, threshold: float = 0.5) -> Dict[str, Any]:
    labels = np.array(labels)
    scores = np.array(scores)
    predictions = (scores >= threshold).astype(int)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    min_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    eer_threshold = float(thresholds[min_idx]) if min_idx < len(thresholds) else 0.5

    auc = float(roc_auc_score(labels, scores))
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr_at = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_at = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "fpr_at_threshold": fpr_at,
        "fnr_at_threshold": fnr_at,
        "threshold": threshold,
    }


def run_inference(model: torch.nn.Module, dataloader: DataLoader, device: str) -> tuple:
    model.train(False)
    all_labels: list = []
    all_scores: list = []
    with torch.no_grad():
        for wav, labels in dataloader:
            outputs = model(wav.to(device))
            all_scores.extend(outputs["score"].cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate WavLM baseline detector")
    parser.add_argument("--data-root", type=str, required=True, help="ASVspoof2019 LA root")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "eval"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification threshold; defaults to the one stored in the checkpoint, else 0.5")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    weights_path = Path(args.weights)
    proto_file = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }[args.split]
    proto_path = data_root / "ASVspoof2019_LA_cm_protocols" / proto_file

    for p in (data_root, weights_path, proto_path):
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"Loading checkpoint from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location="cpu")
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)
    print(f"Using threshold: {threshold:.4f}")

    dataset = ASVspoofLADataset(
        data_root=data_root, protocol_path=proto_path, split=args.split,
        max_length_samples=64000,
    )

    device = get_device()
    print(f"Using device: {device}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    model = WavLMDetector().to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.train(False)

    print(f"Evaluating on {len(dataset)} samples...")
    labels, scores = run_inference(model, dataloader, device)
    metrics = compute_metrics(labels, scores, threshold)

    print("\n" + "=" * 60)
    print(f"Results ({args.split} set)")
    print("=" * 60)
    print(f"EER:              {metrics['eer']:.4f} (at threshold {metrics['eer_threshold']:.4f})")
    print(f"AUC:              {metrics['auc']:.4f}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"\nAt threshold {threshold:.4f}:")
    print(f"  FPR:            {metrics['fpr_at_threshold']:.4f}")
    print(f"  FNR:            {metrics['fnr_at_threshold']:.4f}")
    print("\nConfusion matrix:")
    print(f"  TN: {metrics['true_negatives']:6d}   FP: {metrics['false_positives']:6d}")
    print(f"  FN: {metrics['false_negatives']:6d}   TP: {metrics['true_positives']:6d}")


if __name__ == "__main__":
    main()

