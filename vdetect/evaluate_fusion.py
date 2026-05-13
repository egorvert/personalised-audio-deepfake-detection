import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from vdetect.models.wavlm_baseline import WavLMDetector
from vdetect.models.aasist_encoder import AASISTEncoder, pad_or_crop_for_aasist
from vdetect.models.two_stream import TwoStreamDetector
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
    }


def evaluate_wavlm(dataloader: DataLoader, weights_path: str, device: str) -> Dict[str, Any]:
    print("\nEvaluating WavLM baseline")
    model = WavLMDetector().to(device)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.train(False)

    labels_all: list = []
    scores_all: list = []
    with torch.no_grad():
        for wav, labels in dataloader:
            scores_all.extend(model(wav.to(device))["score"].cpu().tolist())
            labels_all.extend(labels.tolist())

    metrics = compute_metrics(labels_all, scores_all, checkpoint.get("threshold", 0.5))
    metrics["model"] = "WavLM"
    return metrics


def evaluate_aasist(dataloader: DataLoader, weights_path: Optional[str], device: str) -> Dict[str, Any]:
    print("\nEvaluating pretrained AASIST")
    encoder = AASISTEncoder(pretrained_path=weights_path, freeze=True).to(device)
    encoder.train(False)

    labels_all: list = []
    scores_all: list = []
    with torch.no_grad():
        for wav, labels in dataloader:
            wav = pad_or_crop_for_aasist(wav, target_length=64600).to(device)
            scores_all.extend(encoder.get_score(wav).cpu().tolist())
            labels_all.extend(labels.tolist())

    metrics = compute_metrics(labels_all, scores_all, threshold=0.5)
    metrics["model"] = "AASIST"
    return metrics


def evaluate_fusion(
    dataloader: DataLoader,
    weights_path: str,
    wavlm_path: Optional[str],
    aasist_path: Optional[str],
    device: str,
) -> Dict[str, Any]:
    print("\nEvaluating two-stream fusion")
    model = TwoStreamDetector(
        wavlm_checkpoint=wavlm_path,
        aasist_checkpoint=aasist_path,
        freeze_wavlm=True,
        freeze_aasist=True,
    ).to(device)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.train(False)

    labels_all: list = []
    scores_all: list = []
    with torch.no_grad():
        for wav, labels in dataloader:
            scores_all.extend(model(wav.to(device))["score"].cpu().tolist())
            labels_all.extend(labels.tolist())

    metrics = compute_metrics(labels_all, scores_all, checkpoint.get("threshold", 0.5))
    metrics["model"] = "Fusion"
    return metrics


def print_comparison(results: list) -> None:
    print("\nModel comparison")
    print("-" * 60)
    print(f"{'Model':<10} {'EER':>10} {'AUC':>10} {'Accuracy':>10} {'F1':>10}")
    for r in results:
        print(
            f"{r['model']:<10} {r['eer']*100:>9.2f}% {r['auc']:>10.4f} "
            f"{r['accuracy']*100:>9.2f}% {r['f1']:>10.4f}"
        )
    best = min(results, key=lambda x: x["eer"])
    print(f"\nBest model by EER: {best['model']} ({best['eer']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compare WavLM, AASIST, and Fusion models")
    parser.add_argument("--data-root", type=str, required=True, help="ASVspoof2019 LA root")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "eval"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wavlm-weights", type=str, default="assets/checkpoints/wavlm_baseline.pt")
    parser.add_argument("--aasist-weights", type=str, default=None)
    parser.add_argument("--fusion-weights", type=str, default="assets/checkpoints/two_stream.pt")
    parser.add_argument("--models", type=str, default="all", choices=["all", "wavlm", "aasist", "fusion"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    proto_file = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }[args.split]
    proto_path = data_root / "ASVspoof2019_LA_cm_protocols" / proto_file
    for p in (data_root, proto_path):
        if not p.exists():
            raise FileNotFoundError(p)

    device = get_device()
    print(f"Using device: {device}")

    # Use 64600 samples so the same loader works for AASIST/fusion (which require it).
    dataset = ASVspoofLADataset(
        data_root=data_root, protocol_path=proto_path, split=args.split,
        max_length_samples=64600,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    print(f"Evaluating on {len(dataset)} samples...")

    results: list = []

    if args.models in ("all", "wavlm"):
        if Path(args.wavlm_weights).exists():
            results.append(evaluate_wavlm(dataloader, args.wavlm_weights, device))
            print(f"WavLM EER: {results[-1]['eer']*100:.2f}%")
        else:
            print(f"Warning: WavLM weights not found at {args.wavlm_weights}")

    if args.models in ("all", "aasist"):
        results.append(evaluate_aasist(dataloader, args.aasist_weights, device))
        print(f"AASIST EER: {results[-1]['eer']*100:.2f}%")

    if args.models in ("all", "fusion"):
        if Path(args.fusion_weights).exists():
            results.append(evaluate_fusion(
                dataloader, args.fusion_weights,
                args.wavlm_weights if Path(args.wavlm_weights).exists() else None,
                args.aasist_weights, device,
            ))
            print(f"Fusion EER: {results[-1]['eer']*100:.2f}%")
        else:
            print(f"Warning: Fusion weights not found at {args.fusion_weights}")

    if len(results) > 1:
        print_comparison(results)
    elif len(results) == 1:
        r = results[0]
        print(f"\n{r['model']} results: EER={r['eer']*100:.2f}%  AUC={r['auc']:.4f}  "
              f"Acc={r['accuracy']*100:.2f}%  F1={r['f1']:.4f}")


if __name__ == "__main__":
    main()
