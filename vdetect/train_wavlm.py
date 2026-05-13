import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from vdetect.models.wavlm_baseline import WavLMDetector
from vdetect.data.asvspoof2019_la import ASVspoofLADataset


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_eer(labels: list, scores: list) -> Tuple[float, float]:
    # EER is the threshold where false-positive rate and false-negative rate cross.
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    min_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
    threshold = thresholds[min_idx] if min_idx < len(thresholds) else 0.5
    return float(eer), float(threshold)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for wav, labels in dataloader:
        wav = wav.to(device)
        labels = labels.to(device)
        outputs = model(wav)
        loss = criterion(outputs["logit"], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def run_eval(model: nn.Module, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    model.train(False)
    all_scores: list = []
    all_labels: list = []
    with torch.no_grad():
        for wav, labels in dataloader:
            outputs = model(wav.to(device))
            all_scores.extend(outputs["score"].cpu().tolist())
            all_labels.extend(labels.tolist())
    return compute_eer(all_labels, all_scores)


def main():
    parser = argparse.ArgumentParser(description="Train WavLM baseline detector")
    parser.add_argument("--data-root", type=str, required=True, help="Root of ASVspoof2019 LA data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out", type=str, default="assets/checkpoints/wavlm_baseline.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="microsoft/wavlm-base-plus")
    parser.add_argument("--freeze", action="store_true", default=True, help="Freeze WavLM backbone")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    proto_train = data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    proto_dev = data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    for p in (data_root, proto_train, proto_dev):
        if not p.exists():
            raise FileNotFoundError(p)

    print("Loading datasets...")
    train_dataset = ASVspoofLADataset(data_root, proto_train, split="train", max_length_samples=64000)
    dev_dataset = ASVspoofLADataset(data_root, proto_dev, split="dev", max_length_samples=64000)

    device = get_device()
    print(f"Using device: {device}")

    # pin_memory is a CUDA-only optimisation; on MPS or CPU it just slows things down.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    print(f"Loading model: {args.model_name}")
    model = WavLMDetector(model_name=args.model_name, freeze=args.freeze).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}  trainable: {trainable:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )

    best_eer = 1.0
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        dev_eer, dev_threshold = run_eval(model, dev_loader, device)
        print(f"Dev EER: {dev_eer:.4f} (threshold: {dev_threshold:.4f})")

        # Keep only the best checkpoint by dev EER.
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_threshold = dev_threshold

            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "eer": best_eer,
                    "threshold": best_threshold,
                    "epoch": epoch,
                    "args": vars(args),
                },
                out_path,
            )
            print(f"Saved checkpoint to {out_path}")

    print("\n" + "=" * 50)
    print("Training complete.")
    print(f"Best dev EER: {best_eer:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()

