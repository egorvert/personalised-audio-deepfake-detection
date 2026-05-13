import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from vdetect.models.two_stream import TwoStreamDetector
from vdetect.data.asvspoof2019_la import ASVspoofLADataset


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_eer(labels: list, scores: list) -> Tuple[float, float]:
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
    grad_clip: float = 1.0,
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
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def score_split(model: nn.Module, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    model.train(False)
    all_scores: list = []
    all_labels: list = []
    with torch.no_grad():
        for wav, labels in dataloader:
            outputs = model(wav.to(device))
            all_scores.extend(outputs["score"].cpu().tolist())
            all_labels.extend(labels.tolist())
    return compute_eer(all_labels, all_scores)


def set_personalization_trainable(model: nn.Module, include_fusion: bool = True) -> None:
    # Freeze everything except FiLM (and optionally the fusion head).
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "film"):
        for param in model.film.parameters():
            param.requires_grad = True
    for param in model.fusion_head.parameters():
        param.requires_grad = include_fusion


def build_speaker_index(
    dataset: ASVspoofLADataset,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    # Maps speaker -> sample indices; second map keeps only bonafide indices (used as support).
    if not getattr(dataset, "return_speaker_id", False):
        raise ValueError("Dataset must be created with return_speaker_id=True")

    speaker_to_all: Dict[str, List[int]] = {}
    speaker_to_bonafide: Dict[str, List[int]] = {}
    for idx, item in enumerate(dataset.items):
        label = item[2]
        speaker_id = item[3]
        speaker_to_all.setdefault(speaker_id, []).append(idx)
        if label == 0:
            speaker_to_bonafide.setdefault(speaker_id, []).append(idx)
    return speaker_to_all, speaker_to_bonafide


def sample_episode_speakers(
    eligible_speakers: List[str],
    episode_speakers: int,
    rng: random.Random,
) -> List[str]:
    return rng.sample(eligible_speakers, episode_speakers)


def train_episode(
    model: nn.Module,
    dataset: ASVspoofLADataset,
    speaker_to_all: Dict[str, List[int]],
    speaker_to_bonafide: Dict[str, List[int]],
    speakers: List[str],
    support_shots: int,
    query_shots: int,
    device: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    normalize_proto: bool,
    rng: random.Random,
) -> float:
    # One episodic step: build a speaker prototype from support clips, score query clips with FiLM, backprop.
    model.train()
    losses = []

    for speaker_id in speakers:
        support_pool = speaker_to_bonafide[speaker_id]
        all_pool = speaker_to_all[speaker_id]

        support_idx = rng.sample(support_pool, support_shots)
        remaining = [idx for idx in all_pool if idx not in support_idx]
        query_idx = rng.sample(remaining, query_shots)

        support_wavs = [dataset[idx][0] for idx in support_idx]
        support_batch = torch.stack(support_wavs).to(device)
        with torch.no_grad():
            proto = model.extract_embedding(support_batch).mean(dim=0)
        if normalize_proto:
            proto = proto / (proto.norm(p=2) + 1e-8)

        query_wavs = []
        query_labels = []
        for idx in query_idx:
            wav, label, _ = dataset[idx]
            query_wavs.append(wav)
            query_labels.append(label)

        query_batch = torch.stack(query_wavs).to(device)
        labels = torch.stack(query_labels).to(device)
        outputs = model(query_batch, speaker_proto=proto)
        losses.append(criterion(outputs["logit"], labels))

    episode_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    episode_loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(episode_loss.item())


def score_episodic(
    model: nn.Module,
    dataset: ASVspoofLADataset,
    speaker_to_all: Dict[str, List[int]],
    speaker_to_bonafide: Dict[str, List[int]],
    episode_speakers: int,
    support_shots: int,
    query_shots: int,
    episodes: int,
    device: str,
    normalize_proto: bool,
    rng: random.Random,
) -> Tuple[float, float]:
    # Per-speaker prototype-conditioned scoring on a held-out split.
    model.train(False)
    all_scores: List[float] = []
    all_labels: List[float] = []

    eligible_speakers = [
        speaker for speaker, supports in speaker_to_bonafide.items()
        if len(supports) >= support_shots
        and (len(speaker_to_all[speaker]) - support_shots) >= query_shots
    ]
    if len(eligible_speakers) < episode_speakers:
        raise ValueError("Not enough eligible speakers for episodic evaluation.")

    with torch.no_grad():
        for _ in range(episodes):
            speakers = sample_episode_speakers(eligible_speakers, episode_speakers, rng)
            for speaker_id in speakers:
                support_pool = speaker_to_bonafide[speaker_id]
                all_pool = speaker_to_all[speaker_id]
                support_idx = rng.sample(support_pool, support_shots)
                remaining = [idx for idx in all_pool if idx not in support_idx]
                query_idx = rng.sample(remaining, query_shots)

                support_wavs = []
                for idx in support_idx:
                    wav, _, _ = dataset[idx]
                    support_wavs.append(wav)
                support_batch = torch.stack(support_wavs).to(device)
                proto = model.extract_embedding(support_batch).mean(dim=0)
                if normalize_proto:
                    proto = proto / (proto.norm(p=2) + 1e-8)

                query_wavs = []
                query_labels = []
                for idx in query_idx:
                    wav, label, _ = dataset[idx]
                    query_wavs.append(wav)
                    query_labels.append(label)

                query_batch = torch.stack(query_wavs).to(device)
                outputs = model(query_batch, speaker_proto=proto)
                scores = outputs["score"].cpu().tolist()
                all_scores.extend(scores)
                all_labels.extend([float(l.item()) for l in query_labels])

    eer, threshold = compute_eer(all_labels, all_scores)
    return eer, threshold


def main():
    parser = argparse.ArgumentParser(description="Train TwoStreamDetector (WavLM + AASIST)")
    parser.add_argument("--data-root", type=str, required=True, help="ASVspoof2019 LA root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="LR for the fusion head; ~1e-4 is a good warm-start value")
    parser.add_argument("--out", type=str, default="assets/checkpoints/two_stream.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wavlm-checkpoint", type=str,
                        default="assets/checkpoints/wavlm_baseline.pt",
                        help="Trained WavLM baseline checkpoint")
    parser.add_argument("--aasist-checkpoint", type=str, default=None,
                        help="AASIST weights path (defaults to aasist/models/weights/AASIST.pth)")
    parser.add_argument("--freeze-wavlm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-aasist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 to disable")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-fusion", type=str, default=None,
                        help="Warm-start from an existing fusion checkpoint")
    parser.add_argument("--warmstart-lr", type=float, default=1e-4)
    parser.add_argument("--freeze-fusion-epochs", type=int, default=0,
                        help="FiLM-only warm-up: freeze fusion head for the first N epochs")
    parser.add_argument("--episodic", action="store_true",
                        help="Use episodic fine-tuning with speaker prototypes")
    parser.add_argument("--episode-speakers", type=int, default=8)
    parser.add_argument("--support-shots", type=int, default=3)
    parser.add_argument("--query-shots", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-prototypes", action=argparse.BooleanOptionalAction, default=True)
    data_root = Path(args.data_root)
    proto_train = data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    proto_dev = data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    for p in (data_root, proto_train, proto_dev):
        if not p.exists():
            raise FileNotFoundError(p)

    device = get_device()
    print(f"Using device: {device}")

    print("Creating TwoStreamDetector...")
    wavlm_ckpt = args.wavlm_checkpoint if Path(args.wavlm_checkpoint).exists() else None
    if wavlm_ckpt:
        print(f"  Loading WavLM weights from: {wavlm_ckpt}")
    else:
        print("  Warning: WavLM checkpoint not found, using random pooling weights")

    model = TwoStreamDetector(
        wavlm_checkpoint=wavlm_ckpt,
        aasist_checkpoint=args.aasist_checkpoint,
        freeze_wavlm=args.freeze_wavlm,
        freeze_aasist=args.freeze_aasist,
    ).to(device)

    if args.init_fusion:
        init_path = Path(args.init_fusion)
        if not init_path.exists():
            raise FileNotFoundError(f"Init fusion checkpoint not found: {init_path}")
        checkpoint = torch.load(init_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded fusion weights from: {init_path}")
        # Drop to the suggested warm-start LR if the user didn't pass one.
        if args.lr == 2e-4:
            args.lr = args.warmstart_lr
            print(f"Using warm-start LR: {args.lr}")

    total = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}  trainable: {trainable_count:,}  frozen: {total - trainable_count:,}")

    criterion = nn.BCEWithLogitsLoss()

    if args.episodic:
        print("Loading datasets for episodic fine-tuning...")
        train_dataset = ASVspoofLADataset(
            data_root=data_root, protocol_path=proto_train, split="train",
            max_length_samples=64600, return_speaker_id=True,
        )
        dev_dataset = ASVspoofLADataset(
            data_root=data_root, protocol_path=proto_dev, split="dev",
            max_length_samples=64600, return_speaker_id=True,
        )

        speaker_to_all, speaker_to_bonafide = build_speaker_index(train_dataset)
        dev_speaker_to_all, dev_speaker_to_bonafide = build_speaker_index(dev_dataset)

        eligible_speakers = [
            speaker for speaker, supports in speaker_to_bonafide.items()
            if len(supports) >= args.support_shots
            and (len(speaker_to_all[speaker]) - args.support_shots) >= args.query_shots
        ]
        if len(eligible_speakers) < args.episode_speakers:
            raise ValueError("Not enough eligible speakers for episodic training.")

        set_personalization_trainable(model, include_fusion=True)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )

        rng = random.Random(args.seed)
        print(f"\nStarting episodic training for {args.epochs} epochs "
              f"({args.episodes_per_epoch} episodes/epoch)")
        if args.freeze_fusion_epochs > 0:
            print(f"FiLM-only warm-up: {args.freeze_fusion_epochs} epochs")

        best_eer = 1.0
        best_threshold = 0.5

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            if args.freeze_fusion_epochs > 0:
                freeze_fusion = epoch <= args.freeze_fusion_epochs
                set_personalization_trainable(model, include_fusion=not freeze_fusion)
                if freeze_fusion:
                    print("Training FiLM only (fusion head frozen)")

            losses: List[float] = []
            for _ in range(args.episodes_per_epoch):
                speakers = sample_episode_speakers(eligible_speakers, args.episode_speakers, rng)
                losses.append(train_episode(
                    model=model, dataset=train_dataset,
                    speaker_to_all=speaker_to_all, speaker_to_bonafide=speaker_to_bonafide,
                    speakers=speakers,
                    support_shots=args.support_shots, query_shots=args.query_shots,
                    device=device, criterion=criterion, optimizer=optimizer,
                    grad_clip=args.grad_clip, normalize_proto=args.normalize_prototypes, rng=rng,
                ))

            print(f"Train loss: {sum(losses) / len(losses):.4f}")

            dev_eer, dev_threshold = score_episodic(
                model=model, dataset=dev_dataset,
                speaker_to_all=dev_speaker_to_all, speaker_to_bonafide=dev_speaker_to_bonafide,
                episode_speakers=args.episode_speakers,
                support_shots=args.support_shots, query_shots=args.query_shots,
                episodes=args.eval_episodes, device=device,
                normalize_proto=args.normalize_prototypes, rng=rng,
            )
            print(f"Dev EER: {dev_eer*100:.2f}% (threshold: {dev_threshold:.4f})")

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            print(f"Learning rate: {current_lr:.6f}")

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
                print(f"Saved best checkpoint to {out_path}")

        print(f"\nTraining complete. Best dev EER: {best_eer*100:.2f}%  threshold: {best_threshold:.4f}")
        return

    # Non-episodic path: train the fusion head on shuffled batches.
    print("Loading datasets...")
    train_dataset = ASVspoofLADataset(
        data_root=data_root, protocol_path=proto_train, split="train",
        max_length_samples=64600,
    )
    dev_dataset = ASVspoofLADataset(
        data_root=data_root, protocol_path=proto_dev, split="dev",
        max_length_samples=64600,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    print(f"\nStarting training for {args.epochs} epochs (fusion head only)")

    best_eer = 1.0
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip,
        )
        print(f"Train loss: {train_loss:.4f}")

        dev_eer, dev_threshold = score_split(model, dev_loader, device)
        print(f"Dev EER: {dev_eer*100:.2f}% (threshold: {dev_threshold:.4f})")

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"Learning rate: {current_lr:.6f}")

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
            print(f"Saved best checkpoint to {out_path}")

    print(f"\nTraining complete. Best dev EER: {best_eer*100:.2f}%  threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    main()

