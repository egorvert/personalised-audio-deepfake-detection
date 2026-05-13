import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
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


def compute_eer(labels: List[float], scores: List[float]) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    min_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
    threshold = thresholds[min_idx] if min_idx < len(thresholds) else 0.5
    return float(eer), float(threshold)


def build_speaker_index(
    dataset: ASVspoofLADataset
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    speaker_to_all: Dict[str, List[int]] = {}
    speaker_to_bonafide: Dict[str, List[int]] = {}

    for idx, item in enumerate(dataset.items):
        label = item[2]
        speaker_id = item[3]
        speaker_to_all.setdefault(speaker_id, []).append(idx)
        if label == 0:
            speaker_to_bonafide.setdefault(speaker_id, []).append(idx)

    return speaker_to_all, speaker_to_bonafide


def evaluate_generic(
    model: TwoStreamDetector,
    dataloader: DataLoader,
    device: str
) -> Tuple[float, float]:
    model.train(False)
    all_scores: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for wav, labels in dataloader:
            wav = wav.to(device)
            outputs = model(wav)
            scores = outputs["score"].cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(labels.tolist())

    return compute_eer(all_labels, all_scores)


def evaluate_personalized(
    model: TwoStreamDetector,
    dataset: ASVspoofLADataset,
    speaker_to_all: Dict[str, List[int]],
    speaker_to_bonafide: Dict[str, List[int]],
    episode_speakers: int,
    support_shots: int,
    query_shots: int,
    episodes: int,
    device: str,
    normalize_proto: bool,
    rng: random.Random
) -> Tuple[float, float]:
    model.train(False)
    all_scores: List[float] = []
    all_labels: List[float] = []

    eligible_speakers = [
        speaker for speaker, supports in speaker_to_bonafide.items()
        if len(supports) >= support_shots
        and (len(speaker_to_all[speaker]) - support_shots) >= query_shots
    ]
    if len(eligible_speakers) < episode_speakers:
        raise ValueError("Not enough eligible speakers for personalized evaluation.")

    with torch.no_grad():
        for _ in range(episodes):
            speakers = rng.sample(eligible_speakers, episode_speakers)
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

    return compute_eer(all_labels, all_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generic vs personalized fusion detection")
    parser.add_argument("--data-root", type=str, required=True, help="ASVspoof2019 LA root")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "eval"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fusion-weights", type=str, default="assets/checkpoints/two_stream.pt")
    parser.add_argument("--wavlm-weights", type=str, default="assets/checkpoints/wavlm_baseline.pt")
    parser.add_argument("--aasist-weights", type=str, default=None)
    parser.add_argument("--episode-speakers", type=int, default=8)
    parser.add_argument("--support-shots", type=int, default=3)
    parser.add_argument("--query-shots", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize-prototypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize speaker prototypes"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.split == "train":
        proto_file = "ASVspoof2019.LA.cm.train.trn.txt"
    elif args.split == "dev":
        proto_file = "ASVspoof2019.LA.cm.dev.trl.txt"
    else:
        proto_file = "ASVspoof2019.LA.cm.eval.trl.txt"

    proto_path = data_root / "ASVspoof2019_LA_cm_protocols" / proto_file
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not proto_path.exists():
        raise FileNotFoundError(f"Protocol not found: {proto_path}")

    device = get_device()
    print(f"Using device: {device}")

    dataset = ASVspoofLADataset(
        data_root=data_root,
        protocol_path=proto_path,
        split=args.split,
        max_length_samples=64600
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    model = TwoStreamDetector(
        wavlm_checkpoint=args.wavlm_weights if Path(args.wavlm_weights).exists() else None,
        aasist_checkpoint=args.aasist_weights,
        freeze_wavlm=True,
        freeze_aasist=True
    ).to(device)
    checkpoint = torch.load(args.fusion_weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    generic_eer, generic_threshold = evaluate_generic(model, dataloader, device)
    print(f"Generic EER: {generic_eer*100:.2f}% (threshold: {generic_threshold:.4f})")

    personalized_dataset = ASVspoofLADataset(
        data_root=data_root,
        protocol_path=proto_path,
        split=args.split,
        max_length_samples=64600,
        return_speaker_id=True
    )
    speaker_to_all, speaker_to_bonafide = build_speaker_index(personalized_dataset)

    rng = random.Random(args.seed)
    personalized_eer, personalized_threshold = evaluate_personalized(
        model=model,
        dataset=personalized_dataset,
        speaker_to_all=speaker_to_all,
        speaker_to_bonafide=speaker_to_bonafide,
        episode_speakers=args.episode_speakers,
        support_shots=args.support_shots,
        query_shots=args.query_shots,
        episodes=args.episodes,
        device=device,
        normalize_proto=args.normalize_prototypes,
        rng=rng
    )
    print(f"Personalized EER: {personalized_eer*100:.2f}% (threshold: {personalized_threshold:.4f})")


if __name__ == "__main__":
    main()
