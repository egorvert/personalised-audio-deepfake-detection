from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset

from .audio import load_audio, crop_or_pad


# Split prefixes used in the ASVspoof2019 utterance IDs.
_SPLIT_PREFIX = {"LA_T_": "train", "LA_D_": "dev", "LA_E_": "eval"}


def _split_for(utt_id: str) -> str:
    for prefix, name in _SPLIT_PREFIX.items():
        if utt_id.startswith(prefix):
            return name
    return "unknown"


def read_protocol(proto_path: Union[str, Path]) -> List[Tuple[str, str, int]]:
    # Protocol line layout: <speaker_id> <utt_id> <system_id> <key> <label>
    proto_path = Path(proto_path)
    if not proto_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {proto_path}")

    items: List[Tuple[str, str, int]] = []
    with open(proto_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            label = 0 if parts[4] == "bonafide" else 1
            items.append((utt_id, _split_for(utt_id), label))
    return items


def read_protocol_with_speaker(proto_path: Union[str, Path]) -> List[Tuple[str, str, int, str]]:
    proto_path = Path(proto_path)
    if not proto_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {proto_path}")

    items: List[Tuple[str, str, int, str]] = []
    with open(proto_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id = parts[0]
            utt_id = parts[1]
            label = 0 if parts[4] == "bonafide" else 1
            items.append((utt_id, _split_for(utt_id), label, speaker_id))
    return items


def resolve_path(data_root: Union[str, Path], utt_id: str) -> Path:
    # Map an utterance ID to its on-disk .flac path under the ASVspoof LA layout.
    data_root = Path(data_root)
    if utt_id.startswith("LA_T_"):
        return data_root / "ASVspoof2019_LA_train" / "flac" / f"{utt_id}.flac"
    if utt_id.startswith("LA_D_"):
        return data_root / "ASVspoof2019_LA_dev" / "flac" / f"{utt_id}.flac"
    if utt_id.startswith("LA_E_"):
        return data_root / "ASVspoof2019_LA_eval" / "flac" / f"{utt_id}.flac"
    raise ValueError(f"Unknown utterance ID format: {utt_id}")


class ASVspoofLADataset(Dataset):
    # PyTorch dataset wrapping an ASVspoof 2019 LA protocol file.

    def __init__(
        self,
        data_root: Union[str, Path],
        protocol_path: Union[str, Path],
        split: str = "train",
        max_length_samples: int = 64000,
        normalize: bool = False,
        return_speaker_id: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_length = max_length_samples
        self.normalize = normalize
        self.train_mode = (split == "train")
        self.return_speaker_id = return_speaker_id

        all_items = (
            read_protocol_with_speaker(protocol_path)
            if return_speaker_id
            else read_protocol(protocol_path)
        )
        self.items = [item for item in all_items if item[1] == split]

        if not self.items:
            raise ValueError(
                f"No samples found for split '{split}' in {protocol_path}. "
                f"Available splits: {set(item[1] for item in all_items)}"
            )

        bonafide = sum(1 for item in self.items if item[2] == 0)
        spoof = sum(1 for item in self.items if item[2] == 1)
        print(f"Loaded {split} split: {len(self.items)} samples ({bonafide} bonafide, {spoof} spoof)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.return_speaker_id:
            utt_id, _, label, speaker_id = self.items[idx]
        else:
            utt_id, _, label = self.items[idx]
            speaker_id = None

        wav = load_audio(resolve_path(self.data_root, utt_id))
        wav = crop_or_pad(wav, self.max_length, train=self.train_mode)

        if self.normalize:
            wav = (wav - wav.mean()) / (wav.std() + 1e-8)

        # Use float32 so the label tensor works on MPS.
        label_tensor = torch.tensor(label, dtype=torch.float32)
        if self.return_speaker_id:
            return wav, label_tensor, speaker_id
        return wav, label_tensor

    def get_label_distribution(self) -> dict:
        bonafide = sum(1 for item in self.items if item[2] == 0)
        spoof = sum(1 for item in self.items if item[2] == 1)
        return {"bonafide": bonafide, "spoof": spoof, "total": len(self.items)}
