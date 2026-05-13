from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch

from vdetect.data.audio import load_audio, crop_or_pad


@dataclass
class SpeakerPrototype:
    speaker_id: str
    embedding: torch.Tensor
    num_samples: int
    sample_paths: List[str]
    created_at: str
    normalized: bool


def _default_db(embedding_dim: int) -> Dict[str, object]:
    return {"schema_version": 1, "embedding_dim": embedding_dim, "speakers": {}}


def load_db(db_path: Union[str, Path], embedding_dim: Optional[int] = None) -> Dict[str, object]:
    db_path = Path(db_path)
    if not db_path.exists():
        if embedding_dim is None:
            raise ValueError("embedding_dim is required to initialise a new DB.")
        return _default_db(embedding_dim)

    with open(db_path, "r") as f:
        db = json.load(f)

    db.setdefault("speakers", {})

    if embedding_dim is not None:
        existing_dim = db.get("embedding_dim")
        if existing_dim is None:
            db["embedding_dim"] = embedding_dim
        elif existing_dim != embedding_dim:
            raise ValueError(f"Embedding dim mismatch: DB has {existing_dim}, expected {embedding_dim}.")

    return db


def save_db(db_path: Union[str, Path], db: Dict[str, object]) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "w") as f:
        json.dump(db, f, indent=2)


def list_speakers(db: Dict[str, object]) -> List[str]:
    return sorted(db.get("speakers", {}).keys())


def delete_speaker(db: Dict[str, object], speaker_id: str) -> bool:
    speakers = db.get("speakers", {})
    if speaker_id in speakers:
        del speakers[speaker_id]
        return True
    return False


def compute_prototype(embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    # Speaker prototype is the (optionally L2-normalised) mean of per-sample embeddings.
    proto = embeddings.mean(dim=0)
    if normalize:
        proto = proto / (proto.norm(p=2) + 1e-8)
    return proto


def extract_embeddings(
    model,
    audio_paths: Iterable[Union[str, Path]],
    device: str,
    max_len: int = 64600,
) -> torch.Tensor:
    wavs: List[torch.Tensor] = []
    for audio_path in audio_paths:
        wav = load_audio(audio_path)
        wav = crop_or_pad(wav, max_len=max_len, train=False)
        wavs.append(wav)

    if not wavs:
        raise ValueError("No audio files provided for enrolment.")

    batch = torch.stack(wavs).to(device)
    with torch.no_grad():
        return model.extract_embedding(batch)


def upsert_speaker(
    db: Dict[str, object],
    speaker_id: str,
    prototype: torch.Tensor,
    sample_paths: List[Union[str, Path]],
    normalized: bool,
) -> None:
    speakers = db.setdefault("speakers", {})
    speakers[speaker_id] = {
        "embedding": prototype.detach().cpu().tolist(),
        "num_samples": len(sample_paths),
        "sample_paths": [str(p) for p in sample_paths],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalized": normalized,
    }


def load_speaker_prototype(
    db_path: Union[str, Path],
    speaker_id: str,
    device: Optional[str] = None,
) -> SpeakerPrototype:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Prototype DB not found: {db_path}")

    with open(db_path, "r") as f:
        db = json.load(f)

    speakers = db.get("speakers", {})
    if speaker_id not in speakers:
        raise KeyError(f"Speaker '{speaker_id}' not found in {db_path}")

    entry = speakers[speaker_id]
    embedding = torch.tensor(entry["embedding"], dtype=torch.float32)
    if device:
        embedding = embedding.to(device)

    return SpeakerPrototype(
        speaker_id=speaker_id,
        embedding=embedding,
        num_samples=entry.get("num_samples", 0),
        sample_paths=entry.get("sample_paths", []),
        created_at=entry.get("created_at", ""),
        normalized=entry.get("normalized", False),
    )
