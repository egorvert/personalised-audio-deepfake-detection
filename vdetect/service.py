from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch

from vdetect.data.audio import load_audio, load_audio_from_buffer, crop_or_pad
from vdetect.enrollment import (
    compute_prototype,
    extract_embeddings,
    load_db,
    load_speaker_prototype,
    save_db,
    upsert_speaker,
)


class ModelType(str, Enum):
    wavlm = "wavlm"
    aasist = "aasist"
    fusion = "fusion"


@dataclass
class DetectionResult:
    file: str
    model: str
    score: float
    label: str
    confidence: float
    threshold: float
    speaker_id: Optional[str] = None


@dataclass
class BatchDetectionResult:
    results: List[DetectionResult]
    total: int
    spoof_count: int
    bonafide_count: int
    errors: List[str]


@dataclass
class EnrollmentResult:
    speaker_id: str
    action: str  # "Enrolled" or "Updated"
    num_samples: int
    db_path: str


@dataclass
class CheckpointInfo:
    filename: str
    eer: Optional[float] = None
    threshold: Optional[float] = None
    epoch: Optional[int] = None
    model_name: Optional[str] = None
    lr: Optional[float] = None
    batch_size: Optional[int] = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DetectionService:
    # Shared inference + enrolment logic used by the CLI and the FastAPI server.

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None
        self._model_type: Optional[ModelType] = None
        self._device: str = ""
        self._max_len: int = 64000
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type.value if self._model_type else None

    @property
    def device(self) -> str:
        return self._device

    def load_model(
        self,
        model_type: ModelType,
        weights: Path,
        device: Optional[str] = None,
    ) -> None:
        # Call once at startup; subsequent requests reuse the loaded model.
        from vdetect.models.wavlm_baseline import WavLMDetector
        from vdetect.models.two_stream import TwoStreamDetector

        device = device or get_device()
        self._device = device
        self._model_type = model_type

        if model_type == ModelType.wavlm:
            model = WavLMDetector().to(device)
            checkpoint = torch.load(weights, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            self._max_len = 64000
        elif model_type == ModelType.aasist:
            from vdetect.models.aasist_encoder import AASISTEncoder
            model = AASISTEncoder(pretrained_path=None, freeze=True).to(device)
            self._max_len = 64600
        else:  # fusion
            model = TwoStreamDetector(
                wavlm_checkpoint=None,
                aasist_checkpoint=None,
                freeze_wavlm=True,
                freeze_aasist=True,
            ).to(device)
            checkpoint = torch.load(weights, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            self._max_len = 64600

        model.train(False)
        self._model = model

    def _preprocess_wav(self, wav: torch.Tensor) -> torch.Tensor:
        wav = crop_or_pad(wav, max_len=self._max_len, train=False)
        return wav.unsqueeze(0).to(self._device)

    def _resolve_speaker_proto(
        self,
        speaker_id: Optional[str],
        db_path: Optional[Path],
    ) -> Optional[torch.Tensor]:
        if not speaker_id:
            return None
        if not db_path:
            raise ValueError("db_path is required when speaker_id is provided")
        proto = load_speaker_prototype(db_path, speaker_id, device=self._device)
        return proto.embedding

    def _run_inference(
        self,
        wav: torch.Tensor,
        speaker_proto: Optional[torch.Tensor] = None,
    ) -> float:
        with self._lock:
            with torch.no_grad():
                if self._model_type == ModelType.aasist:
                    return self._model.get_score(wav).item()
                else:
                    outputs = self._model(wav, speaker_proto=speaker_proto)
                    return outputs["score"].item()

    @staticmethod
    def _score_to_result(
        filename: str,
        model_name: str,
        score: float,
        threshold: float,
        speaker_id: Optional[str],
    ) -> DetectionResult:
        label = "spoof" if score >= threshold else "bonafide"
        confidence = score if label == "spoof" else (1 - score)
        return DetectionResult(
            file=filename,
            model=model_name,
            score=round(score, 4),
            label=label,
            confidence=round(confidence, 4),
            threshold=threshold,
            speaker_id=speaker_id,
        )

    def detect_file(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        speaker_id: Optional[str] = None,
        db_path: Optional[Path] = None,
    ) -> DetectionResult:
        wav = load_audio(audio_path)
        wav = self._preprocess_wav(wav)
        speaker_proto = self._resolve_speaker_proto(speaker_id, db_path)
        score = self._run_inference(wav, speaker_proto)
        return self._score_to_result(
            str(audio_path), self._model_type.value, score, threshold, speaker_id,
        )

    def detect_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "upload",
        threshold: float = 0.5,
        speaker_id: Optional[str] = None,
        db_path: Optional[Path] = None,
    ) -> DetectionResult:
        # Used by the FastAPI upload path; the suffix is just a format hint for torchaudio.
        ext = Path(filename).suffix.lstrip(".") or None
        wav = load_audio_from_buffer(io.BytesIO(audio_bytes), fmt=ext)
        wav = self._preprocess_wav(wav)
        speaker_proto = self._resolve_speaker_proto(speaker_id, db_path)
        score = self._run_inference(wav, speaker_proto)
        return self._score_to_result(
            filename, self._model_type.value, score, threshold, speaker_id,
        )

    def batch_detect(
        self,
        input_dir: Path,
        threshold: float = 0.5,
        extensions: Optional[List[str]] = None,
        speaker_id: Optional[str] = None,
        db_path: Optional[Path] = None,
        on_progress: Optional[object] = None,
    ) -> BatchDetectionResult:
        if extensions is None:
            extensions = [".wav", ".flac", ".mp3"]

        audio_files: List[Path] = []
        for ext in extensions:
            if not ext.startswith("."):
                ext = "." + ext
            audio_files.extend(input_dir.glob(f"**/*{ext}"))

        speaker_proto = self._resolve_speaker_proto(speaker_id, db_path)

        results: List[DetectionResult] = []
        errors: List[str] = []
        for i, audio_path in enumerate(audio_files, 1):
            if on_progress:
                on_progress(i, len(audio_files), audio_path.name)
            try:
                wav = load_audio(audio_path)
                wav = self._preprocess_wav(wav)
                score = self._run_inference(wav, speaker_proto)
                results.append(self._score_to_result(
                    str(audio_path.relative_to(input_dir)),
                    self._model_type.value,
                    score, threshold, speaker_id,
                ))
            except Exception as e:
                errors.append(f"{audio_path.name}: {e}")

        spoof_count = sum(1 for r in results if r.label == "spoof")
        return BatchDetectionResult(
            results=results,
            total=len(results),
            spoof_count=spoof_count,
            bonafide_count=len(results) - spoof_count,
            errors=errors,
        )

    def enroll_speaker(
        self,
        speaker_id: str,
        audio_paths: List[Path],
        weights: Path,
        db_path: Path,
        normalize: bool = True,
        device: Optional[str] = None,
    ) -> EnrollmentResult:
        # Loads the fusion model standalone so enrolment doesn't require the main service to be loaded.
        from vdetect.models.two_stream import TwoStreamDetector

        device = device or get_device()

        model = TwoStreamDetector(
            wavlm_checkpoint=None,
            aasist_checkpoint=None,
            freeze_wavlm=True,
            freeze_aasist=True,
        ).to(device)
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model.train(False)

        embeddings = extract_embeddings(model, audio_paths, device=device, max_len=64600)
        prototype = compute_prototype(embeddings, normalize=normalize)

        db_data = load_db(db_path, embedding_dim=prototype.numel())
        existing = speaker_id in db_data.get("speakers", {})
        upsert_speaker(
            db_data,
            speaker_id=speaker_id,
            prototype=prototype,
            sample_paths=audio_paths,
            normalized=normalize,
        )
        save_db(db_path, db_data)

        return EnrollmentResult(
            speaker_id=speaker_id,
            action="Updated" if existing else "Enrolled",
            num_samples=len(audio_paths),
            db_path=str(db_path),
        )

    @staticmethod
    def get_checkpoint_info(weights: Path) -> CheckpointInfo:
        checkpoint = torch.load(weights, map_location="cpu")
        args = checkpoint.get("args", {})
        return CheckpointInfo(
            filename=weights.name,
            eer=checkpoint.get("eer"),
            threshold=checkpoint.get("threshold"),
            epoch=checkpoint.get("epoch"),
            model_name=args.get("model_name"),
            lr=args.get("lr"),
            batch_size=args.get("batch_size"),
        )
