from pathlib import Path
from typing import Optional, Tuple, Union
import io
import random
import subprocess

import torch
import torchaudio


SAMPLE_RATE = 16000
MAX_LENGTH_SECONDS = 4.0
MAX_LENGTH_SAMPLES = int(MAX_LENGTH_SECONDS * SAMPLE_RATE)


# Fallback decoder: torchcodec rejects browser-produced webm with incomplete
# EBML headers, so we shell out to ffmpeg which is much more forgiving.
def _ffmpeg_decode_path(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-fflags", "+genpts+igndts",
        "-err_detect", "ignore_err",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg fallback failed (rc={proc.returncode}) for {path.name}: "
            f"{stderr[-512:] if stderr else '(no stderr)'}"
        )
    return torchaudio.load(io.BytesIO(proc.stdout), format="wav")


def _ffmpeg_decode_bytes(data: bytes, target_sr: int) -> Tuple[torch.Tensor, int]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-fflags", "+genpts+igndts",
        "-err_detect", "ignore_err",
        "-i", "pipe:0",
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=data, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg fallback failed (rc={proc.returncode}) for in-memory buffer: "
            f"{stderr[-512:] if stderr else '(no stderr)'}"
        )
    return torchaudio.load(io.BytesIO(proc.stdout), format="wav")


def _to_mono_1d(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    return wav


def load_audio(path: Union[str, Path], target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    # Load audio from disk and return a mono 1-D tensor at target_sr.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        wav, sr = torchaudio.load(str(path))
    except RuntimeError:
        wav, sr = _ffmpeg_decode_path(path, target_sr)

    wav = _to_mono_1d(wav)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def load_audio_from_buffer(
    buffer: io.BytesIO,
    target_sr: int = SAMPLE_RATE,
    fmt: Optional[str] = None,
) -> torch.Tensor:
    # Same as load_audio() but reads from a BytesIO (used by the API upload path).
    try:
        wav, sr = torchaudio.load(buffer, format=fmt)
    except RuntimeError:
        buffer.seek(0)
        wav, sr = _ffmpeg_decode_bytes(buffer.read(), target_sr)

    wav = _to_mono_1d(wav)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def crop_or_pad(
    wav: torch.Tensor,
    max_len: int = MAX_LENGTH_SAMPLES,
    train: bool = True,
) -> torch.Tensor:
    # Random crop during training, centre crop otherwise; zero-pad when too short.
    current_len = wav.numel()

    if current_len > max_len:
        if train:
            start = random.randint(0, current_len - max_len)
        else:
            start = (current_len - max_len) // 2
        return wav[start:start + max_len]

    if current_len < max_len:
        return torch.nn.functional.pad(wav, (0, max_len - current_len))

    return wav


def normalize_audio(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (wav - wav.mean()) / (wav.std() + eps)

