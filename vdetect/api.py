import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Importing this side-effects the PII-scrubbing log formatter onto uvicorn loggers.
from vdetect import logging as _vdetect_logging  # noqa: F401
from vdetect import __version__
from vdetect.schemas import (
    BatchDetectionResponse,
    CheckpointInfoResponse,
    DetectionResponse,
    EnrollmentResponse,
    HealthResponse,
)
from vdetect.mps_lock import mps_lock
from vdetect.service import DetectionService, ModelType


service = DetectionService()

DEFAULT_DB = Path(os.environ.get("VDETECT_DB_PATH", "assets/enrollments/prototypes.json"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_type = ModelType(os.environ.get("VDETECT_MODEL_TYPE", "wavlm"))
    weights = Path(os.environ.get("VDETECT_WEIGHTS", "assets/checkpoints/wavlm_baseline.pt"))
    # Hold the cross-process MPS lock only while loading; never per-request.
    with mps_lock("vdetect-api"):
        service.load_model(model_type, weights)
    yield


app = FastAPI(
    title="VDetect API",
    description="Audio deepfake detection API",
    version=__version__,
    lifespan=lifespan,
)

# CORS is a dev convenience only. In production Caddy mediates traffic and
# uvicorn binds to 127.0.0.1, so the browser never speaks to :8000 directly.
_cors_origins = [
    o.strip()
    for o in os.environ.get("VDETECT_CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
_cors_headers = [
    h.strip()
    for h in os.environ.get("VDETECT_CORS_HEADERS", "Content-Type,Accept").split(",")
    if h.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=_cors_headers,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=service.is_loaded,
        model_type=service.model_type,
        device=service.device,
    )


@app.post("/detect", response_model=DetectionResponse)
def detect(
    audio: UploadFile = File(..., description="Audio file to analyse"),
    threshold: float = Form(0.5),
    speaker_id: Optional[str] = Form(None),
):
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = service.detect_bytes(
            audio_bytes=audio.file.read(),
            filename=audio.filename or "unknown",
            threshold=threshold,
            speaker_id=speaker_id,
            db_path=DEFAULT_DB if speaker_id else None,
        )
        return DetectionResponse(**asdict(result))
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/batch-detect", response_model=BatchDetectionResponse)
def batch_detect(
    files: List[UploadFile] = File(..., description="Audio files to analyse"),
    threshold: float = Form(0.5),
    speaker_id: Optional[str] = Form(None),
):
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results: List[DetectionResponse] = []
    errors: List[str] = []

    for f in files:
        try:
            result = service.detect_bytes(
                audio_bytes=f.file.read(),
                filename=f.filename or "unknown",
                threshold=threshold,
                speaker_id=speaker_id,
                db_path=DEFAULT_DB if speaker_id else None,
            )
            results.append(DetectionResponse(**asdict(result)))
        except Exception as e:
            errors.append(f"{f.filename}: {e}")

    spoof_count = sum(1 for r in results if r.label == "spoof")
    return BatchDetectionResponse(
        results=results,
        total=len(results),
        spoof_count=spoof_count,
        bonafide_count=len(results) - spoof_count,
        errors=errors,
    )


@app.post("/enroll", response_model=EnrollmentResponse)
def enroll(
    speaker_id: str = Form(..., description="Speaker ID to enrol"),
    files: List[UploadFile] = File(
        ..., description="3-5 bonafide audio samples for enrolment",
    ),
    normalize: bool = Form(True),
):
    if not 3 <= len(files) <= 5:
        raise HTTPException(status_code=422, detail="Provide 3-5 enrolment audio samples.")

    weights = Path(os.environ.get("VDETECT_ENROLL_WEIGHTS", "assets/checkpoints/two_stream.pt"))
    if not weights.exists():
        raise HTTPException(status_code=500, detail=f"Enrolment weights not found: {weights}")

    # extract_embeddings reads files by path, so spool uploads to a temp dir first.
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_paths: List[Path] = []
        for f in files:
            dst = Path(tmpdir) / (f.filename or "sample.wav")
            dst.write_bytes(f.file.read())
            audio_paths.append(dst)

        try:
            result = service.enroll_speaker(
                speaker_id=speaker_id,
                audio_paths=audio_paths,
                weights=weights,
                db_path=DEFAULT_DB,
                normalize=normalize,
            )
            return EnrollmentResponse(**asdict(result))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))


@app.get("/info", response_model=CheckpointInfoResponse)
def checkpoint_info(weights: str = Query(..., description="Path to model checkpoint")):
    weights_path = Path(weights)
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {weights}")
    result = DetectionService.get_checkpoint_info(weights_path)
    return CheckpointInfoResponse(**asdict(result))
