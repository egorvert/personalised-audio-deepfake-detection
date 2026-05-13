from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    file: str
    model: str
    score: float = Field(ge=0.0, le=1.0)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float
    speaker_id: Optional[str] = None


class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    total: int
    spoof_count: int
    bonafide_count: int
    errors: List[str] = []


class EnrollmentResponse(BaseModel):
    speaker_id: str
    action: str
    num_samples: int
    db_path: str


class CheckpointInfoResponse(BaseModel):
    filename: str
    eer: Optional[float] = None
    threshold: Optional[float] = None
    epoch: Optional[int] = None
    model_name: Optional[str] = None
    lr: Optional[float] = None
    batch_size: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_type: Optional[str] = None
    device: str
