from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from .common import ErrorDetail


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobPhase(str, Enum):
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"


class JobResponse(BaseModel):
    id: str
    type: str
    state: JobState
    phase: JobPhase | None = None
    progress: float | None = None
    message: str | None = None
    updated_at: datetime
    error: ErrorDetail | None = None