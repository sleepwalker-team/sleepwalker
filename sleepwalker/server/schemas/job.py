from datetime import datetime

from pydantic import BaseModel

from ..stores.job_store import JobPhase, JobState
from .common import ErrorDetail


class InferenceRequest(BaseModel):
    models: list[str]


class JobAcceptedResponse(BaseModel):
    job_id: str
    state: JobState


class JobResponse(BaseModel):
    id: str
    type: str
    state: JobState
    phase: JobPhase | None = None
    progress: float | None = None
    message: str | None = None
    updated_at: datetime
    error: ErrorDetail | None = None
