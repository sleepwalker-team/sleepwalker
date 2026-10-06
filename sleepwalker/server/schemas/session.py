from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from ..validation.study import StudyFormat


class SessionState(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class SignalResponse(BaseModel):
    id: str
    label: str
    unit: str
    sample_rate: float


class SessionResponse(BaseModel):
    id: str
    state: SessionState
    created_at: datetime
    expires_at: datetime
    format: StudyFormat | None = None
    duration: float | None = None
    recording_start: datetime | None = None
    signals: list[SignalResponse] | None = None
