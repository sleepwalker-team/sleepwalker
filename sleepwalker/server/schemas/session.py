from datetime import datetime

from pydantic import BaseModel

from ..stores.session_store import SessionState
from ..validation.study import StudyFormat
from .prediction import TaskResponse


class SignalResponse(BaseModel):
    id: str
    label: str
    unit: str
    sample_rate: float


class SessionResponse(BaseModel):
    id: str
    state: SessionState
    tasks: list[TaskResponse]
    created_at: datetime
    last_accessed_at: datetime
    expires_at: datetime
    format: StudyFormat | None = None
    duration: float | None = None
    recording_start: datetime | None = None
    signals: list[SignalResponse] | None = None
