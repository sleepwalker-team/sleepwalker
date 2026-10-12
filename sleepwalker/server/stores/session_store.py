from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..validation.study import SignalInfo, StudyFormat


class SessionState(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class Session:
    id: str
    created_at: datetime
    expires_at: datetime
    last_accessed_at: datetime
    source_files: list[Path]
    state: SessionState

    format: StudyFormat | None = None
    duration: float | None = None
    recording_start: datetime | None = None
    signals: list[SignalInfo] | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def add(self, session: Session) -> None:
        self._sessions[session.id] = session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> Session | None:
        return self._sessions.pop(session_id, None)

    def list(self) -> list[Session]:
        return list(self._sessions.values())
