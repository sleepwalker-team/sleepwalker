from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..validation.study import SignalInfo, StudyFormat


@dataclass
class Session:
    id: str
    created_at: datetime
    expires_at: datetime
    last_accessed_at: datetime
    source_files: list[Path]

    format: StudyFormat | None = None
    duration: float | None = None
    recording_start: datetime | None = None
    signals: list[SignalInfo] | None = None


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def add(self, session: Session) -> None:
        self._sessions[session.id] = session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> Session | None:
        return self._sessions.pop(session_id, None)
