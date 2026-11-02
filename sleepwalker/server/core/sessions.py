"""Temporary study sessions with an inactivity deadline."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ..recording.study import SignalInfo, StudyFormat, StudyInfo

DEFAULT_SESSION_TTL = timedelta(hours=2)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    # Metadata of the exact models behind this session's predictions.
    model_provenance: list[dict] = field(default_factory=list)

    def apply_study(self, study: StudyInfo) -> None:
        self.source_files = study.source_files
        self.format = study.format
        self.duration = study.duration
        self.recording_start = study.recording_start
        self.signals = study.signals


class SessionService:
    def __init__(
        self,
        *,
        ttl: timedelta = DEFAULT_SESSION_TTL,
        clock: Callable[[], datetime] = utc_now,
        active_job_check: Callable[[str], bool] | None = None,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("Session TTL must be positive.")

        self.ttl = ttl
        self.clock = clock
        self.active_job_check = active_job_check or (lambda _session_id: False)
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def create(self, source_files: list[Path]) -> Session:
        now = self.clock()
        session = Session(
            id=f"sess_{uuid4()}",
            created_at=now,
            expires_at=now + self.ttl,
            last_accessed_at=now,
            source_files=source_files,
            state=SessionState.CREATED,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Return a live session and renew its deadline; this counts as client access."""
        session = self.peek(session_id)
        if session is None:
            return None

        now = self.clock()
        if session.expires_at <= now:
            active = self.active_job_check(session_id)
            # A completing job may renew the deadline while checking activity.
            if not active and session.expires_at <= now:
                return None

        session.last_accessed_at = now
        session.expires_at = max(session.expires_at, now + self.ttl)
        return session

    def peek(self, session_id: str) -> Session | None:
        """Return a session without checking or renewing its deadline."""
        with self._lock:
            return self._sessions.get(session_id)

    def renew_after_job(self, session_id: str) -> None:
        session = self.peek(session_id)
        if session is not None:
            session.expires_at = max(session.expires_at, self.clock() + self.ttl)

    def list_expired(self, at: datetime | None = None) -> list[Session]:
        now = at if at is not None else self.clock()
        return [session for session in self.list() if session.expires_at <= now]

    def delete(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def list(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())
