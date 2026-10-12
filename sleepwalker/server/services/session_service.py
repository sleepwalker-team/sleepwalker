from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from ..stores.session_store import Session, SessionState, SessionStore


DEFAULT_SESSION_TTL = timedelta(hours=2)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionService:
    def __init__(
        self,
        store: SessionStore,
        *,
        ttl: timedelta = DEFAULT_SESSION_TTL,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("Session TTL must be positive.")

        self.store = store
        self.ttl = ttl
        self.clock = clock

    def create(self, source_files):
        now = self.clock()

        session = Session(
            id=f"sess_{uuid4()}",
            created_at=now,
            expires_at=now + self.ttl,
            last_accessed_at=now,
            source_files=source_files,
            state=SessionState.CREATED,
        )

        self.store.add(session)
        return session

    def get(self, session_id: str):
        session = self.store.get(session_id)

        if session is None:
            return None

        now = self.clock()

        if session.expires_at <= now:
            return None

        session.last_accessed_at = now
        session.expires_at = now + self.ttl

        return session

    def list_expired(self) -> list[Session]:
        now = self.clock()

        return [session for session in self.store.list() if session.expires_at <= now]

    def delete(self, session_id: str):
        return self.store.delete(session_id)
