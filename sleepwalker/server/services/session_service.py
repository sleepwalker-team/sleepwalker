from datetime import datetime, timedelta, timezone
from uuid import uuid4

from ..stores.session_store import Session, SessionStore


class SessionService:
    def __init__(self, store: SessionStore):
        self.store = store

    def create(self, source_files):
        now = datetime.now(timezone.utc)

        session = Session(
            id=f"sess_{uuid4()}",
            created_at=now,
            expires_at=now + timedelta(hours=2),
            last_accessed_at=now,
            source_files=source_files,
        )

        self.store.add(session)

        return session

    def get(self, session_id: str):
        return self.store.get(session_id)

    def delete(self, session_id: str):
        return self.store.delete(session_id)