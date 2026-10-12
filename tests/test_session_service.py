from datetime import datetime, timedelta, timezone

from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.stores.session_store import SessionStore


def test_get_renews_session_expiration() -> None:
    created_at = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    accessed_at = created_at + timedelta(minutes=30)
    times = iter([created_at, accessed_at])

    service = SessionService(
        SessionStore(),
        ttl=timedelta(hours=2),
        clock=lambda: next(times),
    )

    session = service.create(source_files=[])
    accessed_session = service.get(session.id)

    assert accessed_session is session
    assert session.last_accessed_at == accessed_at
    assert session.expires_at == accessed_at + timedelta(hours=2)
    
def test_get_does_not_renew_expired_session() -> None:
    created_at = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    checked_at = created_at + timedelta(hours=2, seconds=1)
    times = iter([created_at, checked_at])

    service = SessionService(
        SessionStore(),
        ttl=timedelta(hours=2),
        clock=lambda: next(times),
    )

    session = service.create(source_files=[])
    accessed_session = service.get(session.id)

    assert accessed_session is None
    assert session.last_accessed_at == created_at
    assert session.expires_at == created_at + timedelta(hours=2)
    
def test_list_expired_returns_only_expired_sessions() -> None:
    first_created_at = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    second_created_at = datetime(2026, 10, 12, 14, 0, tzinfo=timezone.utc)
    cleanup_at = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
    times = iter([first_created_at, second_created_at, cleanup_at])

    service = SessionService(
        SessionStore(),
        ttl=timedelta(hours=2),
        clock=lambda: next(times),
    )

    expired_session = service.create(source_files=[])
    active_session = service.create(source_files=[])

    assert service.list_expired() == [expired_session]
    assert active_session.expires_at == datetime(
        2026,
        10,
        12,
        16,
        0,
        tzinfo=timezone.utc,
    )