from datetime import datetime, timedelta, timezone

import pytest

from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.stores.job_store import JobError, JobPhase, JobState, JobStore
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


def test_expired_session_remains_accessible_while_job_is_active() -> None:
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    jobs = JobService(JobStore())
    service = SessionService(SessionStore(), clock=lambda: now, active_job_check=jobs.has_active_job)
    session = service.create([])
    jobs.create(session.id, "inference")
    now += timedelta(hours=3)

    assert service.get(session.id) is session
    assert session.expires_at == now + service.ttl


@pytest.mark.parametrize("state", [JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED])
def test_job_completion_starts_a_fresh_session_deadline(state) -> None:
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    service = SessionService(SessionStore(), clock=lambda: now)
    jobs = JobService(JobStore(), on_completion=service.renew_after_job)
    service.active_job_check = jobs.has_active_job
    session = service.create([])
    created_at = session.created_at
    job = jobs.create(session.id, "inference")
    now += timedelta(hours=3)
    if state == JobState.CANCELLED:
        jobs.request_cancel(job.id)
        jobs.mark_cancelled(job.id)
    else:
        jobs.mark_running(job.id, JobPhase.INFERENCE, "Running")
        if state == JobState.SUCCEEDED:
            jobs.mark_succeeded(job.id)
        else:
            jobs.mark_failed(job.id, JobError("INFERENCE_FAILED", "failed"))

    assert session.expires_at == now + service.ttl
    assert session.last_accessed_at == created_at
    assert service.get(session.id) is session


def test_completed_job_does_not_resurrect_explicitly_deleted_session() -> None:
    service = SessionService(SessionStore())
    jobs = JobService(JobStore(), on_completion=service.renew_after_job)
    session = service.create([])
    job = jobs.create(session.id, "inference")
    service.delete(session.id)
    jobs.request_cancel(job.id)
    jobs.mark_cancelled(job.id)
    assert service.get(session.id) is None


def test_get_rechecks_deadline_if_job_finishes_during_activity_check() -> None:
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    service = SessionService(SessionStore(), clock=lambda: now)
    jobs = JobService(JobStore(), on_completion=service.renew_after_job)
    session = service.create([])
    job = jobs.create(session.id, "inference")
    jobs.mark_running(job.id, JobPhase.INFERENCE, "Running")
    now += timedelta(hours=3)

    def complete_and_check(session_id):
        jobs.mark_succeeded(job.id)
        return jobs.has_active_job(session_id)

    service.active_job_check = complete_and_check
    assert service.get(session.id) is session
