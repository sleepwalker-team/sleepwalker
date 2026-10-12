from datetime import datetime, timezone

import pytest

from sleepwalker.server.stores.job_store import Job, JobState, JobStore


def make_job(job_id: str = "job-test") -> Job:
    now = datetime.now(timezone.utc)

    return Job(
        id=job_id,
        session_id="sess-test",
        type="inference",
        state=JobState.QUEUED,
        created_at=now,
        updated_at=now,
        message="Waiting for execution",
    )


def test_job_store_adds_and_gets_job() -> None:
    job = make_job()
    store = JobStore()

    store.add(job)

    assert store.get(job.id) is job
    assert store.get("unknown-job") is None


def test_job_store_rejects_duplicate_job_id() -> None:
    store = JobStore()
    store.add(make_job("duplicate"))

    with pytest.raises(ValueError, match="already registered"):
        store.add(make_job("duplicate"))
        
def test_job_store_lists_jobs_for_session() -> None:
    first_job = make_job("job-first")
    second_job = make_job("job-second")
    other_job = make_job("job-other")
    other_job.session_id = "sess-other"

    store = JobStore()
    store.add(first_job)
    store.add(other_job)
    store.add(second_job)

    assert store.list_for_session("sess-test") == [
        first_job,
        second_job,
    ]
    assert store.list_for_session("unknown-session") == []
    
def test_job_store_deletes_jobs_for_session() -> None:
    first_job = make_job("job-first")
    second_job = make_job("job-second")
    other_job = make_job("job-other")
    other_job.session_id = "sess-other"

    store = JobStore()
    store.add(first_job)
    store.add(other_job)
    store.add(second_job)

    assert store.delete_session("sess-test") == 2
    assert store.get(first_job.id) is None
    assert store.get(second_job.id) is None
    assert store.get(other_job.id) is other_job