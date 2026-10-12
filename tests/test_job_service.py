from datetime import timezone

import pytest

from sleepwalker.server.services.job_service import ActiveJobExistsError, JobService
from sleepwalker.server.stores.job_store import JobError, JobPhase, JobState, JobStore


def test_job_service_creates_queued_job() -> None:
    store = JobStore()
    service = JobService(store)

    job = service.create(
        session_id="sess-test",
        job_type="inference",
    )

    assert job.id.startswith("job_")
    assert job.session_id == "sess-test"
    assert job.type == "inference"
    assert job.state == JobState.QUEUED
    assert job.phase is None
    assert job.progress is None
    assert job.message == "Waiting for execution"
    assert job.error is None
    assert job.created_at.tzinfo is timezone.utc
    assert job.updated_at == job.created_at
    assert store.get(job.id) is job


def test_job_service_gets_existing_job() -> None:
    service = JobService(JobStore())
    job = service.create(
        session_id="sess-test",
        job_type="inference",
    )

    assert service.get(job.id) is job
    assert service.get("unknown-job") is None


def test_job_service_copies_parameters() -> None:
    service = JobService(JobStore())
    parameters = {"models": ["model-a"]}

    job = service.create(session_id="sess-test", job_type="inference", parameters=parameters)
    parameters["models"].append("model-b")

    assert job.parameters == {"models": ["model-a"]}


def test_job_service_rejects_second_active_job_for_session() -> None:
    service = JobService(JobStore())
    first_job = service.create_if_session_idle(
        session_id="sess-test",
        job_type="inference",
    )

    with pytest.raises(ActiveJobExistsError) as exc_info:
        service.create_if_session_idle(
            session_id="sess-test",
            job_type="inference",
        )

    assert exc_info.value.job is first_job


def test_job_service_marks_queued_job_as_running() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    previous_updated_at = job.updated_at

    result = service.mark_running(job.id, phase=JobPhase.LOADING, message="Loading model")

    assert result is job
    assert job.state == JobState.RUNNING
    assert job.phase == JobPhase.LOADING
    assert job.progress is None
    assert job.message == "Loading model"
    assert job.updated_at >= previous_updated_at

    with pytest.raises(ValueError, match="cannot transition from running to running"):
        service.mark_running(job.id, phase=JobPhase.LOADING, message="Loading model")


def test_job_service_updates_running_job_progress() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    service.mark_running(job.id, phase=JobPhase.LOADING, message="Loading model")

    result = service.update_progress(
        job.id,
        phase=JobPhase.INFERENCE,
        progress=0.42,
        message="Running inference",
    )

    assert result is job
    assert job.state == JobState.RUNNING
    assert job.phase == JobPhase.INFERENCE
    assert job.progress == 0.42
    assert job.message == "Running inference"


@pytest.mark.parametrize("progress", [-0.1, 1.1, float("nan"), True])
def test_job_service_rejects_invalid_progress(progress) -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    service.mark_running(job.id, phase=JobPhase.LOADING, message="Loading model")

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        service.update_progress(job.id, phase=JobPhase.INFERENCE, progress=progress, message="Running inference")


def test_job_service_marks_running_job_as_succeeded() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    service.mark_running(job.id, phase=JobPhase.LOADING, message="Loading model")

    result = service.mark_succeeded(job.id)

    assert result is job
    assert job.state == JobState.SUCCEEDED
    assert job.phase is None
    assert job.progress == 1.0
    assert job.message == "Inference completed"
    assert job.error is None

    with pytest.raises(ValueError, match="cannot transition from succeeded to succeeded"):
        service.mark_succeeded(job.id)


def test_job_service_marks_running_job_as_failed() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    service.mark_running(job.id, phase=JobPhase.INFERENCE, message="Running inference")
    service.update_progress(job.id, phase=JobPhase.INFERENCE, progress=0.25, message="Running inference")
    error = JobError(
        code="INFERENCE_FAILED",
        message="Model execution failed.",
        details={"model_id": "model-a"},
    )

    result = service.mark_failed(job.id, error)

    assert result is job
    assert job.state == JobState.FAILED
    assert job.phase == JobPhase.INFERENCE
    assert job.progress == 0.25
    assert job.message == "Model execution failed."
    assert job.error == error


@pytest.mark.parametrize("start_running", [False, True])
def test_job_service_cancels_active_job(start_running: bool) -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")

    if start_running:
        service.mark_running(job.id, JobPhase.INFERENCE, "Running inference")

    result = service.request_cancel(job.id)

    assert result is job
    assert job.state == JobState.CANCELLING
    assert service.has_active_job(job.session_id) is True

    service.mark_cancelled(job.id)

    assert job.state == JobState.CANCELLED
    assert job.phase is None
    assert job.progress is None
    assert service.has_active_job(job.session_id) is False


def test_job_service_rejects_cancelling_cancelled_job_and_allows_replacement() -> None:
    service = JobService(JobStore())
    job = service.create_if_session_idle(
        session_id="sess-test",
        job_type="inference",
    )

    service.request_cancel(job.id)
    service.mark_cancelled(job.id)

    with pytest.raises(ValueError, match="cannot transition from cancelled to cancelling"):
        service.request_cancel(job.id)

    replacement = service.create_if_session_idle(
        session_id="sess-test",
        job_type="inference",
    )

    assert replacement.id != job.id
    assert replacement.state == JobState.QUEUED


def test_job_service_detects_active_jobs() -> None:
    service = JobService(JobStore())
    job = service.create(
        session_id="sess-test",
        job_type="inference",
    )

    assert service.has_active_job("sess-test") is True
    assert service.has_active_job("unknown-session") is False

    service.mark_running(
        job.id,
        phase=JobPhase.LOADING,
        message="Loading model",
    )

    assert service.has_active_job("sess-test") is True

    service.mark_succeeded(job.id)

    assert service.has_active_job("sess-test") is False
