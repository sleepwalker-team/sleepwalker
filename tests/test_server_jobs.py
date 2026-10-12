import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.stores.job_store import JobError, JobPhase, JobState, JobStore


def test_get_job_returns_queued_job() -> None:
    service = JobService(JobStore())
    job = service.create(
        session_id="sess-test",
        job_type="inference",
    )

    with TestClient(
        create_app(job_service=service)
    ) as client:
        response = client.get(f"/api/v1/jobs/{job.id}")

    assert response.status_code == 200

    body = response.json()
    assert body["id"] == job.id
    assert body["type"] == "inference"
    assert body["state"] == "queued"
    assert body["phase"] is None
    assert body["progress"] is None
    assert body["message"] == "Waiting for execution"
    assert body["error"] is None
    assert body["updated_at"] == job.updated_at.isoformat().replace(
        "+00:00",
        "Z",
    )


def test_get_unknown_job_returns_not_found() -> None:
    service = JobService(JobStore())

    with TestClient(
        create_app(job_service=service)
    ) as client:
        response = client.get("/api/v1/jobs/unknown")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "JOB_NOT_FOUND",
            "message": "Job not found.",
            "details": None,
        }
    }


def test_get_job_returns_structured_failure() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    service.mark_running(job.id, phase=JobPhase.INFERENCE, message="Running inference")
    service.mark_failed(
        job.id,
        JobError(
            code="INFERENCE_FAILED",
            message="Model execution failed.",
            details={"model_id": "model-a"},
        ),
    )

    with TestClient(create_app(job_service=service)) as client:
        response = client.get(f"/api/v1/jobs/{job.id}")

    assert response.status_code == 200
    assert response.json()["state"] == "failed"
    assert response.json()["error"] == {
        "code": "INFERENCE_FAILED",
        "message": "Model execution failed.",
        "details": {"model_id": "model-a"},
    }


def test_cancel_job_requests_cancellation_and_is_idempotent() -> None:
    service = JobService(JobStore())
    job = service.create("sess-test", "inference", {"model_ids": ["demo-model"]})
    app = create_app(job_service=service)

    with TestClient(app) as client:
        first_response = client.post(f"/api/v1/jobs/{job.id}/cancel")
        second_response = client.post(f"/api/v1/jobs/{job.id}/cancel")

    assert first_response.status_code == 202
    assert second_response.status_code == 202
    assert first_response.json()["state"] == "cancelling"
    assert first_response.json()["message"] == "Cancellation requested"
    assert second_response.json() == first_response.json()
    assert service.get(job.id).state == JobState.CANCELLING


def test_cancel_unknown_job_returns_not_found() -> None:
    app = create_app(job_service=JobService(JobStore()))

    with TestClient(app) as client:
        response = client.post("/api/v1/jobs/job-missing/cancel")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "JOB_NOT_FOUND",
            "message": "Job not found.",
            "details": None,
        }
    }


@pytest.mark.parametrize(
    "terminal_state",
    [JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED],
)
def test_cancel_terminal_job_returns_conflict(terminal_state: JobState) -> None:
    service = JobService(JobStore())
    job = service.create("sess-test", "inference", {"model_ids": ["demo-model"]})

    if terminal_state == JobState.CANCELLED:
        service.request_cancel(job.id)
        service.mark_cancelled(job.id)
    else:
        service.mark_running(job.id, JobPhase.INFERENCE, "Running inference")
        if terminal_state == JobState.SUCCEEDED:
            service.mark_succeeded(job.id)
        else:
            service.mark_failed(job.id, JobError(code="INFERENCE_FAILED", message="failed"))

    app = create_app(job_service=service)

    with TestClient(app) as client:
        response = client.post(f"/api/v1/jobs/{job.id}/cancel")

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "INVALID_JOB_STATE",
            "message": "Job cannot be cancelled in its current state.",
            "details": {"state": terminal_state.value},
        }
    }
