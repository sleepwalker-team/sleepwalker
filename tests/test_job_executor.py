from sleepwalker.server.services.job_executor import JobExecutionError, LocalJobExecutor
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.stores.job_store import JobPhase, JobState, JobStore


def test_executor_completes_successful_job() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    executor = LocalJobExecutor(service)
    operation_was_called = False

    def operation(report_progress, check_cancelled) -> None:
        nonlocal operation_was_called
        operation_was_called = True
        check_cancelled()
        report_progress(JobPhase.PREPROCESSING, None, "Preparing EDF data")
        report_progress(JobPhase.INFERENCE, 0.5, "Running inference")

    result = executor.execute(job.id, operation)

    assert operation_was_called
    assert result is job
    assert job.state == JobState.SUCCEEDED
    assert job.phase is None
    assert job.progress == 1.0
    assert job.message == "Inference completed"
    assert job.error is None


def test_executor_exposes_safe_execution_error() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    executor = LocalJobExecutor(service)

    def operation(_report_progress, _check_cancelled) -> None:
        raise JobExecutionError(
            code="MODEL_INCOMPATIBLE",
            message="Model is incompatible with the recording.",
            details={"model_id": "model-a"},
        )

    executor.execute(job.id, operation)

    assert job.state == JobState.FAILED
    assert job.error is not None
    assert job.error.code == "MODEL_INCOMPATIBLE"
    assert job.error.message == "Model is incompatible with the recording."
    assert job.error.details == {"model_id": "model-a"}


def test_executor_hides_unexpected_error_details() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    executor = LocalJobExecutor(service)

    def operation(_report_progress, _check_cancelled) -> None:
        raise RuntimeError("secret implementation detail")

    executor.execute(job.id, operation)

    assert job.state == JobState.FAILED
    assert job.error is not None
    assert job.error.code == "INTERNAL_ERROR"
    assert "secret implementation detail" not in job.error.message


def test_executor_acknowledges_cancellation_without_storing_success() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    executor = LocalJobExecutor(service)

    def operation(_report_progress, check_cancelled) -> None:
        service.request_cancel(job.id)
        check_cancelled()

    result = executor.execute(job.id, operation)

    assert result is job
    assert job.state == JobState.CANCELLED
    assert job.progress is None
    assert job.message == "Cancelled"


def test_executor_does_not_start_job_with_pending_cancellation() -> None:
    service = JobService(JobStore())
    job = service.create(session_id="sess-test", job_type="inference")
    service.request_cancel(job.id)
    executor = LocalJobExecutor(service)
    operation_was_called = False

    def operation(_report_progress, _check_cancelled) -> None:
        nonlocal operation_was_called
        operation_was_called = True

    executor.execute(job.id, operation)

    assert operation_was_called is False
    assert job.state == JobState.CANCELLED
