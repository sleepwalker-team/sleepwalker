import logging
from collections.abc import Callable
from typing import Any

from .job_service import JobService
from ..stores.job_store import Job, JobError, JobPhase


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[JobPhase, float | None, str], None]
CancellationCheck = Callable[[], None]
JobOperation = Callable[[ProgressCallback, CancellationCheck], None]


class JobExecutionError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class JobCancelledError(Exception):
    pass


class LocalJobExecutor:
    def __init__(self, job_service: JobService) -> None:
        self.job_service = job_service

    def execute(self, job_id: str, operation: JobOperation) -> Job:
        if self.job_service.is_cancellation_requested(job_id):
            return self.job_service.mark_cancelled(job_id)

        try:
            self.job_service.mark_running(
                job_id,
                phase=JobPhase.LOADING,
                message="Loading resources",
            )
        except ValueError:
            if self.job_service.is_cancellation_requested(job_id):
                return self.job_service.mark_cancelled(job_id)

            raise

        def check_cancelled() -> None:
            if self.job_service.is_cancellation_requested(job_id):
                raise JobCancelledError

        def report_progress(
            phase: JobPhase,
            progress: float | None,
            message: str,
        ) -> None:
            check_cancelled()
            self.job_service.update_progress(
                job_id,
                phase=phase,
                progress=progress,
                message=message,
            )

        try:
            operation(report_progress, check_cancelled)
            check_cancelled()
        except JobCancelledError:
            return self.job_service.mark_cancelled(job_id)
        except JobExecutionError as exc:
            if self.job_service.is_cancellation_requested(job_id):
                return self.job_service.mark_cancelled(job_id)

            return self.job_service.mark_failed(
                job_id,
                JobError(code=exc.code, message=exc.message, details=exc.details),
            )
        except Exception:
            if self.job_service.is_cancellation_requested(job_id):
                return self.job_service.mark_cancelled(job_id)

            logger.exception("Unexpected error while executing job %s", job_id)
            return self.job_service.mark_failed(
                job_id,
                JobError(
                    code="INTERNAL_ERROR",
                    message="An unexpected error occurred while executing the job.",
                ),
            )

        try:
            return self.job_service.mark_succeeded(job_id)
        except ValueError:
            if self.job_service.is_cancellation_requested(job_id):
                return self.job_service.mark_cancelled(job_id)

            raise
