"""Deleting a session's data, and running jobs so that a pending deletion completes."""

from collections.abc import Callable
from threading import RLock

from ..recording.plots import PlotlyService
from ..results.annotations import AnnotationService
from ..results.exports import ExportService
from ..results.predictions import PredictionStore
from .jobs import Job, JobExecutor, JobOperation, JobService
from .sessions import SessionService
from .storage import TemporaryStorage


class SessionCleanupService:
    def __init__(
        self,
        *,
        session_service: SessionService,
        job_service: JobService,
        storage: TemporaryStorage,
        prediction_store: PredictionStore,
        annotation_service: AnnotationService,
        export_service: ExportService,
        plotly_service: PlotlyService,
    ) -> None:
        self.session_service = session_service
        self.job_service = job_service
        self.storage = storage
        self.prediction_store = prediction_store
        self.annotation_service = annotation_service
        self.export_service = export_service
        self.plotly_service = plotly_service
        self._pending_deletions: set[str] = set()
        self._lock = RLock()

    def cleanup_expired(self) -> list[str]:
        deleted_session_ids = []
        now = self.session_service.clock()

        with self._lock:
            for session in self.session_service.list_expired(at=now):
                if self.job_service.has_active_job(session.id):
                    continue

                # A finishing job may have renewed the deadline after listing.
                if session.expires_at > now:
                    continue

                if self.delete_session(session.id):
                    deleted_session_ids.append(session.id)

        return deleted_session_ids

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if self.session_service.delete(session_id) is None:
                return False

            self._delete_session_resources(session_id)
            return True

    def request_delete(self, session_id: str) -> bool:
        """Delete now, or cancel active jobs and delete once the last one has stopped."""
        with self._lock:
            if self.session_service.delete(session_id) is None:
                return False

            self.plotly_service.delete_session(session_id)
            self._pending_deletions.add(session_id)

            for job in self.job_service.list_active_jobs(session_id):
                try:
                    self.job_service.request_cancel(job.id)
                except ValueError:
                    # The job finished between listing and cancelling it.
                    pass

            self.finalize_pending_deletion(session_id)
            return True

    def finalize_pending_deletion(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._pending_deletions or self.job_service.has_active_job(session_id):
                return False

            self._delete_session_resources(session_id)
            self._pending_deletions.remove(session_id)
            return True

    def _delete_session_resources(self, session_id: str) -> None:
        self.plotly_service.delete_session(session_id)
        self.storage.delete_session(session_id)
        self.prediction_store.delete_session(session_id)
        self.annotation_service.delete_session(session_id)
        self.export_service.delete_session(session_id)
        self.job_service.delete_session(session_id)


def run_session_job(
    job_id: str,
    session_id: str,
    operation: JobOperation,
    executor: JobExecutor,
    cleanup_service: SessionCleanupService,
    *,
    success_message: str = "Inference completed",
    on_finished: Callable[[Job], object] | None = None,
) -> None:
    """Background task body shared by inference, export and import."""
    try:
        job = executor.execute(job_id, operation, success_message=success_message)
        if on_finished is not None:
            on_finished(job)
    finally:
        cleanup_service.finalize_pending_deletion(session_id)
