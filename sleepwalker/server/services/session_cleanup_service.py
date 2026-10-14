from threading import RLock

from .annotation_service import AnnotationService
from .job_service import JobService
from .session_service import SessionService
from ..storage.temporary_storage import TemporaryStorage
from ..stores.export_store import ExportStore
from ..stores.prediction_store import PredictionStore


class SessionCleanupService:
    def __init__(
        self,
        *,
        session_service: SessionService,
        job_service: JobService,
        storage: TemporaryStorage,
        prediction_store: PredictionStore,
        annotation_service: AnnotationService,
        export_store: ExportStore,
    ) -> None:
        self.session_service = session_service
        self.job_service = job_service
        self.storage = storage
        self.prediction_store = prediction_store
        self.annotation_service = annotation_service
        self.export_store = export_store
        self._pending_deletions: set[str] = set()
        self._lock = RLock()

    def cleanup_expired(self) -> list[str]:
        deleted_session_ids = []

        for session in self.session_service.list_expired():
            if self.job_service.has_active_job(session.id):
                continue

            if self.delete_session(session.id):
                deleted_session_ids.append(session.id)

        return deleted_session_ids

    def delete_session(self, session_id: str) -> bool:
        session = self.session_service.delete(session_id)

        if session is None:
            return False

        self._delete_session_resources(session_id)
        return True

    def request_delete(self, session_id: str) -> bool:
        with self._lock:
            session = self.session_service.delete(session_id)

            if session is None:
                return False

            active_jobs = self.job_service.list_active_jobs(session_id)

            if not active_jobs:
                self._delete_session_resources(session_id)
                return True

            self._pending_deletions.add(session_id)

            for job in active_jobs:
                try:
                    self.job_service.request_cancel(job.id)
                except ValueError:
                    # The job reached a terminal state between listing and
                    # requesting cancellation.
                    pass

            self.finalize_pending_deletion(session_id)
            return True

    def finalize_pending_deletion(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._pending_deletions:
                return False

            if self.job_service.has_active_job(session_id):
                return False

            self._delete_session_resources(session_id)
            self._pending_deletions.remove(session_id)
            return True

    def _delete_session_resources(self, session_id: str) -> None:
        self.storage.delete_session(session_id)
        self.prediction_store.delete_session(session_id)
        self.annotation_service.delete_session(session_id)
        self.export_store.delete_session(session_id)
        self.job_service.delete_session(session_id)
