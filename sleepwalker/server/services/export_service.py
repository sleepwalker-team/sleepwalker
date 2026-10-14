from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from ..storage.temporary_storage import TemporaryStorage
from ..stores.export_store import Export, ExportState, ExportStore
from ..stores.job_store import Job, JobPhase, JobState
from ..stores.session_store import Session
from .job_executor import CancellationCheck, ProgressCallback
from .result_export_service import ResultExportOptions, ResultExportService
from .workspace_export_service import WorkspaceExportService


class ExportService:
    def __init__(
        self,
        store: ExportStore,
        result_export_service: ResultExportService,
        storage: TemporaryStorage,
        workspace_export_service: WorkspaceExportService,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.result_export_service = result_export_service
        self.storage = storage
        self.workspace_export_service = workspace_export_service
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def create(
        self,
        session: Session,
        job_id: str,
        *,
        format: str = "json",
        mode: str = "results",
    ) -> Export:
        export = Export(
            id=f"export_{uuid4()}",
            session_id=session.id,
            job_id=job_id,
            state=ExportState.QUEUED,
            format=format,
            mode=mode,
            created_at=self.clock(),
            expires_at=session.expires_at,
        )
        self.store.add(export)
        return export

    def get(self, export_id: str) -> Export | None:
        return self.store.get(export_id)

    def run(
        self,
        export: Export,
        session: Session,
        options: ResultExportOptions,
        report_progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> None:
        export.state = ExportState.RUNNING
        report_progress(JobPhase.EXPORTING, 0.0, "Preparing export")
        if export.mode == "workspace":
            documents, sources = self.workspace_export_service.build_documents(session)
            export.path = self.storage.write_workspace_export(
                session.id, export.id, documents, sources, check_cancelled,
            )
        else:
            payload = self.result_export_service.build_payload(session, options)
            check_cancelled()
            export.path = self.storage.write_json_export(session.id, export.id, payload)

        export.size = export.path.stat().st_size
        report_progress(JobPhase.EXPORTING, 1.0, "Export prepared")

    def finalize(self, export_id: str, job: Job) -> Export | None:
        export = self.store.get(export_id)
        if export is None:
            # Session cleanup may run as soon as the job reaches a terminal state.
            return None

        if job.state == JobState.SUCCEEDED and export.path is not None:
            export.state = ExportState.READY
            return export

        self.storage.delete_export(export.session_id, export.id)
        export.path = None
        export.size = None
        export.state = (
            ExportState.CANCELLED
            if job.state == JobState.CANCELLED
            else ExportState.FAILED
        )
        return export

