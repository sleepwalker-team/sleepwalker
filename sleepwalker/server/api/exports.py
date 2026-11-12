from functools import partial

from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import FileResponse

from ..core.jobs import ActiveJobExistsError
from ..core.lifecycle import run_session_job
from ..results.exports import Export, ExportService, ExportState, ResultExportOptions
from .deps import Cleanup, Executor, Exports, Jobs, SessionDep, require_ready
from .errors import conflict, not_found
from .schemas import ExportAcceptedResponse, ExportRequest, ExportResponse
from .sessions import job_already_running

router = APIRouter(prefix="/api/v1", tags=["exports"])


@router.post("/sessions/{session_id}/exports", response_model=ExportAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def create_export(
    session: SessionDep,
    payload: ExportRequest,
    background_tasks: BackgroundTasks,
    jobs: Jobs,
    executor: Executor,
    exports: Exports,
    cleanup: Cleanup,
) -> ExportAcceptedResponse:
    require_ready(session, "export")

    try:
        job = jobs.create_if_session_idle(session.id, "export")
    except ActiveJobExistsError as exc:
        raise job_already_running(exc) from exc

    export = exports.create(session, job.id, format=payload.format.value, mode=payload.mode.value)
    options = ResultExportOptions(**payload.include.model_dump())
    background_tasks.add_task(
        run_session_job, job.id, session.id, partial(exports.run, export, session, options), executor, cleanup,
        success_message="Export completed", on_finished=partial(exports.finalize, export.id),
    )
    return ExportAcceptedResponse(export_id=export.id, job_id=job.id, state=job.state)


@router.get("/exports/{export_id}", response_model=ExportResponse)
def get_export(export_id: str, exports: Exports) -> ExportResponse:
    export = _get_export(exports, export_id)
    return ExportResponse(
        id=export.id,
        state=export.state,
        format=export.format,
        mode=export.mode,
        size=export.size,
        created_at=export.created_at,
        expires_at=export.expires_at,
    )


@router.get("/exports/{export_id}/download")
def download_export(export_id: str, exports: Exports) -> FileResponse:
    export = _get_export(exports, export_id)
    if export.state != ExportState.READY or export.path is None:
        raise conflict("EXPORT_NOT_READY", "Export is not ready for download.", {"state": export.state.value})

    workspace = export.mode == "workspace"
    return FileResponse(
        export.path,
        media_type="application/zip" if workspace else "application/json",
        filename=f"{export.id}.sleepwalker.zip" if workspace else f"{export.id}.json",
    )


def _get_export(exports: ExportService, export_id: str) -> Export:
    export = exports.get(export_id)
    if export is None:
        raise not_found("EXPORT_NOT_FOUND", "Export not found.")
    return export
