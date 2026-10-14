from functools import partial

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from fastapi.responses import FileResponse

from ..errors import APIError
from ..schemas.export import (
    ExportAcceptedResponse,
    ExportRequest,
    ExportResponse,
)
from ..services.export_service import ExportService
from ..services.job_executor import JobOperation, LocalJobExecutor
from ..services.job_service import ActiveJobExistsError, JobService
from ..services.result_export_service import ResultExportOptions
from ..services.session_cleanup_service import SessionCleanupService
from ..services.session_service import SessionService
from ..stores.export_store import Export, ExportState
from ..stores.session_store import SessionState

router = APIRouter(
    prefix="/api/v1",
    tags=["exports"],
)


def get_export_service(request: Request) -> ExportService:
    return request.app.state.export_service


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def get_job_executor(request: Request) -> LocalJobExecutor:
    return request.app.state.job_executor


def get_session_cleanup_service(request: Request) -> SessionCleanupService:
    return request.app.state.session_cleanup_service


def execute_export_job(
    job_id: str,
    export_id: str,
    session_id: str,
    operation: JobOperation,
    job_executor: LocalJobExecutor,
    export_service: ExportService,
    cleanup_service: SessionCleanupService,
) -> None:
    try:
        job = job_executor.execute(
            job_id,
            operation,
            success_message="Export completed",
        )
        export_service.finalize(export_id, job)
    finally:
        cleanup_service.finalize_pending_deletion(session_id)


@router.post(
    "/sessions/{session_id}/exports",
    response_model=ExportAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_export(
    session_id: str,
    payload: ExportRequest,
    background_tasks: BackgroundTasks,
    session_service: SessionService = Depends(get_session_service),
    job_service: JobService = Depends(get_job_service),
    job_executor: LocalJobExecutor = Depends(get_job_executor),
    export_service: ExportService = Depends(get_export_service),
    cleanup_service: SessionCleanupService = Depends(get_session_cleanup_service),
) -> ExportAcceptedResponse:
    session = session_service.get(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    if session.state != SessionState.READY:
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="INVALID_SESSION_STATE",
            message="Session is not ready for export.",
            details={"state": session.state.value},
        )

    parameters = {
        "format": payload.format.value,
        "mode": payload.mode.value,
        "include": payload.include.model_dump(),
    }

    try:
        job = job_service.create_if_session_idle(session.id, "export", parameters)
    except ActiveJobExistsError as exc:
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="JOB_ALREADY_RUNNING",
            message="Session already has an active job.",
            details={"job_id": exc.job.id, "state": exc.job.state.value},
        ) from exc

    export = export_service.create(
        session, job.id, format=payload.format.value, mode=payload.mode.value,
    )
    options = ResultExportOptions(**payload.include.model_dump())
    operation = partial(export_service.run, export, session, options)
    background_tasks.add_task(
        execute_export_job,
        job.id,
        export.id,
        session.id,
        operation,
        job_executor,
        export_service,
        cleanup_service,
    )
    return ExportAcceptedResponse(export_id=export.id, job_id=job.id, state=job.state)


@router.get("/exports/{export_id}", response_model=ExportResponse)
def get_export(
    export_id: str,
    service: ExportService = Depends(get_export_service),
) -> ExportResponse:
    return _to_export_response(_get_export_or_404(service, export_id))


@router.get("/exports/{export_id}/download")
def download_export(
    export_id: str,
    service: ExportService = Depends(get_export_service),
) -> FileResponse:
    export = _get_export_or_404(service, export_id)

    if export.state != ExportState.READY or export.path is None:
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="EXPORT_NOT_READY",
            message="Export is not ready for download.",
            details={"state": export.state.value},
        )

    workspace = export.mode == "workspace"
    return FileResponse(
        export.path,
        media_type="application/zip" if workspace else "application/json",
        filename=f"{export.id}.sleepwalker.zip" if workspace else f"{export.id}.json",
    )


def _get_export_or_404(service: ExportService, export_id: str) -> Export:
    export = service.get(export_id)

    if export is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="EXPORT_NOT_FOUND",
            message="Export not found.",
        )

    return export


def _to_export_response(export: Export) -> ExportResponse:
    return ExportResponse(
        id=export.id,
        state=export.state,
        format=export.format,
        mode=export.mode,
        size=export.size,
        created_at=export.created_at,
        expires_at=export.expires_at,
    )
