from functools import partial

from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from ..errors import APIError
from ..schemas.workspace import ImportAcceptedResponse
from ..services.job_executor import JobOperation, LocalJobExecutor
from ..services.job_service import JobService
from ..services.session_cleanup_service import SessionCleanupService
from ..services.session_service import SessionService
from ..services.workspace_import_service import WorkspaceImportService
from ..storage.temporary_storage import FileTooLargeError, TemporaryStorage
from ..stores.session_store import Session, SessionState
from ..validation.workspace import InvalidWorkspaceError, UnsupportedWorkspaceVersionError

router = APIRouter(prefix="/api/v1/sessions", tags=["imports"])


def get_import_service(request: Request) -> WorkspaceImportService:
    return request.app.state.workspace_import_service


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_storage(request: Request) -> TemporaryStorage:
    return request.app.state.temporary_storage


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def get_executor(request: Request) -> LocalJobExecutor:
    return request.app.state.job_executor


def get_cleanup_service(request: Request) -> SessionCleanupService:
    return request.app.state.session_cleanup_service


def execute_import_job(
    session: Session,
    job_id: str,
    operation: JobOperation,
    executor: LocalJobExecutor,
    import_service: WorkspaceImportService,
    cleanup_service: SessionCleanupService,
) -> None:
    try:
        job = executor.execute(job_id, operation, success_message="Import completed")
        import_service.finalize(session, job)
    finally:
        cleanup_service.finalize_pending_deletion(session.id)


@router.post("/import", response_model=ImportAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def import_workspace(
    background_tasks: BackgroundTasks,
    bundle: UploadFile = File(...),
    session_service: SessionService = Depends(get_session_service),
    storage: TemporaryStorage = Depends(get_storage),
    job_service: JobService = Depends(get_job_service),
    executor: LocalJobExecutor = Depends(get_executor),
    import_service: WorkspaceImportService = Depends(get_import_service),
    cleanup_service: SessionCleanupService = Depends(get_cleanup_service),
) -> ImportAcceptedResponse:
    session = session_service.create(source_files=[])
    session.state = SessionState.PROCESSING
    try:
        # The uploaded filename does not determine any restoration path.
        bundle.filename = "workspace.sleepwalker.zip"
        paths = await storage.save_files(session.id, [bundle])
        await run_in_threadpool(import_service.inspect, paths[0])
        job = job_service.create(session.id, "import")
    except FileTooLargeError as exc:
        cleanup_service.delete_session(session.id)
        raise APIError(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            code="FILE_TOO_LARGE",
            message="Uploaded workspace exceeds the maximum allowed size.",
            details={"max_size_bytes": exc.max_size_bytes},
        ) from exc
    except UnsupportedWorkspaceVersionError as exc:
        cleanup_service.delete_session(session.id)
        raise APIError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="UNSUPPORTED_FORMAT_VERSION",
            message=str(exc),
        ) from exc
    except (InvalidWorkspaceError, ValueError) as exc:
        cleanup_service.delete_session(session.id)
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_BUNDLE",
            message="Workspace archive or manifest is invalid.",
        ) from exc
    except Exception:
        cleanup_service.delete_session(session.id)
        raise

    operation = partial(import_service.run, session, paths[0])
    background_tasks.add_task(
        execute_import_job, session, job.id, operation, executor, import_service, cleanup_service,
    )
    return ImportAcceptedResponse(session_id=session.id, job_id=job.id, state=session.state)
