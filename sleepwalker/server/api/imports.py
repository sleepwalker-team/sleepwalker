from functools import partial

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, status
from starlette.concurrency import run_in_threadpool

from ..core.lifecycle import run_session_job
from ..core.sessions import SessionState
from ..core.storage import FileTooLargeError
from ..results.workspace import InvalidWorkspaceError, UnsupportedWorkspaceVersionError
from .deps import Cleanup, Executor, Importer, Jobs, Sessions, Storage
from .errors import APIError, bad_request, unprocessable
from .schemas import ImportAcceptedResponse

router = APIRouter(prefix="/api/v1/sessions", tags=["imports"])


@router.post("/import", response_model=ImportAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def import_workspace(
    background_tasks: BackgroundTasks,
    sessions: Sessions,
    storage: Storage,
    jobs: Jobs,
    executor: Executor,
    importer: Importer,
    cleanup: Cleanup,
    bundle: UploadFile = File(...),
) -> ImportAcceptedResponse:
    session = sessions.create(source_files=[])
    session.state = SessionState.PROCESSING
    try:
        # The uploaded filename does not determine any restoration path.
        bundle.filename = "workspace.sleepwalker.zip"
        paths = await storage.save_files(session.id, [bundle])
        await run_in_threadpool(importer.inspect, paths[0])
        job = jobs.create(session.id, "import")
    except BaseException as exc:
        cleanup.delete_session(session.id)
        if isinstance(exc, FileTooLargeError):
            raise APIError(
                status.HTTP_413_CONTENT_TOO_LARGE, "FILE_TOO_LARGE", "Uploaded workspace exceeds the maximum allowed size.",
                {"max_size_bytes": exc.max_size_bytes},
            ) from exc
        if isinstance(exc, UnsupportedWorkspaceVersionError):
            raise unprocessable("UNSUPPORTED_FORMAT_VERSION", str(exc)) from exc
        if isinstance(exc, (InvalidWorkspaceError, ValueError)):
            raise bad_request("Workspace archive or manifest is invalid.", code="INVALID_BUNDLE") from exc
        raise

    background_tasks.add_task(
        run_session_job, job.id, session.id, partial(importer.run, session, paths[0]), executor, cleanup,
        success_message="Import completed", on_finished=partial(importer.finalize, session),
    )
    return ImportAcceptedResponse(session_id=session.id, job_id=job.id, state=session.state)
