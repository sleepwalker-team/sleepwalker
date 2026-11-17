"""Application wiring and background session cleanup."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI

from .api import annotations, exports, health, imports, jobs, models, predictions, sessions, signals
from .api.errors import register_exception_handlers
from .core.jobs import JobExecutor, JobService
from .core.lifecycle import SessionCleanupService
from .core.sessions import SessionService
from .core.storage import TemporaryStorage
from .inference.catalog import ModelCatalog
from .inference.runner import InferenceRunner, PackageInferenceRunner
from .recording.plots import PlotlyService
from .recording.signals import EdfSignalReader
from .results.annotations import AnnotationService
from .results.exports import ExportService
from .results.predictions import PredictionStore
from .results.workspace import WorkspaceImportService

logger = logging.getLogger(__name__)
CLEANUP_INTERVAL_SECONDS = 60.0


async def run_in_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Do not release storage ownership while file deletion is still running.
        await task
        raise


def _cleanup_expired_sessions(app: FastAPI) -> None:
    try:
        deleted = app.state.session_cleanup_service.cleanup_expired()
        if deleted:
            logger.info("Deleted %d expired sessions.", len(deleted))
    except Exception:
        logger.exception("Failed to clean up expired sessions.")


async def _cleanup_periodically(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        await run_in_thread(_cleanup_expired_sessions, app)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    storage = app.state.temporary_storage
    with storage.exclusive_access():
        known_ids = {session.id for session in app.state.session_service.list()}
        orphan_ids = await run_in_thread(storage.cleanup_orphan_sessions, known_ids)
        if orphan_ids:
            logger.info("Deleted %d orphan temporary sessions from an earlier server run.", len(orphan_ids))

        await run_in_thread(_cleanup_expired_sessions, app)
        cleanup_task = asyncio.create_task(_cleanup_periodically(app))
        try:
            yield
        finally:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
            app.state.plotly_service.clear()


def create_app(
    *,
    model_catalog: ModelCatalog | None = None,
    inference_runner: InferenceRunner | None = None,
    prediction_store: PredictionStore | None = None,
    temporary_storage: TemporaryStorage | None = None,
    session_service: SessionService | None = None,
    job_service: JobService | None = None,
    annotation_service: AnnotationService | None = None,
    signal_reader: EdfSignalReader | None = None,
) -> FastAPI:
    """Build the API. Without a catalog and runner it serves everything but inference."""
    app = FastAPI(title="Sleepwalker API", version="0.1.0", lifespan=_lifespan)

    state = app.state
    state.model_catalog = model_catalog or ModelCatalog()
    state.inference_runner = inference_runner
    state.prediction_store = prediction_store if prediction_store is not None else PredictionStore()
    state.temporary_storage = temporary_storage or TemporaryStorage()
    state.session_service = session_service or SessionService()
    state.job_service = job_service or JobService()
    state.annotation_service = annotation_service or AnnotationService()
    state.signal_reader = signal_reader or EdfSignalReader()

    # Sessions stay alive during a job, and a finished job renews its session.
    state.session_service.active_job_check = state.job_service.has_active_job
    state.job_service.on_completion = state.session_service.renew_after_job

    state.job_executor = JobExecutor(state.job_service)
    state.plotly_service = PlotlyService(state.session_service)
    state.export_service = ExportService(state.temporary_storage, state.prediction_store, state.annotation_service)
    state.session_cleanup_service = SessionCleanupService(
        session_service=state.session_service,
        job_service=state.job_service,
        storage=state.temporary_storage,
        prediction_store=state.prediction_store,
        annotation_service=state.annotation_service,
        export_service=state.export_service,
        plotly_service=state.plotly_service,
    )
    state.workspace_import_service = WorkspaceImportService(
        state.session_service, state.temporary_storage, state.prediction_store, state.annotation_service,
    )

    register_exception_handlers(app)
    for module in (health, models, sessions, predictions, signals, annotations, jobs, exports, imports):
        app.include_router(module.router)

    return app


def create_local_app(model_directory: Path, *, inference_python: Path, device: str = "cpu", batch_size: int = 1) -> FastAPI:
    """Build the API with inference: models from a directory, run by a separate Python."""
    prediction_store = PredictionStore()
    return create_app(
        model_catalog=ModelCatalog(model_directory),
        prediction_store=prediction_store,
        inference_runner=PackageInferenceRunner(prediction_store, inference_python=inference_python, device=device, batch_size=batch_size),
    )
