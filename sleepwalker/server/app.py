import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI

from .api import annotations, exports, health, imports, jobs, models, predictions, sessions, signals
from .errors import register_exception_handlers
from .services.annotation_service import AnnotationService
from .services.export_service import ExportService
from .services.inference_runner import InferenceRunner
from .services.job_executor import LocalJobExecutor
from .services.job_service import JobService
from .services.model_registry import ModelRegistry, load_model_registry
from .services.package_inference_runner import PackageInferenceRunner
from .services.result_export_service import ResultExportService
from .services.session_cleanup_service import SessionCleanupService
from .services.session_service import SessionService
from .services.signal_reader import EdfSignalReader, SignalReader
from .services.workspace_export_service import WorkspaceExportService
from .services.workspace_import_service import WorkspaceImportService
from .storage.temporary_storage import TemporaryStorage
from .stores.annotation_store import AnnotationStore
from .stores.export_store import ExportStore
from .stores.job_store import JobStore
from .stores.prediction_store import PredictionStore
from .stores.session_store import SessionStore

logger = logging.getLogger(__name__)


def cleanup_expired_sessions(app: FastAPI) -> None:
    try:
        deleted_session_ids = app.state.session_cleanup_service.cleanup_expired()

        if deleted_session_ids:
            logger.info(
                "Deleted %d expired sessions.",
                len(deleted_session_ids),
            )
    except Exception:
        logger.exception("Failed to clean up expired sessions.")


async def run_session_cleanup(
    app: FastAPI,
    interval_seconds: float,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        cleanup_expired_sessions(app)


def create_lifespan(interval_seconds: float):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cleanup_expired_sessions(app)
        cleanup_task = asyncio.create_task(
            run_session_cleanup(app, interval_seconds)
        )

        try:
            yield
        finally:
            cleanup_task.cancel()

            with suppress(asyncio.CancelledError):
                await cleanup_task

    return lifespan


def create_app(
    model_registry: ModelRegistry | None = None,
    job_service: JobService | None = None,
    session_service: SessionService | None = None,
    temporary_storage: TemporaryStorage | None = None,
    inference_runner: InferenceRunner | None = None,
    prediction_store: PredictionStore | None = None,
    signal_reader: SignalReader | None = None,
    annotation_service: AnnotationService | None = None,
    export_store: ExportStore | None = None,
    cleanup_interval_seconds: float = 60.0,
) -> FastAPI:
    if cleanup_interval_seconds <= 0:
        raise ValueError("Cleanup interval must be positive.")

    app = FastAPI(
        title="Sleepwalker API",
        version="0.1.0",
        lifespan=create_lifespan(cleanup_interval_seconds),
    )

    app.state.model_registry = model_registry if model_registry is not None else ModelRegistry()
    app.state.job_service = job_service if job_service is not None else JobService(JobStore())
    app.state.session_service = (
        session_service
        if session_service is not None
        else SessionService(SessionStore())
    )
    app.state.temporary_storage = (
        temporary_storage
        if temporary_storage is not None
        else TemporaryStorage()
    )
    app.state.prediction_store = (
        prediction_store
        if prediction_store is not None
        else PredictionStore()
    )
    app.state.job_executor = LocalJobExecutor(app.state.job_service)
    app.state.inference_runner = inference_runner
    app.state.signal_reader = signal_reader if signal_reader is not None else EdfSignalReader()
    app.state.annotation_service = (
        annotation_service
        if annotation_service is not None
        else AnnotationService(AnnotationStore())
    )
    app.state.export_store = export_store if export_store is not None else ExportStore()
    app.state.result_export_service = ResultExportService(
        app.state.prediction_store,
        app.state.annotation_service,
    )
    app.state.export_service = ExportService(
        app.state.export_store,
        app.state.result_export_service,
        app.state.temporary_storage,
        WorkspaceExportService(app.state.result_export_service, app.state.model_registry),
    )
    app.state.session_cleanup_service = SessionCleanupService(
        session_service=app.state.session_service,
        job_service=app.state.job_service,
        storage=app.state.temporary_storage,
        prediction_store=app.state.prediction_store,
        annotation_service=app.state.annotation_service,
        export_store=app.state.export_store,
    )
    app.state.workspace_import_service = WorkspaceImportService(
        app.state.session_service,
        app.state.temporary_storage,
        app.state.prediction_store,
        app.state.annotation_service,
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(sessions.router)
    app.include_router(predictions.router)
    app.include_router(signals.router)
    app.include_router(annotations.router)
    app.include_router(jobs.router)
    app.include_router(exports.router)
    app.include_router(imports.router)

    return app


def create_local_app(
    model_directory: Path,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> FastAPI:
    prediction_store = PredictionStore()
    model_registry = load_model_registry(model_directory)
    inference_runner = PackageInferenceRunner(
        prediction_store,
        device=device,
        batch_size=batch_size,
    )
    return create_app(
        model_registry=model_registry,
        prediction_store=prediction_store,
        inference_runner=inference_runner,
    )


app = create_app()
