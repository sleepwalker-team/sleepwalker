"""FastAPI dependencies shared by the routes."""

from typing import Annotated

from fastapi import Depends, Path, Request

from ..core.jobs import JobExecutor, JobService
from ..core.lifecycle import SessionCleanupService
from ..core.sessions import Session, SessionService, SessionState
from ..core.storage import TemporaryStorage
from ..inference.catalog import ModelCatalog
from ..inference.runner import InferenceRunner
from ..recording.signals import EdfSignalReader
from ..results.annotations import AnnotationService
from ..results.exports import ExportService
from ..results.predictions import PredictionStore
from ..results.workspace import WorkspaceImportService
from .errors import conflict, not_found


def _from_state(name: str) -> Depends:
    def dependency(request: Request):
        return getattr(request.app.state, name)

    return Depends(dependency)


Annotations = Annotated[AnnotationService, _from_state("annotation_service")]
Catalog = Annotated[ModelCatalog, _from_state("model_catalog")]
Cleanup = Annotated[SessionCleanupService, _from_state("session_cleanup_service")]
Executor = Annotated[JobExecutor, _from_state("job_executor")]
Exports = Annotated[ExportService, _from_state("export_service")]
Importer = Annotated[WorkspaceImportService, _from_state("workspace_import_service")]
Inference = Annotated[InferenceRunner | None, _from_state("inference_runner")]
Jobs = Annotated[JobService, _from_state("job_service")]
Predictions = Annotated[PredictionStore, _from_state("prediction_store")]
Sessions = Annotated[SessionService, _from_state("session_service")]
SignalReader = Annotated[EdfSignalReader, _from_state("signal_reader")]
Storage = Annotated[TemporaryStorage, _from_state("temporary_storage")]


def _get_session(session_id: Annotated[str, Path()], sessions: Sessions) -> Session:
    session = sessions.get(session_id)
    if session is None:
        raise not_found("SESSION_NOT_FOUND", "Session not found.")
    return session


SessionDep = Annotated[Session, Depends(_get_session)]


def require_ready(session: Session, purpose: str) -> None:
    if session.state != SessionState.READY:
        raise conflict("INVALID_SESSION_STATE", f"Session is not ready for {purpose}.", {"state": session.state.value})
