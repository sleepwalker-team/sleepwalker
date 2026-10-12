from functools import partial

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Request,
    UploadFile,
    status,
)

from ..errors import APIError
from ..schemas.job import InferenceRequest, JobAcceptedResponse
from ..schemas.prediction import TaskResponse
from ..schemas.session import SessionResponse, SignalResponse
from ..services.annotation_service import AnnotationService
from ..services.inference_runner import InferenceRunner
from ..services.job_executor import JobOperation, LocalJobExecutor
from ..services.job_service import ActiveJobExistsError, JobService
from ..services.model_registry import ModelRegistry
from ..services.session_cleanup_service import SessionCleanupService
from ..services.session_service import SessionService
from ..storage.temporary_storage import TemporaryStorage
from ..stores.prediction_store import PredictionStore
from ..stores.session_store import Session, SessionState
from ..validation import InvalidStudyError, UnsupportedStudyFormatError, validate_study
from .predictions import to_task_response

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
)


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_temporary_storage(request: Request) -> TemporaryStorage:
    return request.app.state.temporary_storage


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def get_job_executor(request: Request) -> LocalJobExecutor:
    return request.app.state.job_executor


def get_inference_runner(request: Request) -> InferenceRunner | None:
    return request.app.state.inference_runner


def get_prediction_store(request: Request) -> PredictionStore:
    return request.app.state.prediction_store


def get_annotation_service(request: Request) -> AnnotationService:
    return request.app.state.annotation_service


def get_session_cleanup_service(request: Request) -> SessionCleanupService:
    return request.app.state.session_cleanup_service


def execute_inference_job(
    job_id: str,
    session_id: str,
    operation: JobOperation,
    job_executor: LocalJobExecutor,
    cleanup_service: SessionCleanupService,
) -> None:
    try:
        job_executor.execute(job_id, operation)
    finally:
        cleanup_service.finalize_pending_deletion(session_id)


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    files: list[UploadFile] = File(...),
    service: SessionService = Depends(get_session_service),
    storage: TemporaryStorage = Depends(get_temporary_storage),
) -> SessionResponse:
    session = service.create(source_files=[])
    session.state = SessionState.PROCESSING

    try:
        # Uploads temporär speichern
        saved_files = await storage.save_files(session.id, files)

        # Gesamte Study erkennen und validieren
        study_info = validate_study(saved_files)

        session.source_files = study_info.source_files
        session.format = study_info.format
        session.duration = study_info.duration
        session.recording_start = study_info.recording_start
        session.signals = study_info.signals
        session.state = SessionState.READY

    except (InvalidStudyError, UnsupportedStudyFormatError) as exc:
        storage.delete_session(session.id)
        service.delete(session.id)

        raise APIError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="INVALID_STUDY",
            message=str(exc),
        ) from exc

    except ValueError as exc:
        # z.B. doppelter Dateiname oder ungültiger Upload
        storage.delete_session(session.id)
        service.delete(session.id)

        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message=str(exc),
        ) from exc

    except Exception:
        # Auch bei unerwarteten Fehlern keine halbfertige Session und keine
        # temporären Dateien liegen lassen.
        storage.delete_session(session.id)
        service.delete(session.id)
        raise

    return to_session_response(session, [])


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
)
def get_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
    prediction_store: PredictionStore = Depends(get_prediction_store),
) -> SessionResponse:
    session = service.get(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    tasks = [
        to_task_response(result)
        for result in prediction_store.list_for_session(session.id)
    ]

    return to_session_response(session, tasks)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_session(
    session_id: str,
    cleanup_service: SessionCleanupService = Depends(get_session_cleanup_service),
) -> None:
    if not cleanup_service.request_delete(session_id):
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )


@router.post(
    "/{session_id}/inference",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_inference(
    session_id: str,
    payload: InferenceRequest,
    background_tasks: BackgroundTasks,
    session_service: SessionService = Depends(get_session_service),
    job_service: JobService = Depends(get_job_service),
    model_registry: ModelRegistry = Depends(get_model_registry),
    job_executor: LocalJobExecutor = Depends(get_job_executor),
    inference_runner: InferenceRunner | None = Depends(get_inference_runner),
    prediction_store: PredictionStore = Depends(get_prediction_store),
    cleanup_service: SessionCleanupService = Depends(get_session_cleanup_service),
) -> JobAcceptedResponse:
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
            message="Session is not ready for inference.",
            details={"state": session.state.value},
        )

    if not payload.models or any(
        not model_id.strip() for model_id in payload.models
    ):
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="At least one valid model ID is required.",
        )

    if len(payload.models) != len(set(payload.models)):
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Model IDs must be unique.",
        )

    models = []

    for model_id in payload.models:
        model = model_registry.get(model_id)

        if model is None:
            raise APIError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="MODEL_NOT_FOUND",
                message="Model not found.",
                details={"model_id": model_id},
            )

        models.append(model)

    models_by_task = {}

    for model in models:
        models_by_task.setdefault(model.task, []).append(model.id)

    duplicate_task = next(
        (
            task
            for task, model_ids in models_by_task.items()
            if len(model_ids) > 1
        ),
        None,
    )

    if duplicate_task is not None:
        raise APIError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="MODEL_INCOMPATIBLE",
            message="Multiple models cannot produce the same task in one session.",
            details={
                "task": duplicate_task,
                "model_ids": models_by_task[duplicate_task],
            },
        )

    existing_task = next(
        (
            model.task
            for model in models
            if prediction_store.get_by_task(session.id, model.task)
        ),
        None,
    )

    if existing_task is not None:
        raise APIError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="MODEL_INCOMPATIBLE",
            message="Predictions already exist for this task.",
            details={"task": existing_task},
        )

    available_signals = {
        name.strip().casefold()
        for signal in session.signals or []
        for name in (signal.id, signal.label)
    }

    for model in models:
        missing_channels = [
            channel
            for channel in model.input_channels
            if channel.strip().casefold() not in available_signals
        ]

        if missing_channels:
            raise APIError(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                code="MISSING_REQUIRED_CHANNEL",
                message="Required signal could not be resolved.",
                details={"model_id": model.id, "channels": missing_channels},
            )

    if inference_runner is None:
        raise APIError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="COMPUTE_UNAVAILABLE",
            message="Inference execution is not available.",
        )

    try:
        job = job_service.create_if_session_idle(
            session_id=session_id,
            job_type="inference",
            parameters={"models": payload.models},
        )
    except ActiveJobExistsError as exc:
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="JOB_ALREADY_RUNNING",
            message="Session already has an active job.",
            details={
                "job_id": exc.job.id,
                "state": exc.job.state.value,
            },
        ) from exc
    operation = partial(inference_runner.run, session, models)
    background_tasks.add_task(
        execute_inference_job,
        job.id,
        session.id,
        operation,
        job_executor,
        cleanup_service,
    )

    return JobAcceptedResponse(job_id=job.id, state=job.state)


def to_session_response(session: Session, tasks: list[TaskResponse]) -> SessionResponse:
    return SessionResponse(
        id=session.id,
        state=session.state,
        tasks=tasks,
        created_at=session.created_at,
        last_accessed_at=session.last_accessed_at,
        expires_at=session.expires_at,
        format=session.format,
        duration=session.duration,
        recording_start=session.recording_start,
        signals=(
            [SignalResponse(**vars(signal)) for signal in session.signals]
            if session.signals is not None
            else None
        ),
    )
