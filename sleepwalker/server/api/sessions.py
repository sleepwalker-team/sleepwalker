from functools import partial

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, status
from starlette.concurrency import run_in_threadpool

from ..core.jobs import ActiveJobExistsError
from ..core.lifecycle import run_session_job
from ..core.sessions import Session, SessionState
from ..core.storage import FileTooLargeError
from ..inference.catalog import ModelCatalog, ModelInfo
from ..recording.study import InvalidStudyError, UnsupportedStudyFormatError, validate_study
from ..results.predictions import PredictionStore
from .deps import Catalog, Cleanup, Executor, Inference, Jobs, Predictions, SessionDep, Sessions, Storage, require_ready
from .errors import APIError, bad_request, conflict, not_found, unprocessable
from .predictions import to_task_response
from .schemas import InferenceRequest, JobAcceptedResponse, SessionResponse, SignalResponse, TaskResponse

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(sessions: Sessions, storage: Storage, files: list[UploadFile] = File(...)) -> SessionResponse:
    session = sessions.create(source_files=[])
    session.state = SessionState.PROCESSING
    try:
        saved_files = await storage.save_files(session.id, files)
        session.apply_study(await run_in_threadpool(validate_study, saved_files))
        session.state = SessionState.READY
    except BaseException as exc:
        # Never leave a half-created session or its temporary files behind.
        storage.delete_session(session.id)
        sessions.delete(session.id)
        if isinstance(exc, FileTooLargeError):
            raise APIError(
                status.HTTP_413_CONTENT_TOO_LARGE, "FILE_TOO_LARGE", "Uploaded file exceeds the maximum allowed size.",
                {"filename": exc.filename, "max_size_bytes": exc.max_size_bytes},
            ) from exc
        if isinstance(exc, (InvalidStudyError, UnsupportedStudyFormatError)):
            raise unprocessable("INVALID_STUDY", str(exc)) from exc
        if isinstance(exc, ValueError):
            raise bad_request(str(exc)) from exc
        raise

    return to_session_response(session, [])


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session: SessionDep, predictions: Predictions) -> SessionResponse:
    tasks = [to_task_response(result) for result in predictions.list_for_session(session.id)]
    return to_session_response(session, tasks)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, cleanup: Cleanup) -> None:
    if not cleanup.request_delete(session_id):
        raise not_found("SESSION_NOT_FOUND", "Session not found.")


@router.post("/{session_id}/inference", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def start_inference(
    session: SessionDep,
    payload: InferenceRequest,
    background_tasks: BackgroundTasks,
    jobs: Jobs,
    catalog: Catalog,
    executor: Executor,
    runner: Inference,
    predictions: Predictions,
    cleanup: Cleanup,
) -> JobAcceptedResponse:
    require_ready(session, "inference")
    # Resolved once here: this job keeps using these packages even if a newer one appears.
    models = _resolve_models(payload.models, catalog)
    _check_tasks_available(models, session, predictions)
    _check_channel_map(payload.channel_map, models, session)
    _check_channels([model for model in models if model.id not in payload.channel_map], session)

    if runner is None:
        raise APIError(status.HTTP_503_SERVICE_UNAVAILABLE, "COMPUTE_UNAVAILABLE", "Inference execution is not available.")

    try:
        job = jobs.create_if_session_idle(session.id, "inference", {"models": payload.models})
    except ActiveJobExistsError as exc:
        raise job_already_running(exc) from exc

    background_tasks.add_task(run_session_job, job.id, session.id, partial(runner.run, session, models, channel_map=payload.channel_map), executor, cleanup)
    return JobAcceptedResponse(job_id=job.id, state=job.state)


def job_already_running(exc: ActiveJobExistsError) -> APIError:
    return conflict("JOB_ALREADY_RUNNING", "Session already has an active job.", {"job_id": exc.job.id, "state": exc.job.state.value})


def _resolve_models(model_ids: list[str], catalog: ModelCatalog) -> list[ModelInfo]:
    if not model_ids or any(not model_id.strip() for model_id in model_ids):
        raise bad_request("At least one valid model ID is required.")
    if len(model_ids) != len(set(model_ids)):
        raise bad_request("Model IDs must be unique.")

    models = []
    for model_id in model_ids:
        model = catalog.get(model_id)
        if model is None:
            raise not_found("MODEL_NOT_FOUND", "Model not found.", {"model_id": model_id})
        models.append(model)
    return models


def _check_tasks_available(models: list[ModelInfo], session: Session, predictions: PredictionStore) -> None:
    by_task: dict[str, list[str]] = {}
    for model in models:
        by_task.setdefault(model.task, []).append(model.id)

    for task, model_ids in by_task.items():
        if len(model_ids) > 1:
            raise unprocessable("MODEL_INCOMPATIBLE", "Multiple models cannot produce the same task in one session.", {"task": task, "model_ids": model_ids})

    for task in by_task:
        if predictions.get_by_task(session.id, task):
            raise unprocessable("MODEL_INCOMPATIBLE", "Predictions already exist for this task.", {"task": task})


def _check_channel_map(channel_map: dict[str, dict[str, str]], models: list[ModelInfo], session: Session) -> None:
    by_id = {model.id: model for model in models}
    signals = {signal.id for signal in session.signals or []}
    for model_id, mapping in channel_map.items():
        model = by_id.get(model_id)
        if model is None:
            raise bad_request("Channel map names a model that is not selected.", {"model_id": model_id})
        if set(mapping) != set(model.input_channels):
            raise bad_request("Channel map must assign every input channel of the model.", {"model_id": model_id, "channels": list(model.input_channels)})
        unknown = sorted(set(mapping.values()) - signals)
        if unknown:
            raise bad_request("Channel map names unknown signals.", {"model_id": model_id, "signals": unknown})
        if len(set(mapping.values())) != len(mapping):
            # One EDF signal carries one name, so it cannot feed two inputs of one model.
            raise bad_request("Channel map uses a signal for more than one input of the model.", {"model_id": model_id})


def _check_channels(models: list[ModelInfo], session: Session) -> None:
    available = {name.strip().casefold() for signal in session.signals or [] for name in (signal.id, signal.label)}
    for model in models:
        missing = [
            channel for channel in model.input_channels
            if not any(alias.strip().casefold() in available for alias in model.channel_aliases.get(channel, (channel,)))
        ]
        if missing:
            raise unprocessable("MISSING_REQUIRED_CHANNEL", "Required signal could not be resolved.", {"model_id": model.id, "channels": missing})


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
        signals=[SignalResponse(**vars(signal)) for signal in session.signals] if session.signals is not None else None,
        patient=session.patient,
        models=session.model_provenance,
    )
