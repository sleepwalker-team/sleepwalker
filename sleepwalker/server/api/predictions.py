import math

from fastapi import APIRouter, Depends, Request, status

from ..errors import APIError
from ..schemas.prediction import (
    ContinuousPredictionResponse,
    EpochPredictionResponse,
    EventPredictionResponse,
    PredictionResponse,
    TaskListResponse,
    TaskResponse,
)
from ..services.prediction_mapper import (
    map_continuous_predictions,
    map_epoch_predictions,
    map_event_predictions,
)
from ..services.session_service import SessionService
from ..stores.prediction_store import PredictionKind, PredictionResult, PredictionStore
from ..stores.session_store import SessionState


router = APIRouter(prefix="/api/v1/sessions", tags=["predictions"])


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_prediction_store(request: Request) -> PredictionStore:
    return request.app.state.prediction_store


def to_task_response(result: PredictionResult) -> TaskResponse:
    return TaskResponse(
        id=result.task,
        name=result.task.replace("_", " ").capitalize(),
        kind=result.kind,
        classes=list(result.classes),
        resolution=result.resolution,
    )


@router.get("/{session_id}/tasks", response_model=TaskListResponse)
def list_tasks(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    prediction_store: PredictionStore = Depends(get_prediction_store),
) -> TaskListResponse:
    if session_service.get(session_id) is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    return TaskListResponse(
        tasks=[
            to_task_response(result)
            for result in prediction_store.list_for_session(session_id)
        ]
    )


@router.get("/{session_id}/predictions/{task_id}", response_model=PredictionResponse)
def get_predictions(
    session_id: str,
    task_id: str,
    start: float = 0.0,
    end: float | None = None,
    session_service: SessionService = Depends(get_session_service),
    prediction_store: PredictionStore = Depends(get_prediction_store),
) -> PredictionResponse:
    session = session_service.get(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    if (
        session.state != SessionState.READY
        or session.recording_start is None
        or session.duration is None
    ):
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="INVALID_SESSION_STATE",
            message="Session is not ready for prediction access.",
            details={"state": session.state.value},
        )

    result = prediction_store.get_by_task(session_id, task_id)

    if result is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="TASK_NOT_FOUND",
            message="Task not found.",
            details={"task_id": task_id},
        )

    range_end = session.duration if end is None else end

    if (
        not math.isfinite(start)
        or not math.isfinite(range_end)
        or start < 0.0
        or range_end <= start
        or range_end > session.duration
    ):
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_TIME_RANGE",
            message="Prediction time range is invalid.",
            details={"start": start, "end": range_end},
        )

    if result.kind == PredictionKind.EPOCH:
        predictions = [
            EpochPredictionResponse(**vars(prediction))
            for prediction in map_epoch_predictions(result, session.recording_start)
            if prediction.end > start and prediction.start < range_end
        ]
    elif result.kind == PredictionKind.CONTINUOUS:
        predictions = [
            ContinuousPredictionResponse(**vars(prediction))
            for prediction in map_continuous_predictions(result, session.recording_start)
            if start <= prediction.time < range_end
        ]
    else:
        predictions = [
            EventPredictionResponse(**vars(prediction))
            for prediction in map_event_predictions(result, session.recording_start)
            if prediction.end > start and prediction.start < range_end
        ]

    return PredictionResponse(
        task=result.task,
        kind=result.kind,
        classes=list(result.classes),
        predictions=predictions,
    )
