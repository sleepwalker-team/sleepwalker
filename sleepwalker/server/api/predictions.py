import math

from fastapi import APIRouter

from ..core.sessions import SessionState
from ..results.predictions import IntervalPrediction, PredictionResult, map_predictions
from .deps import Predictions, SessionDep
from .errors import bad_request, conflict, not_found
from .schemas import ContinuousPredictionResponse, IntervalPredictionResponse, PredictionResponse, TaskListResponse, TaskResponse

router = APIRouter(prefix="/api/v1/sessions", tags=["predictions"])


def to_task_response(result: PredictionResult) -> TaskResponse:
    return TaskResponse(
        id=result.task,
        name=result.task.replace("_", " ").capitalize(),
        kind=result.kind,
        classes=list(result.classes),
        resolution=result.resolution,
    )


@router.get("/{session_id}/tasks", response_model=TaskListResponse)
def list_tasks(session: SessionDep, predictions: Predictions) -> TaskListResponse:
    return TaskListResponse(tasks=[to_task_response(result) for result in predictions.list_for_session(session.id)])


@router.get("/{session_id}/predictions/{task_id}", response_model=PredictionResponse)
def get_predictions(session: SessionDep, task_id: str, predictions: Predictions, start: float = 0.0, end: float | None = None) -> PredictionResponse:
    if session.state != SessionState.READY or session.recording_start is None or session.duration is None:
        raise conflict("INVALID_SESSION_STATE", "Session is not ready for prediction access.", {"state": session.state.value})

    result = predictions.get_by_task(session.id, task_id)
    if result is None:
        raise not_found("TASK_NOT_FOUND", "Task not found.", {"task_id": task_id})

    range_end = session.duration if end is None else end
    if not math.isfinite(start) or not math.isfinite(range_end) or start < 0.0 or range_end <= start or range_end > session.duration:
        raise bad_request("Prediction time range is invalid.", {"start": start, "end": range_end}, code="INVALID_TIME_RANGE")

    records = []
    for prediction in map_predictions(result, session.recording_start):
        if isinstance(prediction, IntervalPrediction):
            if prediction.end > start and prediction.start < range_end:
                records.append(IntervalPredictionResponse(**vars(prediction)))
        elif start <= prediction.time < range_end:
            records.append(ContinuousPredictionResponse(**vars(prediction)))

    return PredictionResponse(task=result.task, kind=result.kind, classes=list(result.classes), predictions=records)
