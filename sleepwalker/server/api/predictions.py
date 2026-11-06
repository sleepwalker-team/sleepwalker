import math
from bisect import bisect_left, bisect_right
from operator import attrgetter

from fastapi import APIRouter

from ..core.sessions import SessionState
from ..results.predictions import PredictionKind, PredictionResult
from .deps import Predictions, SessionDep
from .errors import bad_request, conflict, not_found
from .schemas import ContinuousPredictionResponse, IntervalPredictionResponse, ModelEventResponse, PredictionResponse, TaskListResponse, TaskResponse

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


@router.get("/{session_id}/predictions/{task_id}", response_model=PredictionResponse, response_model_exclude_none=True)
def get_predictions(session: SessionDep, task_id: str, predictions: Predictions, start: float = 0.0, end: float | None = None) -> PredictionResponse:
    if session.state != SessionState.READY or session.recording_start is None or session.duration is None:
        raise conflict("INVALID_SESSION_STATE", "Session is not ready for prediction access.", {"state": session.state.value})

    result = predictions.get_by_task(session.id, task_id)
    if result is None:
        raise not_found("TASK_NOT_FOUND", "Task not found.", {"task_id": task_id})

    range_end = session.duration if end is None else end
    if not math.isfinite(start) or not math.isfinite(range_end) or start < 0.0 or range_end <= start or range_end > session.duration:
        raise bad_request("Prediction time range is invalid.", {"start": start, "end": range_end}, code="INVALID_TIME_RANGE")

    # Records are sorted by time, so the requested range is a slice.
    mapped, model_events = predictions.mapped(result, session.recording_start)
    if result.kind == PredictionKind.CONTINUOUS:
        time = attrgetter("time")
        records = [ContinuousPredictionResponse(**vars(p)) for p in mapped[bisect_left(mapped, start, key=time):bisect_left(mapped, range_end, key=time)]]
    else:
        visible = mapped[bisect_right(mapped, start, key=attrgetter("end")):bisect_left(mapped, range_end, key=attrgetter("start"))]
        records = [IntervalPredictionResponse(**vars(p)) for p in visible]

    events = None
    if result.kind == PredictionKind.EVENT:
        events = [ModelEventResponse(**vars(event)) for event in model_events if event.end > start and event.start < range_end]
    return PredictionResponse(task=result.task, kind=result.kind, classes=list(result.classes), predictions=records, events=events)
