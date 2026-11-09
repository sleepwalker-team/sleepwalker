import math
from bisect import bisect_left, bisect_right
from operator import attrgetter

from fastapi import APIRouter

from ..core.sessions import Session, SessionState
from ..recording.signals import MAX_POINTS_LIMIT, MIN_POINTS
from ..results.annotations import effective_annotations
from ..results.predictions import IntervalPrediction, PointPrediction, PredictionKind, PredictionResult, downsample
from .deps import Annotations, Predictions, SessionDep
from .errors import bad_request, conflict, not_found
from .schemas import (
    ContextResponse, ContinuousPredictionResponse, EffectiveAnnotationResponse, IntervalPredictionResponse, ModelEventResponse,
    PredictionResponse, TaskContextResponse, TaskListResponse, TaskResponse,
)

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
def get_predictions(
    session: SessionDep, task_id: str, predictions: Predictions, annotations: Annotations,
    start: float = 0.0, end: float | None = None, max_points: int = 2000, effective: bool = False,
) -> PredictionResponse:
    _require_prediction_access(session)
    if not MIN_POINTS <= max_points <= MAX_POINTS_LIMIT:
        raise bad_request(f"max_points must be between {MIN_POINTS} and {MAX_POINTS_LIMIT}.", code="INVALID_MAX_POINTS")
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
        visible = mapped[bisect_left(mapped, start, key=time):bisect_left(mapped, range_end, key=time)]
        record_type = ContinuousPredictionResponse
    else:
        visible = mapped[bisect_right(mapped, start, key=attrgetter("end")):bisect_left(mapped, range_end, key=attrgetter("start"))]
        record_type = IntervalPredictionResponse
    reduced = downsample(visible, max_points)
    records = [record_type(**vars(p)) for p in reduced]

    events = None
    if result.kind == PredictionKind.EVENT:
        events = [ModelEventResponse(**vars(event)) for event in model_events if event.end > start and event.start < range_end]
    applied = None
    if effective:
        # Only the visible epochs; events are already few.
        applied = [
            EffectiveAnnotationResponse(**vars(item))
            for item in effective_annotations(result, visible, model_events, annotations.snapshot(session.id))
            if item.end > start and item.start < range_end
        ]
    return PredictionResponse(
        task=result.task, kind=result.kind, classes=list(result.classes), predictions=records,
        downsampled=len(reduced) < len(visible), events=events, effective=applied,
    )


@router.get("/{session_id}/context", response_model=ContextResponse)
def get_context(session: SessionDep, predictions: Predictions, time: float) -> ContextResponse:
    """Every task's prediction at one time, for the contextual panel."""
    _require_prediction_access(session)
    if not math.isfinite(time) or not 0.0 <= time <= session.duration:
        raise bad_request("Time is outside the recording.", {"time": time}, code="INVALID_TIME_RANGE")

    tasks = {}
    for result in predictions.list_for_session(session.id):
        mapped, _ = predictions.mapped(result, session.recording_start)
        if result.kind == PredictionKind.CONTINUOUS:
            index = bisect_left(mapped, time, key=attrgetter("time"))
            record = min(mapped[max(index - 1, 0):index + 1], key=lambda point: abs(point.time - time), default=None)
        else:
            index = bisect_right(mapped, time, key=attrgetter("start")) - 1
            record = mapped[index] if index >= 0 and time < mapped[index].end else None
        tasks[result.task] = _context(record) if record is not None else None
    return ContextResponse(time=time, tasks=tasks)


def _context(record: IntervalPrediction | PointPrediction) -> TaskContextResponse:
    label = max(record.probabilities, key=record.probabilities.__getitem__)
    return TaskContextResponse(prediction=label, confidence=record.probabilities[label], probabilities=record.probabilities)


def _require_prediction_access(session: Session) -> None:
    if session.state != SessionState.READY or session.recording_start is None or session.duration is None:
        raise conflict("INVALID_SESSION_STATE", "Session is not ready for prediction access.", {"state": session.state.value})
