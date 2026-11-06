import math

from fastapi import APIRouter, status

from ..core.sessions import SessionState
from ..results.annotations import AnnotationEdit, AnnotationEditNotFoundError, AnnotationOperation
from ..results.predictions import PredictionKind, PredictionResult, epoch_starts, resolve_epoch_start
from .deps import Annotations, Predictions, SessionDep, require_ready
from .errors import bad_request, conflict, not_found
from .schemas import AnnotationEditResponse, AnnotationListResponse, AnnotationRequest

router = APIRouter(prefix="/api/v1/sessions", tags=["annotations"])


def to_annotation_response(edit: AnnotationEdit) -> AnnotationEditResponse:
    return AnnotationEditResponse(
        edit_id=edit.id,
        operation=edit.operation,
        created_at=edit.created_at,
        task=edit.task,
        start=edit.start,
        end=edit.end,
        label=edit.label,
        event_id=edit.event_id,
        reverted_edit_id=edit.reverted_edit_id,
    )


@router.get("/{session_id}/annotations", response_model=AnnotationListResponse)
def get_annotations(session: SessionDep, annotations: Annotations) -> AnnotationListResponse:
    require_ready(session, "annotation access")
    snapshot = annotations.snapshot(session.id)
    return AnnotationListResponse(
        edits=[to_annotation_response(edit) for edit in snapshot.edits],
        labels=[to_annotation_response(edit) for edit in snapshot.labels],
        events=[to_annotation_response(edit) for edit in snapshot.events],
        deleted_events=[to_annotation_response(edit) for edit in snapshot.deleted_events],
    )


@router.post("/{session_id}/annotations", response_model=AnnotationEditResponse, status_code=status.HTTP_201_CREATED)
def create_annotation(session: SessionDep, payload: AnnotationRequest, predictions: Predictions, annotations: Annotations) -> AnnotationEditResponse:
    if session.state != SessionState.READY or session.duration is None or session.recording_start is None:
        raise conflict("INVALID_SESSION_STATE", "Session is not ready for annotations.", {"state": session.state.value})

    if payload.operation == AnnotationOperation.REVERT:
        edit_id = _required(payload.edit_id, "edit_id")
        try:
            return to_annotation_response(annotations.revert(session.id, edit_id))
        except AnnotationEditNotFoundError as exc:
            raise not_found("EDIT_NOT_FOUND", "Annotation edit not found.", {"edit_id": edit_id}) from exc

    task_id = _required(payload.task, "task")
    task = predictions.get_by_task(session.id, task_id)
    if task is None:
        raise not_found("TASK_NOT_FOUND", "Task not found.", {"task_id": task_id})

    allowed_kinds = {PredictionKind.EPOCH} if payload.operation == AnnotationOperation.SET_LABEL else {PredictionKind.CONTINUOUS, PredictionKind.EVENT}
    if task.kind not in allowed_kinds:
        raise bad_request("Annotation operation is incompatible with the task kind.", {"operation": payload.operation.value, "task_kind": task.kind.value})

    if payload.operation == AnnotationOperation.SET_LABEL:
        start = _required_number(payload.start, "start")
        label = _valid_label(payload.label, task)
        try:
            start = resolve_epoch_start(start, session.duration, task.resolution, epoch_starts(task, session.recording_start))
        except ValueError as exc:
            raise bad_request(str(exc)) from exc
        edit = annotations.set_label(session.id, task.task, start, label)
    elif payload.operation == AnnotationOperation.ADD_EVENT:
        start = _required_number(payload.start, "start")
        end = _required_number(payload.end, "end")
        label = _valid_label(payload.label, task)
        if start < 0.0 or end <= start or end > session.duration:
            raise bad_request("Annotation event range is invalid.")
        edit = annotations.add_event(session.id, task.task, start, end, label)
    else:
        event_id = _required(payload.event_id, "event_id")
        snapshot = annotations.snapshot(session.id)
        human = {event.event_id for event in snapshot.events if event.task == task.task}
        model = {event.id for event in predictions.mapped(task, session.recording_start)[1]} if task.kind == PredictionKind.EVENT else set()
        model -= {edit.event_id for edit in snapshot.deleted_events}
        if event_id not in human | model:
            raise not_found("EVENT_NOT_FOUND", "Annotation event not found.", {"event_id": event_id})
        edit = annotations.delete_event(session.id, task.task, event_id)

    return to_annotation_response(edit)


def _required(value: str | None, field: str) -> str:
    if value is None or not value.strip():
        raise bad_request(f"Field '{field}' is required for this operation.")
    return value


def _required_number(value: float | None, field: str) -> float:
    if value is None or not math.isfinite(value):
        raise bad_request(f"Field '{field}' must be a finite number.")
    return value


def _valid_label(label: str | None, task: PredictionResult) -> str:
    value = _required(label, "label")
    if value not in task.classes:
        raise bad_request("Label is not valid for this task.", {"label": value, "classes": list(task.classes)})
    return value
