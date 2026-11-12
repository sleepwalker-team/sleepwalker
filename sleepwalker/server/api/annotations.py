import math

from fastapi import APIRouter, status

from ..core.sessions import SessionState
from ..results.annotations import (
    AnnotationEdit, AnnotationEditNotFoundError, AnnotationEventNotFoundError, AnnotationOperation, InvalidAnnotationError,
)
from ..results.predictions import PredictionKind
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

    operation = payload.operation
    values = {}
    if operation == AnnotationOperation.DELETE_EVENT:
        values["event_id"] = _required(payload.event_id, "event_id")
    else:
        values["start"] = _required_number(payload.start, "start")
        values["label"] = _required(payload.label, "label")
        if operation == AnnotationOperation.ADD_EVENT:
            values["end"] = _required_number(payload.end, "end")
    model_event_ids = {event.id for event in predictions.mapped(task, session.recording_start)[1]} if task.kind == PredictionKind.EVENT else set()
    try:
        edit = annotations.record(
            session.id, operation, task, duration=session.duration, recording_start=session.recording_start, model_event_ids=model_event_ids, **values,
        )
    except InvalidAnnotationError as exc:
        raise bad_request(str(exc), exc.details) from exc
    except AnnotationEventNotFoundError as exc:
        raise not_found("EVENT_NOT_FOUND", "Annotation event not found.", {"event_id": values["event_id"]}) from exc
    return to_annotation_response(edit)


def _required(value: str | None, field: str) -> str:
    if value is None or not value.strip():
        raise bad_request(f"Field '{field}' is required for this operation.")
    return value


def _required_number(value: float | None, field: str) -> float:
    if value is None or not math.isfinite(value):
        raise bad_request(f"Field '{field}' must be a finite number.")
    return value

