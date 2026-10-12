import math

from fastapi import APIRouter, Depends, Request, status

from ..errors import APIError
from ..schemas.annotation import AnnotationEditResponse, AnnotationRequest
from ..services.annotation_service import (
    AnnotationEditNotFoundError,
    AnnotationEventNotFoundError,
    AnnotationService,
)
from ..services.session_service import SessionService
from ..stores.annotation_store import AnnotationEdit, AnnotationOperation
from ..stores.prediction_store import PredictionKind, PredictionResult, PredictionStore
from ..stores.session_store import SessionState


router = APIRouter(prefix="/api/v1/sessions", tags=["annotations"])


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_prediction_store(request: Request) -> PredictionStore:
    return request.app.state.prediction_store


def get_annotation_service(request: Request) -> AnnotationService:
    return request.app.state.annotation_service


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


@router.post(
    "/{session_id}/annotations",
    response_model=AnnotationEditResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_annotation(
    session_id: str,
    payload: AnnotationRequest,
    session_service: SessionService = Depends(get_session_service),
    prediction_store: PredictionStore = Depends(get_prediction_store),
    annotation_service: AnnotationService = Depends(get_annotation_service),
) -> AnnotationEditResponse:
    session = session_service.get(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    if session.state != SessionState.READY or session.duration is None:
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="INVALID_SESSION_STATE",
            message="Session is not ready for annotations.",
            details={"state": session.state.value},
        )

    if payload.operation == AnnotationOperation.REVERT:
        edit_id = _required(payload.edit_id, "edit_id")

        try:
            return to_annotation_response(
                annotation_service.revert(session.id, edit_id)
            )
        except AnnotationEditNotFoundError as exc:
            raise APIError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="EDIT_NOT_FOUND",
                message="Annotation edit not found.",
                details={"edit_id": edit_id},
            ) from exc

    task_id = _required(payload.task, "task")
    task = prediction_store.get_by_task(session.id, task_id)

    if task is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="TASK_NOT_FOUND",
            message="Task not found.",
            details={"task_id": task_id},
        )

    _validate_operation_kind(payload.operation, task)

    if payload.operation == AnnotationOperation.SET_LABEL:
        start = _required_number(payload.start, "start")
        label = _valid_label(payload.label, task)
        _validate_epoch_start(start, session.duration, task.resolution)
        edit = annotation_service.set_label(session.id, task.task, start, label)
    elif payload.operation == AnnotationOperation.ADD_EVENT:
        start = _required_number(payload.start, "start")
        end = _required_number(payload.end, "end")
        label = _valid_label(payload.label, task)
        _validate_event_range(start, end, session.duration)
        edit = annotation_service.add_event(session.id, task.task, start, end, label)
    elif payload.operation == AnnotationOperation.DELETE_EVENT:
        event_id = _required(payload.event_id, "event_id")

        try:
            event = next(
                event
                for event in annotation_service.current_events(session.id)
                if event.event_id == event_id and event.task == task.task
            )
        except StopIteration as exc:
            raise APIError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="EVENT_NOT_FOUND",
                message="Annotation event not found.",
                details={"event_id": event_id},
            ) from exc

        try:
            edit = annotation_service.delete_event(session.id, event.event_id or "")
        except AnnotationEventNotFoundError as exc:
            raise APIError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="EVENT_NOT_FOUND",
                message="Annotation event not found.",
                details={"event_id": event_id},
            ) from exc
    else:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Unsupported annotation operation.",
        )

    return to_annotation_response(edit)


def _validate_operation_kind(
    operation: AnnotationOperation,
    task: PredictionResult,
) -> None:
    if operation == AnnotationOperation.SET_LABEL:
        allowed_kinds = {PredictionKind.EPOCH}
    else:
        allowed_kinds = {PredictionKind.CONTINUOUS, PredictionKind.EVENT}

    if task.kind not in allowed_kinds:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Annotation operation is incompatible with the task kind.",
            details={
                "operation": operation.value,
                "task_kind": task.kind.value,
            },
        )


def _required(value: str | None, field: str) -> str:
    if value is None or not value.strip():
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message=f"Field '{field}' is required for this operation.",
        )

    return value


def _required_number(value: float | None, field: str) -> float:
    if value is None or not math.isfinite(value):
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message=f"Field '{field}' must be a finite number.",
        )

    return value


def _valid_label(label: str | None, task: PredictionResult) -> str:
    value = _required(label, "label")

    if value not in task.classes:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Label is not valid for this task.",
            details={"label": value, "classes": list(task.classes)},
        )

    return value


def _validate_epoch_start(start: float, duration: float, resolution: float) -> None:
    aligned = math.isclose(
        start / resolution,
        round(start / resolution),
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    if start < 0.0 or start + resolution > duration or not aligned:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Epoch start is outside the recording or not aligned to the task resolution.",
        )


def _validate_event_range(start: float, end: float, duration: float) -> None:
    if start < 0.0 or end <= start or end > duration:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="Annotation event range is invalid.",
        )
