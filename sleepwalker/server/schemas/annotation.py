from datetime import datetime

from pydantic import BaseModel

from ..stores.annotation_store import AnnotationOperation


class AnnotationRequest(BaseModel):
    operation: AnnotationOperation
    task: str | None = None
    start: float | None = None
    end: float | None = None
    label: str | None = None
    event_id: str | None = None
    edit_id: str | None = None


class AnnotationEditResponse(BaseModel):
    edit_id: str
    operation: AnnotationOperation
    created_at: datetime
    task: str | None = None
    start: float | None = None
    end: float | None = None
    label: str | None = None
    event_id: str | None = None
    reverted_edit_id: str | None = None


class AnnotationListResponse(BaseModel):
    edits: list[AnnotationEditResponse]
    labels: list[AnnotationEditResponse]
    events: list[AnnotationEditResponse]
