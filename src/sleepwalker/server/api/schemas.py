"""Pydantic request and response models of the HTTP API."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..core.jobs import JobPhase, JobState
from ..core.sessions import SessionState
from ..recording.study import StudyFormat
from ..results.annotations import AnnotationOperation
from ..results.exports import ExportState
from ..results.predictions import PredictionKind


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class TaskResponse(BaseModel):
    id: str
    name: str
    kind: PredictionKind
    classes: list[str]
    background_class: str | None
    resolution: float


class SignalResponse(BaseModel):
    id: str
    label: str
    unit: str
    sample_rate: float


class SessionResponse(BaseModel):
    id: str
    state: SessionState
    tasks: list[TaskResponse]
    created_at: datetime
    last_accessed_at: datetime
    expires_at: datetime
    format: StudyFormat | None = None
    duration: float | None = None
    recording_start: datetime | None = None
    signals: list[SignalResponse] | None = None
    # From the EDF+ header, if stated: "sex" as written and "age" in years.
    patient: dict[str, int | str] = {}
    # Exactly which models produced the predictions; "version" is the SHA-256 of the weights.
    models: list[dict[str, Any]] = []


class SignalSummaryResponse(BaseModel):
    id: str
    label: str
    unit: str
    sample_rate: float
    start: float
    end: float


class SignalListResponse(BaseModel):
    signals: list[SignalSummaryResponse]


class SignalDataResponse(BaseModel):
    signal: str
    unit: str
    sample_rate: float
    start: float
    end: float
    downsampled: bool
    points: list[tuple[float, float]]


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]


class IntervalPredictionResponse(BaseModel):
    """One epoch or event."""

    start: float
    end: float
    label: str
    probabilities: dict[str, float]


class ModelEventResponse(BaseModel):
    id: str
    start: float
    end: float
    label: str
    confidence: float


class ContinuousPredictionResponse(BaseModel):
    time: float
    probabilities: dict[str, float]


class EffectiveAnnotationResponse(BaseModel):
    start: float
    end: float
    label: str
    source: Literal["model", "human"]
    confidence: float | None = None
    id: str | None = None


class PredictionResponse(BaseModel):
    task: str
    kind: PredictionKind
    classes: list[str]
    predictions: list[IntervalPredictionResponse | ContinuousPredictionResponse]
    # True if records were merged to max_points, keeping each class's highest probability.
    downsampled: bool = False
    # Only for event tasks: the intervals merged into events.
    events: list[ModelEventResponse] | None = None
    # Only with ?effective=true: model output with the active corrections applied.
    effective: list[EffectiveAnnotationResponse] | None = None


class TaskContextResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class ContextResponse(BaseModel):
    time: float
    # None if the task has no prediction at this time.
    tasks: dict[str, TaskContextResponse | None]


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
    deleted_events: list[AnnotationEditResponse]


class InferenceRequest(BaseModel):
    models: list[str]
    # Per model, input channel -> signal ID. Covers every input of a listed model;
    # models not listed are matched by the names in their package.
    channel_map: dict[str, dict[str, str]] = {}


class JobAcceptedResponse(BaseModel):
    job_id: str
    state: JobState


class JobResponse(BaseModel):
    id: str
    type: str
    state: JobState
    phase: JobPhase | None = None
    progress: float | None = None
    message: str | None = None
    updated_at: datetime
    error: ErrorDetail | None = None


class ModelResponse(BaseModel):
    id: str
    name: str
    task: str
    classes: list[str]
    input_channels: list[str]
    # Per input channel, the EDF signal names its package accepts.
    channel_aliases: dict[str, list[str]]
    output_kind: PredictionKind
    output_resolution: float


class ModelListResponse(BaseModel):
    models: list[ModelResponse]


class ExportFormat(str, Enum):
    JSON = "json"
    SLEEPWALKER = "sleepwalker"


class ExportMode(str, Enum):
    RESULTS = "results"
    WORKSPACE = "workspace"


class ExportInclude(BaseModel):
    predictions: bool = True
    probabilities: bool = True
    annotations: bool = True
    metadata: bool = True


class ExportRequest(BaseModel):
    format: ExportFormat
    mode: ExportMode
    include: ExportInclude = Field(default_factory=ExportInclude)

    @model_validator(mode="after")
    def validate_format_and_mode(self) -> "ExportRequest":
        supported = {
            (ExportFormat.JSON, ExportMode.RESULTS),
            (ExportFormat.SLEEPWALKER, ExportMode.WORKSPACE),
        }
        if (self.format, self.mode) not in supported:
            raise ValueError("Supported exports are json/results and sleepwalker/workspace.")

        if self.mode == ExportMode.WORKSPACE and not all(self.include.model_dump().values()):
            raise ValueError("Workspace exports must include all data to allow restoration.")

        return self


class ExportAcceptedResponse(BaseModel):
    export_id: str
    job_id: str
    state: JobState


class ExportResponse(BaseModel):
    id: str
    state: ExportState
    format: ExportFormat
    mode: ExportMode
    size: int | None = None
    created_at: datetime
    expires_at: datetime


class ImportAcceptedResponse(BaseModel):
    session_id: str
    job_id: str
    state: SessionState
