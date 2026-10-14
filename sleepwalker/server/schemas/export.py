from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from ..stores.export_store import ExportState
from ..stores.job_store import JobState


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
