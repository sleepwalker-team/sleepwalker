from pydantic import BaseModel

from ..stores.prediction_store import PredictionKind

# Backwards-compatible name used by earlier server code.
TaskKind = PredictionKind


class TaskResponse(BaseModel):
    id: str
    name: str
    kind: PredictionKind
    classes: list[str]
    resolution: float


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]


class EpochPredictionResponse(BaseModel):
    start: float
    end: float
    label: str
    probabilities: dict[str, float]


class ContinuousPredictionResponse(BaseModel):
    time: float
    probabilities: dict[str, float]


class EventPredictionResponse(BaseModel):
    start: float
    end: float
    label: str
    probabilities: dict[str, float]


class PredictionResponse(BaseModel):
    task: str
    kind: PredictionKind
    classes: list[str]
    predictions: list[
        EpochPredictionResponse
        | ContinuousPredictionResponse
        | EventPredictionResponse
    ]
