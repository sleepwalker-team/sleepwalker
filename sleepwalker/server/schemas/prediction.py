from enum import Enum

from pydantic import BaseModel


class TaskKind(str, Enum):
    EPOCH = "epoch"


class TaskResponse(BaseModel):
    id: str
    name: str
    kind: TaskKind
    classes: list[str]
    resolution: float


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]


class EpochPredictionResponse(BaseModel):
    start: float
    end: float
    label: str
    probabilities: dict[str, float]


class PredictionResponse(BaseModel):
    task: str
    kind: TaskKind
    classes: list[str]
    predictions: list[EpochPredictionResponse]
