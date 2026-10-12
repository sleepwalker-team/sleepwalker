from dataclasses import dataclass, replace
from enum import Enum

import pandas as pd


class PredictionKind(str, Enum):
    EPOCH = "epoch"
    CONTINUOUS = "continuous"
    EVENT = "event"


@dataclass(frozen=True)
class PredictionResult:
    session_id: str
    model_id: str
    task: str
    classes: tuple[str, ...]
    resolution: float
    frame: pd.DataFrame
    kind: PredictionKind = PredictionKind.EPOCH


class PredictionStore:
    def __init__(self) -> None:
        self._results: dict[tuple[str, str], PredictionResult] = {}

    def add(self, result: PredictionResult) -> None:
        self.add_many([result])

    def add_many(self, results: list[PredictionResult]) -> None:
        keys = [(result.session_id, result.model_id) for result in results]
        task_keys = [(result.session_id, result.task) for result in results]

        if len(keys) != len(set(keys)):
            raise ValueError("Prediction batch contains duplicate session and model IDs.")

        if len(task_keys) != len(set(task_keys)):
            raise ValueError("Prediction batch contains duplicate session and task IDs.")

        conflicts = [key for key in keys if key in self._results]

        if conflicts:
            session_id, model_id = conflicts[0]
            raise ValueError(
                "Predictions already exist for session "
                f"{session_id} and model {model_id}."
            )

        existing_task_keys = {
            (result.session_id, result.task)
            for result in self._results.values()
        }
        task_conflicts = [key for key in task_keys if key in existing_task_keys]

        if task_conflicts:
            session_id, task = task_conflicts[0]
            raise ValueError(
                "Predictions already exist for session "
                f"{session_id} and task {task}."
            )

        for result, key in zip(results, keys):
            self._results[key] = replace(result, frame=result.frame.copy(deep=True))

    def get(self, session_id: str, model_id: str) -> PredictionResult | None:
        return self._results.get((session_id, model_id))

    def get_by_task(self, session_id: str, task: str) -> PredictionResult | None:
        return next(
            (
                result
                for (stored_session_id, _), result in self._results.items()
                if stored_session_id == session_id and result.task == task
            ),
            None,
        )

    def list_for_session(self, session_id: str) -> list[PredictionResult]:
        return sorted(
            (
                result
                for (stored_session_id, _), result in self._results.items()
                if stored_session_id == session_id
            ),
            key=lambda result: result.model_id,
        )

    def delete_session(self, session_id: str) -> int:
        keys = [key for key in self._results if key[0] == session_id]

        for key in keys:
            del self._results[key]

        return len(keys)
