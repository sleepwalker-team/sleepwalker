from typing import Protocol

from ..stores.session_store import Session
from .job_executor import CancellationCheck, ProgressCallback
from .model_registry import ModelInfo


class InferenceRunner(Protocol):
    def run(
        self,
        session: Session,
        models: list[ModelInfo],
        report_progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> None:
        ...
