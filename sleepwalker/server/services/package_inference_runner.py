from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

from .inference_runner import InferenceRunner
from .job_executor import CancellationCheck, JobExecutionError, ProgressCallback
from .model_registry import ModelInfo
from ..stores.job_store import JobPhase
from ..stores.prediction_store import PredictionResult, PredictionStore
from ..stores.session_store import Session
from ..validation.study import StudyFormat


class LoadedPredictionPackage(Protocol):
    def predict_patient(
        self,
        edf_path: Path,
        batch_size: int,
        num_workers_dataset: int = 1,
        num_workers_loader: int = 0,
    ) -> pd.DataFrame:
        ...


PackageLoader = Callable[[Path, str], LoadedPredictionPackage]


def load_package(package_path: Path, device: str) -> LoadedPredictionPackage:
    from sleepwalker.deployment import load_prediction_package

    return load_prediction_package(str(package_path), map_location=device)


class PackageInferenceRunner(InferenceRunner):
    def __init__(
        self,
        prediction_store: PredictionStore,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        package_loader: PackageLoader = load_package,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Inference batch size must be positive.")

        self.prediction_store = prediction_store
        self.device = device
        self.batch_size = batch_size
        self.package_loader = package_loader

    def run(
        self,
        session: Session,
        models: list[ModelInfo],
        report_progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> None:
        check_cancelled()

        if session.format != StudyFormat.EDF or len(session.source_files) != 1:
            raise JobExecutionError(
                code="MODEL_INCOMPATIBLE",
                message="Session format is incompatible with model inference.",
                details={
                    "format": (
                        session.format.value
                        if session.format is not None
                        else None
                    )
                },
            )

        source_path = session.source_files[0]
        results: list[PredictionResult] = []

        for index, model in enumerate(models):
            check_cancelled()
            base_progress = index / len(models)
            report_progress(
                JobPhase.LOADING,
                base_progress,
                f"Loading model {model.id}",
            )
            package = self.package_loader(model.package_path, self.device)
            check_cancelled()

            report_progress(
                JobPhase.PREPROCESSING,
                base_progress,
                f"Preparing input for model {model.id}",
            )
            report_progress(
                JobPhase.INFERENCE,
                base_progress,
                f"Running model {model.id}",
            )
            frame = package.predict_patient(
                source_path,
                batch_size=self.batch_size,
                num_workers_dataset=1,
                num_workers_loader=0,
            )
            check_cancelled()

            if not isinstance(frame, pd.DataFrame):
                raise JobExecutionError(
                    code="MODEL_INCOMPATIBLE",
                    message="Model returned an unsupported prediction format.",
                    details={"model_id": model.id},
                )

            results.append(
                PredictionResult(
                    session_id=session.id,
                    model_id=model.id,
                    task=model.task,
                    classes=model.classes,
                    resolution=model.output_resolution,
                    frame=frame,
                    kind=model.output_kind,
                )
            )
            report_progress(
                JobPhase.POSTPROCESSING,
                (index + 1) / len(models),
                f"Storing predictions for model {model.id}",
            )

        check_cancelled()
        self.prediction_store.add_many(results)
