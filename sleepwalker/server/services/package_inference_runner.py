from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .inference_runner import InferenceRunner
from .job_executor import CancellationCheck, JobExecutionError, ProgressCallback
from .model_registry import ModelInfo
from .prediction_mapper import validate_prediction_result
from .subprocess_inference import SubprocessInferenceClient
from ..stores.job_store import JobPhase
from ..stores.prediction_store import PredictionKind, PredictionResult, PredictionStore
from ..stores.session_store import Session
from ..validation.study import StudyFormat


def normalize_predictions(frame: pd.DataFrame, model: ModelInfo, recording_start: datetime, duration: float | None) -> pd.DataFrame:
    """Adapt official interval-start timestamps once to the API's center convention."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or model.output_kind != PredictionKind.EPOCH:
        raise ValueError("Expected nonempty interval-classification predictions.")
    if "task" not in frame or not frame["task"].eq(model.task).all():
        raise ValueError("Unexpected prediction task.")
    if "valid" not in frame or not frame["valid"].eq(True).all():
        raise ValueError("Invalid prediction windows.")
    probabilities = frame[[f"prob__{label}" for label in model.classes]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid prediction probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Prediction probabilities do not sum to one.")
    if pd.api.types.is_numeric_dtype(frame["time"]):
        raise ValueError("Expected absolute framework timestamps.")
    times = pd.to_datetime(frame["time"], format="ISO8601", errors="raise")
    if times.dt.tz is not None:
        times = times.dt.tz_localize(None)
    start = pd.Timestamp(recording_start).tz_localize(None)
    seconds = (times - start).dt.total_seconds()
    if seconds.isna().any() or seconds.duplicated().any() or (seconds < 0).any():
        raise ValueError("Invalid or overlapping prediction timestamps.")
    if duration is not None and (seconds + model.output_resolution > duration + 1e-6).any():
        raise ValueError("Prediction interval exceeds the recording.")
    normalized = frame.copy()
    normalized["time"] = seconds + model.output_resolution / 2
    return normalized


class PackageInferenceRunner(InferenceRunner):
    def __init__(self, prediction_store: PredictionStore, *, inference_python: Path | None = None, device: str = "cpu", batch_size: int = 1, prediction_client: SubprocessInferenceClient | None = None) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Inference batch size must be positive.")
        if prediction_client is None and inference_python is None:
            raise ValueError("A separate inference Python executable is required.")
        self.prediction_store = prediction_store
        self.device = device
        self.batch_size = batch_size
        self.prediction_client = prediction_client if prediction_client is not None else SubprocessInferenceClient(inference_python)

    def run(self, session: Session, models: list[ModelInfo], report_progress: ProgressCallback, check_cancelled: CancellationCheck) -> None:
        check_cancelled()
        if session.format != StudyFormat.EDF or len(session.source_files) != 1 or session.recording_start is None:
            raise JobExecutionError("MODEL_INCOMPATIBLE", "Inference requires one EDF and recording start metadata.")
        if not models:
            raise JobExecutionError("MODEL_INCOMPATIBLE", "At least one model is required.")
        results = []
        for index, model in enumerate(models):
            check_cancelled()
            base_progress = index / len(models)
            report_progress(JobPhase.LOADING, base_progress, f"Starting isolated worker for {model.id}")
            report_progress(JobPhase.PREPROCESSING, base_progress, f"Using packaged preprocessing for {model.id}")
            report_progress(JobPhase.INFERENCE, base_progress, f"Running model {model.id}")
            frame = self.prediction_client.predict(model, session.source_files[0], device=self.device, batch_size=self.batch_size, check_cancelled=check_cancelled)
            check_cancelled()
            try:
                frame = normalize_predictions(frame, model, session.recording_start, session.duration)
                result = PredictionResult(
                    session_id=session.id, model_id=model.id, task=model.task, classes=model.classes,
                    resolution=model.output_resolution, frame=frame, kind=model.output_kind,
                )
                validate_prediction_result(result, session.recording_start)
            except (KeyError, TypeError, ValueError) as exc:
                raise JobExecutionError("MODEL_INCOMPATIBLE", "Model returned predictions in an incompatible format.", {"model_id": model.id, "output_kind": model.output_kind.value}) from exc
            results.append(result)
            report_progress(JobPhase.POSTPROCESSING, (index + 1) / len(models), f"Storing predictions for model {model.id}")
        check_cancelled()
        self.prediction_store.add_many(results)
