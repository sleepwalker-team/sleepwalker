"""Run model packages in an isolated worker process and collect their predictions."""

import json
import logging
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Semaphore
from typing import Protocol

import numpy as np
import pandas as pd

from ..core.jobs import CancellationCheck, JobExecutionError, JobPhase, ProgressCallback
from ..core.sessions import Session
from ..recording.study import StudyFormat
from ..results.predictions import PredictionKind, PredictionResult, PredictionStore, map_predictions
from .catalog import ModelInfo

logger = logging.getLogger(__name__)
WORKER_PATH = Path(__file__).with_name("worker.py")
WORKER_ERRORS = {
    "MODEL_INCOMPATIBLE": "Recording or model package is incompatible.",
    "INFERENCE_FAILED": "Model inference failed.",
    "RESOURCE_EXHAUSTED": "Inference could not be completed because compute memory was exhausted.",
}


class InferenceRunner(Protocol):
    def run(self, session: Session, models: list[ModelInfo], report_progress: ProgressCallback, check_cancelled: CancellationCheck) -> None:
        ...


def normalize_predictions(frame: pd.DataFrame, model: ModelInfo, recording_start: datetime, duration: float | None) -> pd.DataFrame:
    """Adapt official interval-start timestamps once to the API's center convention."""
    if frame.empty or model.output_kind != PredictionKind.EPOCH:
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


def stop_worker(process: subprocess.Popen) -> None:
    """Let the worker clean up its framework cache, then kill whatever is left."""
    try:
        # Only the worker: its cache manager child must stay alive for the cleanup.
        process.terminate()
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        # Also stops the framework's multiprocessing children, even after the worker exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class SubprocessInferenceClient:
    def __init__(self, python_executable: Path) -> None:
        # Do not resolve the executable's symlink: that would bypass its venv.
        self.python_executable = python_executable.expanduser().absolute()

    def predict(self, model: ModelInfo, edf_path: Path, *, device: str, batch_size: int, check_cancelled: CancellationCheck) -> pd.DataFrame:
        check_cancelled()
        with TemporaryDirectory(prefix="sleepwalker-inference-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            request_path.write_text(json.dumps({
                "version": 1, "package_path": str(model.package_path.absolute()), "edf_path": str(edf_path.absolute()),
                "device": device, "batch_size": batch_size, "sha256": model.payload_sha256,
            }), encoding="utf-8")
            with (root / "worker.log").open("w+b") as log:
                try:
                    process = subprocess.Popen(
                        [str(self.python_executable), str(WORKER_PATH), str(request_path)], cwd=root, env=_worker_environment(),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except OSError as exc:
                    raise JobExecutionError("COMPUTE_UNAVAILABLE", "Inference Python executable could not be started.") from exc
                try:
                    while process.poll() is None:
                        check_cancelled()
                        try:
                            process.wait(timeout=0.1)
                        except subprocess.TimeoutExpired:
                            pass
                    check_cancelled()
                finally:
                    stop_worker(process)
                return _read_worker_result(root, model, process.returncode, log)


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # The older API checkout must not shadow the worker's framework. Cluster or
    # container overrides could otherwise load a system libtorch instead of the
    # native libraries bundled with the worker venv's Torch.
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _read_worker_result(root: Path, model: ModelInfo, returncode: int, log) -> pd.DataFrame:
    def log_tail(reason: str) -> None:
        log.seek(max(0, log.seek(0, os.SEEK_END) - 8192))
        logger.error("%s for %s:\n%s", reason, model.id, log.read().decode("utf-8", errors="replace"))

    try:
        response = json.loads((root / "response.json").read_text(encoding="utf-8"))
        if response["version"] != 1 or not isinstance(response["ok"], bool):
            raise ValueError("Invalid worker protocol.")
        if not response["ok"]:
            code = response["code"]
            if code not in WORKER_ERRORS:
                raise ValueError("Unknown worker error code.")
            log_tail("Inference worker failed")
            raise JobExecutionError(code, WORKER_ERRORS[code], {"model_id": model.id})
        if returncode != 0:
            raise ValueError("Worker exited unsuccessfully.")
        return pd.read_csv(root / "predictions.csv")
    except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        log_tail("Invalid inference worker response")
        raise JobExecutionError("COMPUTE_UNAVAILABLE", "Inference worker did not return a valid response.", {"model_id": model.id}) from exc


class PackageInferenceRunner:
    """Runs the selected models one after another and publishes all results or none."""

    def __init__(
        self, prediction_store: PredictionStore, *, inference_python: Path | None = None, device: str = "cpu", batch_size: int = 1,
        prediction_client: SubprocessInferenceClient | None = None, max_parallel_jobs: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Inference batch size must be positive.")
        if prediction_client is None and inference_python is None:
            raise ValueError("A separate inference Python executable is required.")
        self.prediction_store = prediction_store
        self.device = device
        self.batch_size = batch_size
        self.prediction_client = prediction_client if prediction_client is not None else SubprocessInferenceClient(inference_python)
        self._slots = Semaphore(max_parallel_jobs)

    def run(self, session: Session, models: list[ModelInfo], report_progress: ProgressCallback, check_cancelled: CancellationCheck) -> None:
        if session.format != StudyFormat.EDF or len(session.source_files) != 1 or session.recording_start is None:
            raise JobExecutionError("MODEL_INCOMPATIBLE", "Inference requires one EDF and recording start metadata.")
        if not models:
            raise JobExecutionError("MODEL_INCOMPATIBLE", "At least one model is required.")

        report_progress(JobPhase.LOADING, 0.0, "Waiting for a free inference slot")
        while not self._slots.acquire(timeout=0.1):
            check_cancelled()
        try:
            results = [self._predict(session, model, index / len(models), report_progress, check_cancelled) for index, model in enumerate(models)]
        finally:
            self._slots.release()

        check_cancelled()
        self.prediction_store.add_many(results)
        session.model_provenance = [*session.model_provenance, *(model.provenance() for model in models)]

    def _predict(self, session: Session, model: ModelInfo, progress: float, report_progress: ProgressCallback, check_cancelled: CancellationCheck) -> PredictionResult:
        report_progress(JobPhase.INFERENCE, progress, f"Running model {model.id}")
        frame = self.prediction_client.predict(model, session.source_files[0], device=self.device, batch_size=self.batch_size, check_cancelled=check_cancelled)
        check_cancelled()
        try:
            result = PredictionResult(
                session_id=session.id, model_id=model.id, task=model.task, classes=model.classes,
                resolution=model.output_resolution, kind=model.output_kind,
                frame=normalize_predictions(frame, model, session.recording_start, session.duration),
            )
            map_predictions(result, session.recording_start)
        except (KeyError, TypeError, ValueError) as exc:
            raise JobExecutionError("MODEL_INCOMPATIBLE", "Model returned predictions in an incompatible format.", {"model_id": model.id, "output_kind": model.output_kind.value}) from exc
        return result
