import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .job_executor import CancellationCheck, JobExecutionError
from .model_registry import ModelInfo

logger = logging.getLogger(__name__)
WORKER_PATH = Path(__file__).parents[1] / "inference_worker.py"


def stop_worker(process: subprocess.Popen) -> None:
    """Also stop the official framework's multiprocessing cache children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A child may have survived even after the worker itself exited.
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
            request = {
                "version": 1, "package_path": str(model.package_path.absolute()), "edf_path": str(edf_path.absolute()),
                "device": device, "batch_size": batch_size, "task": model.task, "classes": list(model.classes),
                "input_channels": list(model.input_channels), "resolution": model.output_resolution, "sha256": model.payload_sha256,
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            # Cluster/container overrides can otherwise load a system libtorch
            # instead of the native libraries bundled with this venv's Torch.
            environment.pop("LD_LIBRARY_PATH", None)
            environment.pop("LD_PRELOAD", None)
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            with (root / "worker.log").open("w+b") as log:
                try:
                    process = subprocess.Popen(
                        [str(self.python_executable), str(WORKER_PATH), str(request_path)], cwd=root, env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except OSError as exc:
                    raise JobExecutionError("COMPUTE_UNAVAILABLE", "Inference Python executable could not be started.") from exc
                try:
                    while True:
                        check_cancelled()
                        try:
                            process.wait(timeout=0.1)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    check_cancelled()
                finally:
                    stop_worker(process)

                try:
                    response = json.loads((root / "response.json").read_text(encoding="utf-8"))
                    if response["version"] != 1 or not isinstance(response["ok"], bool):
                        raise ValueError("Invalid worker protocol.")
                    if not response["ok"]:
                        log.seek(max(0, log.seek(0, os.SEEK_END) - 8192))
                        logger.error("Inference worker failed for %s:\n%s", model.id, log.read().decode("utf-8", errors="replace"))
                        code = response["code"]
                        if code not in {"MODEL_INCOMPATIBLE", "INFERENCE_FAILED"}:
                            raise ValueError("Unknown worker error code.")
                        message = "Recording or model package is incompatible." if code == "MODEL_INCOMPATIBLE" else "Model inference failed."
                        raise JobExecutionError(code, message, {"model_id": model.id})
                    if process.returncode != 0:
                        raise ValueError("Worker exited unsuccessfully.")
                    return pd.read_csv(root / "predictions.csv")
                except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
                    log.seek(max(0, log.seek(0, os.SEEK_END) - 8192))
                    logger.error("Invalid inference worker response for %s:\n%s", model.id, log.read().decode("utf-8", errors="replace"))
                    raise JobExecutionError("COMPUTE_UNAVAILABLE", "Inference worker did not return a valid response.", {"model_id": model.id}) from exc
