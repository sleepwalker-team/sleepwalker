"""Explicit test-only UI server with deterministic model results, never production."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import uvicorn

from sleepwalker.server.app import create_app
from sleepwalker.server.services.model_registry import ModelInfo, ModelRegistry
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.stores.job_store import JobPhase
from sleepwalker.server.stores.prediction_store import PredictionKind, PredictionResult, PredictionStore

from ui.app import create_ui_app


class TestRunner:
    def __init__(self, store):
        self.store = store

    def run(self, session, models, report_progress, check_cancelled):
        results = []
        for model in models:
            check_cancelled()
            report_progress(JobPhase.INFERENCE, None, "Deterministic GUI test predictions")
            if model.output_kind == PredictionKind.EVENT:
                starts = np.array([15., 65., 110.])
                frame = pd.DataFrame({"start": starts, "end": starts + 5, "prob__arousal": 1.})
            elif model.output_kind == PredictionKind.CONTINUOUS:
                times = np.arange(0, session.duration, model.output_resolution)
                p = .3 + .2 * np.sin(times / 20)
                frame = pd.DataFrame({"time": times, "prob__desaturation": p, "prob__no desaturation": 1 - p})
            else:
                times = np.arange(model.output_resolution / 2, session.duration, model.output_resolution)
                winner = "n2" if "n2" in model.classes else model.classes[0]
                values = {f"prob__{label}": np.full(len(times), 1 - .04 * (len(model.classes) - 1) if label == winner else .04) for label in model.classes}
                frame = pd.DataFrame({"time": times, **values})
            results.append(PredictionResult(session_id=session.id, model_id=model.id, task=model.task,
                                            classes=model.classes, resolution=model.output_resolution,
                                            frame=frame, kind=model.output_kind))
        check_cancelled()
        self.store.add_many(results)


def test_api(root):
    specs = [
        ("sleep", ("wake", "n1", "n2", "n3", "rem"), 30, PredictionKind.EPOCH),
        ("arousal", ("arousal",), 1, PredictionKind.EVENT),
        ("desaturation", ("desaturation", "no desaturation"), 1, PredictionKind.CONTINUOUS),
    ]
    models = [ModelInfo(id=f"test-{task}", name=f"TEST · {task}", task=task, classes=classes,
                        input_channels=("EEG",), capabilities=("classification",), output_resolution=resolution,
                        package_path=Path("test-only"), output_kind=kind) for task, classes, resolution, kind in specs]
    models.append(ModelInfo(id="test-unmatched", name="TEST · unmatched inputs", task="unmatched",
                            classes=("artifact",), input_channels=("C3-M2", "E1-M2", "Chin EMG", "ECG"),
                            capabilities=("classification",), output_resolution=1, package_path=Path("test-only")))
    store = PredictionStore()
    return create_app(model_registry=ModelRegistry(models), prediction_store=store,
                      inference_runner=TestRunner(store), temporary_storage=TemporaryStorage(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6019)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="sleepwalker-ui-browser-") as directory:
        uvicorn.run(create_ui_app(test_api(Path(directory))), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
