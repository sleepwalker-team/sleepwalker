from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from sleepwalker.server.services.job_executor import JobCancelledError, JobExecutionError
from sleepwalker.server.services.model_registry import ModelInfo
from sleepwalker.server.services.package_inference_runner import PackageInferenceRunner
from sleepwalker.server.stores.job_store import JobPhase
from sleepwalker.server.stores.prediction_store import PredictionKind, PredictionStore
from sleepwalker.server.stores.session_store import Session, SessionState
from sleepwalker.server.validation.study import StudyFormat


class FakePackage:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[Path, int]] = []

    def predict_patient(
        self,
        edf_path: Path,
        batch_size: int,
        num_workers_dataset: int = 1,
        num_workers_loader: int = 0,
    ) -> pd.DataFrame:
        self.calls.append((edf_path, batch_size))
        return self.frame


def create_session(source_path: Path) -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id="sess-test",
        created_at=now,
        expires_at=now,
        last_accessed_at=now,
        source_files=[source_path],
        state=SessionState.READY,
        format=StudyFormat.EDF,
        recording_start=datetime(2025, 1, 1),
    )


def create_model(
    package_path: Path,
    output_kind: PredictionKind = PredictionKind.EPOCH,
) -> ModelInfo:
    return ModelInfo(
        id="sleep-model",
        name="Sleep model",
        task="sleep_stage",
        classes=("wake", "n1"),
        input_channels=("eeg",),
        capabilities=("classification",),
        output_resolution=30.0,
        package_path=package_path,
        output_kind=output_kind,
    )


def test_package_runner_stores_predictions_after_success(tmp_path) -> None:
    source_path = tmp_path / "recording.edf"
    package_path = tmp_path / "model.swmodel"
    package = FakePackage(
        pd.DataFrame(
            {
                "time": [15.0],
                "prediction": ["wake"],
                "prob__wake": [0.9],
                "prob__n1": [0.1],
            }
        )
    )
    loaded: list[tuple[Path, str]] = []

    def load_fake(path: Path, device: str) -> FakePackage:
        loaded.append((path, device))
        return package

    store = PredictionStore()
    runner = PackageInferenceRunner(store, batch_size=16, package_loader=load_fake)
    progress: list[tuple[JobPhase, float | None, str]] = []

    runner.run(
        create_session(source_path),
        [create_model(package_path)],
        lambda *args: progress.append(args),
        lambda: None,
    )

    result = store.get("sess-test", "sleep-model")
    assert result is not None
    assert result.task == "sleep_stage"
    assert result.kind == PredictionKind.EPOCH
    assert result.classes == ("wake", "n1")
    assert result.frame["prediction"].tolist() == ["wake"]
    assert loaded == [(package_path, "cpu")]
    assert package.calls == [(source_path, 16)]
    assert [event[0] for event in progress] == [
        JobPhase.LOADING,
        JobPhase.PREPROCESSING,
        JobPhase.INFERENCE,
        JobPhase.POSTPROCESSING,
    ]


def test_package_runner_preserves_model_output_kind(tmp_path) -> None:
    source_path = tmp_path / "recording.edf"
    package = FakePackage(
        pd.DataFrame(
            {
                "time": [1.0],
                "prob__wake": [0.1],
                "prob__n1": [0.9],
            }
        )
    )
    store = PredictionStore()
    runner = PackageInferenceRunner(
        store,
        package_loader=lambda _path, _device: package,
    )

    runner.run(
        create_session(source_path),
        [create_model(tmp_path / "model.swmodel", PredictionKind.CONTINUOUS)],
        lambda *_args: None,
        lambda: None,
    )

    result = store.get("sess-test", "sleep-model")
    assert result is not None
    assert result.kind == PredictionKind.CONTINUOUS


@pytest.mark.parametrize(
    ("output_kind", "frame"),
    [
        (
            PredictionKind.CONTINUOUS,
            pd.DataFrame({"time": [1.0], "prob__wake": [0.9]}),
        ),
        (
            PredictionKind.EVENT,
            pd.DataFrame(
                {
                    "time": [1.0],
                    "prob__wake": [0.9],
                    "prob__n1": [0.1],
                }
            ),
        ),
    ],
    ids=["continuous", "event"],
)
def test_package_runner_rejects_output_that_does_not_match_declared_kind(
    tmp_path,
    output_kind: PredictionKind,
    frame: pd.DataFrame,
) -> None:
    source_path = tmp_path / "recording.edf"
    store = PredictionStore()
    runner = PackageInferenceRunner(
        store,
        package_loader=lambda _path, _device: FakePackage(frame),
    )

    with pytest.raises(JobExecutionError) as error:
        runner.run(
            create_session(source_path),
            [create_model(tmp_path / "model.swmodel", output_kind)],
            lambda *_args: None,
            lambda: None,
        )

    assert error.value.code == "MODEL_INCOMPATIBLE"
    assert error.value.details == {
        "model_id": "sleep-model",
        "output_kind": output_kind.value,
    }
    assert store.list_for_session("sess-test") == []


def test_package_runner_rejects_non_edf_session(tmp_path) -> None:
    session = create_session(tmp_path / "recording.edf")
    session.format = StudyFormat.NOX
    runner = PackageInferenceRunner(
        PredictionStore(),
        package_loader=lambda _path, _device: FakePackage(pd.DataFrame()),
    )

    with pytest.raises(JobExecutionError) as error:
        runner.run(
            session,
            [create_model(tmp_path / "model.swmodel")],
            lambda *_args: None,
            lambda: None,
        )

    assert error.value.code == "MODEL_INCOMPATIBLE"


def test_package_runner_discards_results_when_cancelled_after_inference(
    tmp_path,
) -> None:
    source_path = tmp_path / "recording.edf"
    package_path = tmp_path / "model.swmodel"
    package = FakePackage(pd.DataFrame({"prediction": ["wake"]}))
    store = PredictionStore()
    runner = PackageInferenceRunner(
        store,
        package_loader=lambda _path, _device: package,
    )
    checks = 0

    def check_cancelled() -> None:
        nonlocal checks
        checks += 1

        if checks == 4:
            raise JobCancelledError

    with pytest.raises(JobCancelledError):
        runner.run(
            create_session(source_path),
            [create_model(package_path)],
            lambda *_args: None,
            check_cancelled,
        )

    assert store.list_for_session("sess-test") == []
