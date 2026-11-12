from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from sleepwalker.server.core.jobs import JobCancelledError, JobExecutionError
from sleepwalker.server.inference.catalog import ModelInfo
from sleepwalker.server.inference.runner import PackageInferenceRunner
from sleepwalker.server.core.jobs import JobPhase
from sleepwalker.server.results.predictions import map_predictions, PredictionKind, PredictionStore
from sleepwalker.server.core.sessions import Session, SessionState
from sleepwalker.server.recording.study import StudyFormat


def create_session(path):
    now = datetime.now(timezone.utc)
    return Session(id="sess-test", created_at=now, last_accessed_at=now, expires_at=now, source_files=[path], state=SessionState.READY, format=StudyFormat.EDF, recording_start=datetime(2025, 1, 1), duration=690)


def create_model(path):
    return ModelInfo(id="sleep-model", name="Sleep model", task="sleep", classes=("wake", "n1"), input_channels=("eeg",), output_resolution=30, package_path=path)


def prediction_frame():
    return pd.DataFrame({"time": ["2025-01-01 00:05:00", "2025-01-01 00:05:30"], "task": ["sleep", "sleep"], "valid": [True, True], "prob__wake": [0.9, 0.1], "prob__n1": [0.1, 0.9]})


class FakeClient:
    def __init__(self, frame=None):
        self.frame = prediction_frame() if frame is None else frame
        self.calls = []

    def predict(self, model, edf_path, *, device, batch_size, check_cancelled):
        check_cancelled()
        self.calls.append((model.package_path, edf_path, device, batch_size))
        return self.frame


def test_runner_stores_predictions_with_correct_interval_boundaries(tmp_path):
    model = create_model(tmp_path / "model")
    session = create_session(tmp_path / "recording.edf")
    client = FakeClient()
    store = PredictionStore()
    runner = PackageInferenceRunner(store, batch_size=16, prediction_client=client)
    progress = []
    runner.run(session, [model], lambda *args: progress.append(args), lambda: None)
    result = store.get_by_task(session.id, model.task)
    assert result.kind == PredictionKind.EPOCH
    assert result.frame["time"].tolist() == [315, 345]
    assert client.frame["time"].tolist() == ["2025-01-01 00:05:00", "2025-01-01 00:05:30"]
    predictions = map_predictions(result, session.recording_start)
    assert [(p.start, p.end, p.label) for p in predictions] == [(300, 330, "wake"), (330, 360, "n1")]
    assert client.calls == [(model.package_path, session.source_files[0], "cpu", 16)]
    assert session.model_provenance == [model.provenance()]
    assert [phase for phase, _, _ in progress] == [JobPhase.LOADING, JobPhase.INFERENCE]


@pytest.mark.parametrize("bad_case", ["empty", "missing_class", "wrong_task", "invalid_window", "nan_probability", "negative_probability", "probability_sum", "numeric_time", "bad_time", "duplicate_time", "before_recording", "past_recording"])
def test_runner_rejects_incompatible_output_without_publishing(tmp_path, bad_case):
    frame = prediction_frame()
    if bad_case == "empty":
        frame = pd.DataFrame()
    elif bad_case == "missing_class":
        frame = frame.drop(columns="prob__n1")
    elif bad_case == "wrong_task":
        frame.loc[0, "task"] = "arousal"
    elif bad_case == "invalid_window":
        frame.loc[0, "valid"] = False
    elif bad_case == "nan_probability":
        frame.loc[0, "prob__wake"] = float("nan")
    elif bad_case == "negative_probability":
        frame.loc[0, "prob__wake"] = -0.1
    elif bad_case == "probability_sum":
        frame.loc[0, "prob__wake"] = 0.1
    elif bad_case == "numeric_time":
        frame["time"] = [300, 330]
    elif bad_case == "bad_time":
        frame.loc[0, "time"] = "not a time"
    elif bad_case == "duplicate_time":
        frame.loc[1, "time"] = frame.loc[0, "time"]
    elif bad_case == "before_recording":
        frame.loc[0, "time"] = "2024-12-31 23:59:59"
    else:
        frame.loc[0, "time"] = "2025-01-01 00:11:29"
    store = PredictionStore()
    runner = PackageInferenceRunner(store, prediction_client=FakeClient(frame))
    with pytest.raises(JobExecutionError) as error:
        runner.run(create_session(tmp_path / "recording.edf"), [create_model(tmp_path / "model")], lambda *_: None, lambda: None)
    assert error.value.code == "MODEL_INCOMPATIBLE"
    assert store.list_for_session("sess-test") == []


def test_runner_does_not_publish_partial_batch_when_second_model_fails(tmp_path):
    class FailingSecondClient(FakeClient):
        def predict(self, model, *args, **kwargs):
            if model.id == "second":
                raise JobExecutionError("INFERENCE_FAILED", "Failed")
            return super().predict(model, *args, **kwargs)
    first = create_model(tmp_path / "model")
    second = replace(first, id="second", task="arousal")
    store = PredictionStore()
    runner = PackageInferenceRunner(store, prediction_client=FailingSecondClient())
    with pytest.raises(JobExecutionError):
        runner.run(create_session(tmp_path / "recording.edf"), [first, second], lambda *_: None, lambda: None)
    assert store.list_for_session("sess-test") == []


def test_runner_discards_results_if_cancelled_during_worker(tmp_path):
    cancelled = False
    class CancellingClient(FakeClient):
        def predict(self, *args, **kwargs):
            nonlocal cancelled
            cancelled = True
            kwargs["check_cancelled"]()
    def check_cancelled():
        if cancelled:
            raise JobCancelledError
    store = PredictionStore()
    runner = PackageInferenceRunner(store, prediction_client=CancellingClient())
    with pytest.raises(JobCancelledError):
        runner.run(create_session(tmp_path / "recording.edf"), [create_model(tmp_path / "model")], lambda *_: None, check_cancelled)
    assert store.list_for_session("sess-test") == []


def test_runner_rejects_non_edf_before_starting_worker(tmp_path):
    session = create_session(tmp_path / "recording.ndb")
    session.format = StudyFormat.NOX
    client = FakeClient()
    with pytest.raises(JobExecutionError):
        PackageInferenceRunner(PredictionStore(), prediction_client=client).run(session, [create_model(tmp_path / "model")], lambda *_: None, lambda: None)
    assert client.calls == []


def test_runner_requires_explicit_inference_environment():
    with pytest.raises(ValueError, match="separate inference Python"):
        PackageInferenceRunner(PredictionStore())


@pytest.mark.parametrize("batch_size", [0, -1])
def test_runner_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch size"):
        PackageInferenceRunner(PredictionStore(), prediction_client=FakeClient(), batch_size=batch_size)


def test_waiting_for_a_busy_inference_slot_can_be_cancelled(tmp_path):
    model = create_model(tmp_path / "model")
    session = create_session(tmp_path / "recording.edf")
    client = FakeClient()
    runner = PackageInferenceRunner(PredictionStore(), prediction_client=client)
    runner._slots.acquire()

    def cancel():
        raise JobCancelledError

    with pytest.raises(JobCancelledError):
        runner.run(session, [model], lambda *args: None, cancel)
    assert client.calls == []


def test_runner_gives_the_worker_a_relabeled_copy_and_records_the_mapping(tmp_path):
    import numpy as np
    from pyedflib import highlevel
    from sleepwalker.server.recording.relabel import read_edf_labels

    edf = tmp_path / "session" / "recording.edf"
    edf.parent.mkdir()
    headers = highlevel.make_signal_headers(["EEG Fpz-Cz", "C4-M1"], dimension="uV", sample_frequency=100, physical_min=-2, physical_max=2)
    highlevel.write_edf(str(edf), [np.zeros(1000), np.zeros(1000)], headers)
    model = replace(create_model(tmp_path / "model"), channel_aliases={"eeg": ("C3-M2", "C4-M1")})
    session = create_session(edf)
    seen = {}

    class InspectingClient(FakeClient):
        def predict(self, model, edf_path, **kwargs):
            seen["path"], seen["labels"] = edf_path, read_edf_labels(edf_path)
            return super().predict(model, edf_path, **kwargs)

    runner = PackageInferenceRunner(PredictionStore(), prediction_client=InspectingClient())
    runner.run(session, [model], lambda *args: None, lambda: None, channel_map={model.id: {"eeg": "EEG Fpz-Cz"}})
    assert seen["labels"][:2] == ["C3-M2", "unused 0"]
    assert seen["path"] != edf and not seen["path"].exists()
    assert read_edf_labels(edf)[:2] == ["EEG Fpz-Cz", "C4-M1"]
    assert session.model_provenance == [{**model.provenance(), "channel_map": {"eeg": "EEG Fpz-Cz"}}]
