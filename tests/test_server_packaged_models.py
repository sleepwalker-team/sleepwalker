from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import numpy as np
from fastapi.testclient import TestClient
from pyedflib import highlevel
import pytest

from sleepwalker.server.app import create_local_app
from sleepwalker.server.services import subprocess_inference
from sleepwalker.server.services.job_executor import JobCancelledError
from sleepwalker.server.services.model_registry import load_model_registry
from sleepwalker.server.services.subprocess_inference import SubprocessInferenceClient


@pytest.fixture
def real_model_configuration(request):
    python = request.config.getoption("--inference-python")
    models = request.config.getoption("--inference-models")
    if not python or not models:
        pytest.skip("Provide --inference-python and --inference-models for real-model API integration tests")
    return Path(python), Path(models)


@pytest.fixture
def synthetic_model_edf(tmp_path):
    path = tmp_path / "synthetic.edf"
    rng = np.random.default_rng(42)
    t = np.arange(690 * 100) / 100
    definitions = [
        ("C3-M2", "uV", -500, 500, 35 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 5, len(t))),
        ("E1-M2", "uV", -500, 500, 25 * np.sin(2 * np.pi * t) + rng.normal(0, 3, len(t))),
        ("CHIN1-CHIN2", "uV", -500, 500, 10 * np.sin(2 * np.pi * 20 * t) + rng.normal(0, 3, len(t))),
        ("EKG", "mV", -5, 5, np.sin(2 * np.pi * 1.2 * t)),
        ("ABD", "a.u.", -10, 10, np.sin(2 * np.pi * 0.25 * t)),
        ("CHEST", "a.u.", -10, 10, np.sin(2 * np.pi * 0.25 * t + 0.1)),
        ("PTAF", "a.u.", -10, 10, np.sin(2 * np.pi * 0.25 * t + 0.2)),
        ("SaO2", "%", 0, 100, 94 + 2 * np.sin(2 * np.pi * 0.01 * t)),
    ]
    headers = [highlevel.make_signal_header(label, dimension=unit, sample_frequency=100, physical_min=lower, physical_max=upper) for label, unit, lower, upper, _ in definitions]
    assert highlevel.write_edf(str(path), [values for *_, values in definitions], headers, header=highlevel.make_header(startdate=datetime(2025, 1, 1)))
    return path


@pytest.mark.parametrize(("task", "first_start", "resolution", "rows"), [("sleep", 300, 30, 2), ("arousal", 10, 1, 600), ("breathing", 25, 5, 120), ("desaturation", 35, 10, 48)])
def test_real_package_from_edf_upload_to_api_predictions(real_model_configuration, synthetic_model_edf, task, first_start, resolution, rows):
    python, models = real_model_configuration
    app = create_local_app(models, inference_python=python, batch_size=1)
    with TestClient(app) as client:
        listed = client.get("/api/v1/models").json()["models"]
        assert {model["task"] for model in listed} == {"sleep", "arousal", "breathing", "desaturation"}
        model = next(model for model in listed if model["task"] == task)
        with synthetic_model_edf.open("rb") as file:
            uploaded = client.post("/api/v1/sessions", files={"files": (synthetic_model_edf.name, file, "application/octet-stream")})
        assert uploaded.status_code == 201
        session_id = uploaded.json()["id"]
        accepted = client.post(f"/api/v1/sessions/{session_id}/inference", json={"models": [model["id"]]})
        assert accepted.status_code == 202, accepted.text
        job = client.get(f"/api/v1/jobs/{accepted.json()['job_id']}").json()
        assert job["state"] == "succeeded", job
        response = client.get(f"/api/v1/sessions/{session_id}/predictions/{task}")
        assert response.status_code == 200
        result = response.json()
        assert result["kind"] == "epoch"
        assert result["classes"] == model["classes"]
        assert len(result["predictions"]) == rows
        assert result["predictions"][0]["start"] == first_start
        assert result["predictions"][0]["end"] == first_start + resolution
        assert all(prediction["end"] - prediction["start"] == resolution for prediction in result["predictions"])
        tasks = client.get(f"/api/v1/sessions/{session_id}/tasks").json()["tasks"]
        assert tasks[0]["id"] == task
        assert tasks[0]["resolution"] == resolution
        assert client.delete(f"/api/v1/sessions/{session_id}").status_code == 204


@pytest.mark.parametrize("problem", ["payload_checksum", "executable_contract"])
def test_real_worker_rejects_modified_model_without_partial_results(real_model_configuration, synthetic_model_edf, tmp_path, problem):
    python, models = real_model_configuration
    modified = tmp_path / "modified-models" / "sw-sleep"
    modified.mkdir(parents=True)
    for name in ["manifest.json", "model.pt"]:
        shutil.copy2(models / "sw-sleep" / name, modified / name)
    if problem == "payload_checksum":
        with (modified / "model.pt").open("r+b") as payload:
            payload.write(b"corrupted")
    else:
        metadata = json.loads((modified / "manifest.json").read_text())
        metadata["classification"]["classes"].reverse()
        (modified / "manifest.json").write_text(json.dumps(metadata))
    app = create_local_app(modified.parent, inference_python=python, batch_size=1)
    model = app.state.model_registry.list()[0]
    with TestClient(app) as client:
        with synthetic_model_edf.open("rb") as file:
            created = client.post("/api/v1/sessions", files={"files": ("synthetic.edf", file)})
        session_id = created.json()["id"]
        accepted = client.post(f"/api/v1/sessions/{session_id}/inference", json={"models": [model.id]})
        assert accepted.status_code == 202
        job = client.get(f"/api/v1/jobs/{accepted.json()['job_id']}").json()
        assert job["state"] == "failed", job
        assert job["error"]["code"] == "MODEL_INCOMPATIBLE"
        assert "Traceback" not in json.dumps(job["error"])
        assert str(modified) not in json.dumps(job["error"])
        assert app.state.prediction_store.list_for_session(session_id) == []


def test_real_worker_rejects_recording_shorter_than_context(real_model_configuration, tmp_path):
    python, models = real_model_configuration
    path = tmp_path / "short.edf"
    headers = [highlevel.make_signal_header("C3-M2", dimension="uV", sample_frequency=100, physical_min=-500, physical_max=500)]
    assert highlevel.write_edf(str(path), [np.sin(np.arange(3000))], headers, header=highlevel.make_header(startdate=datetime(2025, 1, 1)))
    app = create_local_app(models, inference_python=python, batch_size=1)
    model = next(model for model in app.state.model_registry.list() if model.task == "sleep")
    with TestClient(app) as client:
        with path.open("rb") as file:
            created = client.post("/api/v1/sessions", files={"files": (path.name, file)})
        session_id = created.json()["id"]
        accepted = client.post(f"/api/v1/sessions/{session_id}/inference", json={"models": [model.id]})
        assert accepted.status_code == 202
        job = client.get(f"/api/v1/jobs/{accepted.json()['job_id']}").json()
        assert job["state"] == "failed", job
        assert job["error"]["code"] == "MODEL_INCOMPATIBLE"
        assert app.state.prediction_store.list_for_session(session_id) == []


def test_real_worker_cancellation_removes_owned_edf_cache(real_model_configuration, synthetic_model_edf, tmp_path, monkeypatch):
    python, models = real_model_configuration
    model = next(model for model in load_model_registry(models).list() if model.task == "sleep")
    original_directory = subprocess_inference.TemporaryDirectory
    paths = []
    def create_directory(**kwargs):
        directory = original_directory(dir=tmp_path, **kwargs)
        paths.append(Path(directory.name))
        return directory
    monkeypatch.setattr(subprocess_inference, "TemporaryDirectory", create_directory)
    started = time.monotonic()
    def check_cancelled():
        if any(list(path.glob("sleepwalker-edf-cache-*")) for path in paths):
            raise JobCancelledError
        assert time.monotonic() - started < 20, "Real worker did not create its owned EDF cache"
    with pytest.raises(JobCancelledError):
        SubprocessInferenceClient(python).predict(model, synthetic_model_edf, device="cpu", batch_size=1, check_cancelled=check_cancelled)
    assert paths and all(not path.exists() for path in paths)
