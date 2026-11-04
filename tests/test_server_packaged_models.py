from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient
from pyedflib import highlevel
import pytest

from sleepwalker.server.inference.catalog import ModelCatalog
from sleepwalker.server.app import create_local_app
from sleepwalker.server.core.jobs import JobCancelledError
from sleepwalker.server.inference.runner import SubprocessInferenceClient


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
        corrected_label = result["classes"][0]
        corrected = client.post(f"/api/v1/sessions/{session_id}/annotations", json={"operation": "set_label", "task": task, "start": first_start, "label": corrected_label})
        assert corrected.status_code == 201, corrected.text
        assert corrected.json()["start"] == first_start
        assert client.get(f"/api/v1/sessions/{session_id}/predictions/{task}").json() == result
        if task == "desaturation":
            # Exercise the real 35-second offset through HTTP export/import too.
            exported = client.post(f"/api/v1/sessions/{session_id}/exports", json={"format": "sleepwalker", "mode": "workspace"})
            assert exported.status_code == 202, exported.text
            downloaded = client.get(f"/api/v1/exports/{exported.json()['export_id']}/download")
            assert downloaded.status_code == 200
            imported = client.post("/api/v1/sessions/import", files={"bundle": ("corrected.sleepwalker.zip", downloaded.content, "application/zip")})
            assert imported.status_code == 202, imported.text
            imported_id = imported.json()["session_id"]
            imported_job = client.get(f"/api/v1/jobs/{imported.json()['job_id']}").json()
            assert imported_job["state"] == "succeeded", imported_job
            restored = client.get(f"/api/v1/sessions/{imported_id}/annotations").json()
            assert len(restored["labels"]) == 1
            assert restored["labels"][0]["start"] == first_start
            assert restored["labels"][0]["label"] == corrected_label
            assert client.get(f"/api/v1/sessions/{imported_id}/predictions/{task}").json() == result
            assert client.delete(f"/api/v1/sessions/{imported_id}").status_code == 204
        assert client.delete(f"/api/v1/sessions/{session_id}").status_code == 204


@pytest.mark.parametrize("problem", ["payload_checksum", "unknown_class"])
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
        # A class the model never outputs: the runner must reject the result.
        metadata["classification"]["classes"][0] = "not_a_model_class"
        (modified / "manifest.json").write_text(json.dumps(metadata))
    app = create_local_app(modified.parent, inference_python=python, batch_size=1)
    model = app.state.model_catalog.list()[0]
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
    model = next(model for model in app.state.model_catalog.list() if model.task == "sleep")
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


def test_real_worker_cancellation_removes_framework_edf_cache(real_model_configuration, tmp_path):
    python, models = real_model_configuration
    model = next(model for model in ModelCatalog(models).list() if model.task == "sleep")
    # Long enough that the cache is still filled when the job is cancelled.
    edf = tmp_path / "long.edf"
    t = np.arange(3 * 3600 * 100) / 100
    header = highlevel.make_signal_header("C3-M2", dimension="uV", sample_frequency=100, physical_min=-500, physical_max=500)
    assert highlevel.write_edf(str(edf), [35 * np.sin(2 * np.pi * 10 * t)], [header], header=highlevel.make_header(startdate=datetime(2025, 1, 1)))
    # The framework puts its cache in /dev/shm, or the temporary directory if that is not writable.
    def caches():
        return {path for root in (Path("/dev/shm"), Path(tempfile.gettempdir())) for path in root.glob("sleepwalker-edf-cache-*")}
    before = caches()
    created = set()
    started = time.monotonic()
    def check_cancelled():
        created.update(caches() - before)
        # Cancel once the recording is in RAM; that is what must not be left behind.
        if any(any(path.iterdir()) for path in created if path.exists()):
            raise JobCancelledError
        assert time.monotonic() - started < 30, "Real worker did not fill its EDF cache"
    with pytest.raises(JobCancelledError):
        SubprocessInferenceClient(python).predict(model, edf, device="cpu", batch_size=1, check_cancelled=check_cancelled)
    assert created and not any(path.exists() for path in created)


def test_real_model_runs_with_a_hand_mapped_channel_and_exports_the_mapping(real_model_configuration, synthetic_model_edf, tmp_path):
    from sleepwalker.server.recording.relabel import relabel_edf

    python, models = real_model_configuration
    renamed = tmp_path / "renamed.edf"
    relabel_edf(synthetic_model_edf, renamed, {"C3-M2": "EEG Fpz-Cz"})
    app = create_local_app(models, inference_python=python, batch_size=1)
    with TestClient(app) as client:
        model = next(model for model in client.get("/api/v1/models").json()["models"] if model["task"] == "sleep")
        with renamed.open("rb") as file:
            session_id = client.post("/api/v1/sessions", files={"files": (renamed.name, file)}).json()["id"]
        unmapped = client.post(f"/api/v1/sessions/{session_id}/inference", json={"models": [model["id"]]})
        assert unmapped.json()["error"]["code"] == "MISSING_REQUIRED_CHANNEL"
        accepted = client.post(f"/api/v1/sessions/{session_id}/inference", json={"models": [model["id"]], "channel_map": {model["id"]: {"eeg": "EEG Fpz-Cz"}}})
        assert accepted.status_code == 202, accepted.text
        job = client.get(f"/api/v1/jobs/{accepted.json()['job_id']}").json()
        assert job["state"] == "succeeded", job
        assert client.get(f"/api/v1/sessions/{session_id}/predictions/sleep").json()["predictions"]
        exported = client.post(f"/api/v1/sessions/{session_id}/exports", json={"format": "sleepwalker", "mode": "workspace"})
        assert exported.status_code == 202, exported.text
        bundle = client.get(f"/api/v1/exports/{exported.json()['export_id']}/download").content
        imported = client.post("/api/v1/sessions/import", files={"bundle": ("mapped.sleepwalker.zip", bundle, "application/zip")})
        assert client.get(f"/api/v1/jobs/{imported.json()['job_id']}").json()["state"] == "succeeded"
    session = app.state.session_service.peek(imported.json()["session_id"])
    assert session.model_provenance[0]["channel_map"] == {"eeg": "EEG Fpz-Cz"}
