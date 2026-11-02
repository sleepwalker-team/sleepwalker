from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.api.schemas import SessionState
from sleepwalker.server.core.jobs import JobExecutionError
from sleepwalker.server.core.jobs import JobService
from sleepwalker.server.inference.catalog import ModelInfo, ModelCatalog
from sleepwalker.server.core.sessions import SessionService
from sleepwalker.server.core.jobs import JobPhase, JobState
from sleepwalker.server.recording.study import SignalInfo


class SuccessfulInferenceRunner:
    def __init__(self) -> None:
        self.session_id: str | None = None
        self.model_ids: list[str] = []

    def run(self, session, models, report_progress, check_cancelled) -> None:
        self.session_id = session.id
        self.model_ids = [model.id for model in models]
        check_cancelled()
        report_progress(JobPhase.PREPROCESSING, None, "Preparing EDF data")
        report_progress(JobPhase.INFERENCE, 0.5, "Running inference")


class FailingInferenceRunner:
    def run(self, _session, models, _report_progress, _check_cancelled) -> None:
        raise JobExecutionError(
            code="MODEL_INCOMPATIBLE",
            message="Model is incompatible with the recording.",
            details={"model_id": models[0].id},
        )


def create_model() -> ModelInfo:
    return ModelInfo(
        id="sleep-model",
        name="Sleep Stage Model",
        task="sleep_stage",
        classes=("wake", "n1", "n2", "n3", "rem"),
        input_channels=("eeg",),
        capabilities=("classification",),
        output_resolution=30.0,
        package_path=Path("/models/sleep-model.swmodel"),
    )


def test_start_inference_creates_queued_job() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)]
    model = create_model()
    job_store = JobService()
    runner = SuccessfulInferenceRunner()

    app = create_app(
        model_catalog=ModelCatalog(models=[model]),
        job_service=job_store,
        session_service=session_service,
        inference_runner=runner,
    )

    with TestClient(app) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": [model.id]})

    assert response.status_code == 202
    assert response.json()["state"] == "queued"

    job = job_store.get(response.json()["job_id"])
    assert job is not None
    assert job.session_id == session.id
    assert job.type == "inference"
    assert job.parameters == {"models": [model.id]}
    assert job.state == JobState.SUCCEEDED
    assert runner.session_id == session.id
    assert runner.model_ids == [model.id]


def test_start_inference_accepts_physical_signal_aliases() -> None:
    service = SessionService()
    session = service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [SignalInfo(id="C3-M2", label="C3-M2", unit="uV", sample_rate=100)]
    model = replace(create_model(), channel_aliases={"eeg": ("C3-M2", "C4-M1")})
    runner = SuccessfulInferenceRunner()
    with TestClient(create_app(session_service=service, model_catalog=ModelCatalog(models=[model]), inference_runner=runner)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": [model.id]})
    assert response.status_code == 202
    assert runner.model_ids == [model.id]


def test_start_inference_does_not_accept_logical_name_instead_of_declared_aliases() -> None:
    service = SessionService()
    session = service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=100)]
    model = replace(create_model(), channel_aliases={"eeg": ("C3-M2", "C4-M1")})
    with TestClient(create_app(session_service=service, model_catalog=ModelCatalog(models=[model]), inference_runner=SuccessfulInferenceRunner())) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": [model.id]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MISSING_REQUIRED_CHANNEL"


def test_start_inference_rejects_unknown_session() -> None:
    model = create_model()

    with TestClient(create_app(model_catalog=ModelCatalog(models=[model]))) as client:
        response = client.post("/api/v1/sessions/unknown/inference", json={"models": [model.id]})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_start_inference_rejects_unknown_model() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY

    with TestClient(create_app(model_catalog=ModelCatalog(), session_service=session_service)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": ["unknown"]})

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "MODEL_NOT_FOUND",
        "message": "Model not found.",
        "details": {"model_id": "unknown"},
    }


def test_start_inference_rejects_empty_model_list() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY

    with TestClient(create_app(session_service=session_service)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": []})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_start_inference_rejects_duplicate_models() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY

    with TestClient(create_app(session_service=session_service)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": ["model-a", "model-a"]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_start_inference_rejects_second_active_job() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [
        SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)
    ]
    model = create_model()
    job_service = JobService()
    active_job = job_service.create(
        session_id=session.id,
        job_type="inference",
    )

    app = create_app(
        model_catalog=ModelCatalog(models=[model]),
        job_service=job_service,
        session_service=session_service,
        inference_runner=SuccessfulInferenceRunner(),
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/inference",
            json={"models": [model.id]},
        )

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "JOB_ALREADY_RUNNING",
        "message": "Session already has an active job.",
        "details": {
            "job_id": active_job.id,
            "state": "queued",
        },
    }


def test_start_inference_rejects_missing_required_channel() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [SignalInfo(id="EOG", label="EOG", unit="uV", sample_rate=100.0)]
    model = create_model()

    with TestClient(create_app(model_catalog=ModelCatalog(models=[model]), session_service=session_service)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": [model.id]})

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "MISSING_REQUIRED_CHANNEL",
        "message": "Required signal could not be resolved.",
        "details": {"model_id": model.id, "channels": ["eeg"]},
    }


def test_start_inference_rejects_session_that_is_not_ready() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.CREATED
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)]
    model = create_model()

    with TestClient(create_app(model_catalog=ModelCatalog(models=[model]), session_service=session_service)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": [model.id]})

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "INVALID_SESSION_STATE",
        "message": "Session is not ready for inference.",
        "details": {"state": "created"},
    }


def test_start_inference_rejects_unavailable_compute() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)]
    model = create_model()

    with TestClient(create_app(model_catalog=ModelCatalog(models=[model]), session_service=session_service)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": [model.id]})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "COMPUTE_UNAVAILABLE"


def test_start_inference_records_background_failure() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)]
    model = create_model()
    job_store = JobService()

    app = create_app(
        model_catalog=ModelCatalog(models=[model]),
        job_service=job_store,
        session_service=session_service,
        inference_runner=FailingInferenceRunner(),
    )

    with TestClient(app) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/inference", json={"models": [model.id]})

    assert response.status_code == 202
    assert response.json()["state"] == "queued"

    job = job_store.get(response.json()["job_id"])
    assert job is not None
    assert job.state == JobState.FAILED
    assert job.error is not None
    assert job.error.code == "MODEL_INCOMPATIBLE"


def test_start_inference_rejects_models_with_duplicate_task() -> None:
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)]
    first_model = create_model()
    second_model = replace(first_model, id="second-sleep-model")

    app = create_app(
        model_catalog=ModelCatalog(models=[first_model, second_model]),
        session_service=session_service,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/inference",
            json={"models": [first_model.id, second_model.id]},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MODEL_INCOMPATIBLE"
    assert response.json()["error"]["details"] == {
        "task": "sleep_stage",
        "model_ids": [first_model.id, second_model.id],
    }
