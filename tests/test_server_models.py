from pathlib import Path

from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.model_registry import ModelInfo, ModelRegistry


def test_list_models_returns_registered_models_sorted_by_id() -> None:
    registry = ModelRegistry(
        [
            ModelInfo(
                id="model-z",
                name="Arousal Model",
                task="arousal",
                classes=("no_arousal", "arousal"),
                input_channels=("eeg",),
                capabilities=("classification",),
                output_resolution=0.1,
                package_path=Path("/models/model-z.swmodel"),
            ),
            ModelInfo(
                id="model-a",
                name="Sleep Stage Model",
                task="sleep_stage",
                classes=("wake", "n1", "n2", "n3", "rem"),
                input_channels=("eeg",),
                capabilities=("classification",),
                output_resolution=30.0,
                package_path=Path("/models/model-a.swmodel"),
            ),
        ]
    )

    with TestClient(create_app(model_registry=registry)) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "id": "model-a",
                "name": "Sleep Stage Model",
                "task": "sleep_stage",
                "classes": ["wake", "n1", "n2", "n3", "rem"],
                "input_channels": ["eeg"],
                "capabilities": ["classification"],
                "output_resolution": 30.0,
            },
            {
                "id": "model-z",
                "name": "Arousal Model",
                "task": "arousal",
                "classes": ["no_arousal", "arousal"],
                "input_channels": ["eeg"],
                "capabilities": ["classification"],
                "output_resolution": 0.1,
            },
        ]
    }


def test_get_model_returns_registered_model() -> None:
    model = ModelInfo(
        id="sleep-model",
        name="Sleep Stage Model",
        task="sleep_stage",
        classes=("wake", "n1", "n2", "n3", "rem"),
        input_channels=("eeg",),
        capabilities=("classification",),
        output_resolution=30.0,
        package_path=Path("/models/sleep-model.swmodel"),
    )

    with TestClient(
        create_app(model_registry=ModelRegistry([model]))
    ) as client:
        response = client.get("/api/v1/models/sleep-model")

    assert response.status_code == 200
    assert response.json()["id"] == "sleep-model"
    assert response.json()["classes"] == [
        "wake",
        "n1",
        "n2",
        "n3",
        "rem",
    ]


def test_get_unknown_model_returns_not_found() -> None:
    with TestClient(
        create_app(model_registry=ModelRegistry())
    ) as client:
        response = client.get("/api/v1/models/unknown")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "MODEL_NOT_FOUND",
            "message": "Model not found.",
            "details": None,
        }
    }
