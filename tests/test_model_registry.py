import pytest
from pathlib import Path
import json
from zipfile import ZipFile
from sleepwalker.server.services.model_registry import ModelInfo, ModelRegistry, read_model_manifest, InvalidModelPackageError


def make_model(model_id: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name=f"Model {model_id}",
        task="sleep_stage",
        classes=("wake", "n1", "n2", "n3", "rem"),
        input_channels=("eeg",),
        capabilities=("classification",),
        output_resolution=30.0,
        package_path=Path(f"/models/{model_id}.swmodel"),
    )


def test_registry_gets_registered_model() -> None:
    model = make_model("sleep-staging-attnsleep")
    registry = ModelRegistry([model])

    assert registry.get(model.id) is model
    assert registry.get("unknown-model") is None


def test_registry_lists_models_sorted_by_id() -> None:
    registry = ModelRegistry(
        [
            make_model("model-z"),
            make_model("model-a"),
        ]
    )

    assert [model.id for model in registry.list()] == ["model-a", "model-z"]


def test_registry_rejects_duplicate_model_id() -> None:
    registry = ModelRegistry([make_model("duplicate")])

    with pytest.raises(ValueError, match="already registered"):
        registry.add(make_model("duplicate"))
        
def test_read_model_manifest_from_swmodel_package(tmp_path) -> None:
    package_path = tmp_path / "sleep-model.swmodel"

    metadata = {
        "format": "sleepwalker-swmodel-v1",
        "model": {
            "id": "sleep-model",
            "name": "Sleep Stage Model",
            "task": "sleep_stage",
            "classes": ["wake", "n1", "n2", "n3", "rem"],
            "input_channels": ["eeg"],
            "capabilities": ["classification"],
            "output_resolution": 30.0,
        },
    }

    with ZipFile(package_path, "w") as package:
        package.writestr(
            "meta.json",
            json.dumps(metadata),
        )

    model = read_model_manifest(package_path)

    assert model == ModelInfo(
        id="sleep-model",
        name="Sleep Stage Model",
        task="sleep_stage",
        classes=("wake", "n1", "n2", "n3", "rem"),
        input_channels=("eeg",),
        capabilities=("classification",),
        output_resolution=30.0,
        package_path=package_path,
    )

def test_read_model_manifest_rejects_non_zip_file(tmp_path) -> None:
    package_path = tmp_path / "broken.swmodel"
    package_path.write_bytes(b"not a ZIP file")

    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(package_path)


def test_read_model_manifest_rejects_missing_metadata(tmp_path) -> None:
    package_path = tmp_path / "missing-meta.swmodel"

    with ZipFile(package_path, "w") as package:
        package.writestr("model.pt", b"weights")

    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(package_path)


def test_read_model_manifest_rejects_invalid_json(tmp_path) -> None:
    package_path = tmp_path / "invalid-json.swmodel"

    with ZipFile(package_path, "w") as package:
        package.writestr("meta.json", "{not valid JSON")

    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(package_path)

def test_read_model_manifest_rejects_unsupported_format(tmp_path) -> None:
    package_path = tmp_path / "unsupported.swmodel"

    metadata = {
        "format": "some-other-format",
    }

    with ZipFile(package_path, "w") as package:
        package.writestr(
            "meta.json",
            json.dumps(metadata),
        )

    with pytest.raises(
        InvalidModelPackageError,
        match="Unsupported model package format",
    ):
        read_model_manifest(package_path)