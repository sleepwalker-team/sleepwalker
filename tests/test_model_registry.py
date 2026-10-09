import pytest
from pathlib import Path
import json
from zipfile import ZipFile
from sleepwalker.server.services.model_registry import ModelInfo, ModelRegistry, read_model_manifest, InvalidModelPackageError
from sleepwalker.server.services.model_registry import InvalidModelPackageError, ModelInfo, ModelRegistry, load_model_registry, read_model_manifest


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

def test_read_model_manifest_rejects_missing_model_block(tmp_path) -> None:
    package_path = tmp_path / "missing-model.swmodel"

    metadata = {
        "format": "sleepwalker-swmodel-v1",
    }

    with ZipFile(package_path, "w") as package:
        package.writestr(
            "meta.json",
            json.dumps(metadata),
        )

    with pytest.raises(
        InvalidModelPackageError,
        match="Invalid model metadata",
    ):
        read_model_manifest(package_path)

def test_read_model_manifest_rejects_non_list_classes(tmp_path) -> None:
    package_path = tmp_path / "invalid-classes.swmodel"

    metadata = {
        "format": "sleepwalker-swmodel-v1",
        "model": {
            "id": "sleep-model",
            "name": "Sleep Stage Model",
            "task": "sleep_stage",
            "classes": "wake",
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

    with pytest.raises(
        InvalidModelPackageError,
        match="Invalid model metadata",
    ):
        read_model_manifest(package_path)

def make_manifest_metadata() -> dict:
    return {
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

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("name", 123),
        ("task", None),
    ],
)
def test_read_model_manifest_rejects_invalid_string_field(
    tmp_path,
    field,
    value,
) -> None:
    package_path = tmp_path / f"invalid-{field}.swmodel"
    metadata = make_manifest_metadata()
    metadata["model"][field] = value

    with ZipFile(package_path, "w") as package:
        package.writestr("meta.json", json.dumps(metadata))

    with pytest.raises(
        InvalidModelPackageError,
        match="Invalid model metadata",
    ):
        read_model_manifest(package_path)


@pytest.mark.parametrize(
    "resolution",
    [0, -1, "30.0", True],
)
def test_read_model_manifest_rejects_invalid_output_resolution(
    tmp_path,
    resolution,
) -> None:
    package_path = tmp_path / "invalid-resolution.swmodel"
    metadata = make_manifest_metadata()
    metadata["model"]["output_resolution"] = resolution

    with ZipFile(package_path, "w") as package:
        package.writestr("meta.json", json.dumps(metadata))

    with pytest.raises(
        InvalidModelPackageError,
        match="Invalid model metadata",
    ):
        read_model_manifest(package_path)

def test_load_model_registry_from_directory(tmp_path) -> None:
    for model_id in ("model-z", "model-a"):
        package_path = tmp_path / f"{model_id}.swmodel"
        metadata = make_manifest_metadata()
        metadata["model"]["id"] = model_id
        metadata["model"]["name"] = f"Model {model_id}"

        with ZipFile(package_path, "w") as package:
            package.writestr(
                "meta.json",
                json.dumps(metadata),
            )

    # Dateien ohne .swmodel-Endung sollen ignoriert werden.
    (tmp_path / "notes.txt").write_text(
        "not a model",
        encoding="utf-8",
    )

    registry = load_model_registry(tmp_path)

    assert [model.id for model in registry.list()] == [
        "model-a",
        "model-z",
    ]
    assert [model.package_path.name for model in registry.list()] == [
        "model-a.swmodel",
        "model-z.swmodel",
    ]

def test_load_model_registry_from_empty_directory(tmp_path) -> None:
    registry = load_model_registry(tmp_path)

    assert registry.list() == []


def test_load_model_registry_rejects_missing_directory(tmp_path) -> None:
    missing_directory = tmp_path / "missing"

    with pytest.raises(
        ValueError,
        match="Model directory does not exist",
    ):
        load_model_registry(missing_directory)