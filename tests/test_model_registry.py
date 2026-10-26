import json
from dataclasses import replace
from pathlib import Path

import pytest

from sleepwalker.server.services.model_registry import InvalidModelPackageError, ModelInfo, ModelRegistry, UnsupportedModelPackageError, load_model_registry, read_model_manifest
from sleepwalker.server.stores.prediction_store import PredictionKind


def make_model(model_id):
    return ModelInfo(id=model_id, name=model_id, task="sleep", classes=("wake", "n1"), input_channels=("eeg",), capabilities=("classification",), output_resolution=30, package_path=Path("/models") / model_id)


def save_manifest(path, metadata):
    (path / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_registry_gets_registered_model():
    model = make_model("sleep-model")
    registry = ModelRegistry([model])
    assert registry.get(model.id) is model
    assert registry.get("unknown") is None


def test_registry_lists_models_sorted_by_id():
    assert [model.id for model in ModelRegistry([make_model("z"), make_model("a")]).list()] == ["a", "z"]


def test_registry_rejects_duplicate_model_id():
    with pytest.raises(ValueError, match="already registered"):
        ModelRegistry([make_model("duplicate"), make_model("duplicate")])


def test_read_model_manifest_without_loading_payload(make_model_package):
    path, metadata = make_model_package()
    before = (path / "model.pt").read_bytes()
    model = read_model_manifest(path)
    assert model == replace(make_model("test-model"), package_path=path, channel_aliases={"eeg": ("C3-M2", "C4-M1")}, payload_sha256=metadata["sha256"])
    assert model.output_kind == PredictionKind.EPOCH
    assert (path / "model.pt").read_bytes() == before


@pytest.mark.parametrize(("sequence_len", "target_resolution", "expected"), [(1, "30s", 30), (40, "40s", 1), (8, "40s", 5), (8, "80s", 10)])
def test_manifest_uses_resolution_per_output_step(make_model_package, sequence_len, target_resolution, expected):
    path, _ = make_model_package(sequence_len=sequence_len, target_resolution=target_resolution)
    assert read_model_manifest(path).output_resolution == expected


@pytest.mark.parametrize(("field", "value"), [("name", ""), ("name", "../escape"), ("task", None), ("input_channels", "eeg"), ("capabilities", []), ("capabilities", ["embeddings"]), ("sha256", "bad"), ("payload", "../weights.pt"), ("payload", "/tmp/weights.pt"), ("config", None)])
def test_manifest_rejects_invalid_metadata(make_model_package, field, value):
    path, metadata = make_model_package()
    metadata[field] = value
    save_manifest(path, metadata)
    with pytest.raises(InvalidModelPackageError, match="Invalid model metadata"):
        read_model_manifest(path)


@pytest.mark.parametrize(("field", "value"), [("sequence_len", 0), ("sequence_len", True), ("sequence_len", 1.5), ("classes", "wake"), ("classes", ["wake", "wake"]), ("target_resolution", "NaT"), ("target_resolution", "0s"), ("target_resolution", "-30s"), ("target_resolution", 30), ("type", "unknown")])
def test_manifest_rejects_invalid_classification(make_model_package, field, value):
    path, metadata = make_model_package()
    metadata["classification"][field] = value
    save_manifest(path, metadata)
    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(path)


@pytest.mark.parametrize("contents", ["{broken", "[]", "null"])
def test_manifest_rejects_invalid_json_document(make_model_package, contents):
    path, _ = make_model_package()
    (path / "manifest.json").write_text(contents, encoding="utf-8")
    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(path)


def test_manifest_rejects_missing_payload(make_model_package):
    path, _ = make_model_package()
    (path / "model.pt").unlink()
    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(path)


def test_manifest_rejects_inconsistent_aliases(make_model_package):
    path, metadata = make_model_package()
    metadata["config"]["data"]["channels"][0]["logical_name"] = "wrong"
    save_manifest(path, metadata)
    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(path)


def test_manifest_rejects_payload_symlink_outside_package(make_model_package, tmp_path):
    path, metadata = make_model_package()
    external = tmp_path / "outside.pt"
    external.write_bytes(b"outside")
    (path / "link.pt").symlink_to(external)
    metadata["payload"] = "link.pt"
    save_manifest(path, metadata)
    with pytest.raises(InvalidModelPackageError):
        read_model_manifest(path)


def test_registry_reads_model_subdirectories_and_skips_joint(make_model_package, tmp_path, caplog):
    make_model_package("z")
    make_model_package("a")
    joint, metadata = make_model_package("joint")
    metadata["classification"] = {"type": "multitask", "tasks": {}}
    save_manifest(joint, metadata)
    (tmp_path / "old.swmodel").write_bytes(b"retired format")
    (tmp_path / "notes.txt").write_text("not a model")
    registry = load_model_registry(tmp_path)
    assert [model.id for model in registry.list()] == ["a", "z"]
    assert "Multi-task package is not enabled yet: joint" in caplog.text
    with pytest.raises(UnsupportedModelPackageError):
        read_model_manifest(joint)


def test_registry_does_not_silently_skip_broken_packages(make_model_package, tmp_path):
    path, _ = make_model_package()
    (path / "manifest.json").write_text("{}")
    with pytest.raises(InvalidModelPackageError):
        load_model_registry(tmp_path)


def test_load_model_registry_from_empty_directory(tmp_path):
    assert load_model_registry(tmp_path).list() == []


def test_load_model_registry_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="Model directory does not exist"):
        load_model_registry(tmp_path / "missing")
