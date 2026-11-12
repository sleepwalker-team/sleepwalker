import os
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sleepwalker.server.inference.catalog import InvalidModelPackageError, ModelInfo, ModelCatalog, UnsupportedModelPackageError, read_model_manifest
from sleepwalker.server.results.predictions import PredictionKind


def make_model(model_id):
    return ModelInfo(id=model_id, name=model_id, task="sleep", classes=("wake", "n1"), input_channels=("eeg",), output_resolution=30, package_path=Path("/models") / model_id)


def save_manifest(path, metadata):
    (path / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_catalog_gets_registered_model():
    model = make_model("sleep-model")
    catalog = ModelCatalog(models=[model])
    assert catalog.get(model.id) is model
    assert catalog.get("unknown") is None


def test_catalog_lists_models_sorted_by_id():
    assert [model.id for model in ModelCatalog(models=[make_model("z"), make_model("a")]).list()] == ["a", "z"]


def test_read_model_manifest_without_loading_payload(make_model_package):
    path, metadata = make_model_package()
    before = (path / "model.pt").read_bytes()
    model = read_model_manifest(path)
    assert model == replace(make_model("test-model"), package_path=path, channel_aliases={"eeg": ("C3-M2", "C4-M1")}, payload_sha256=metadata["sha256"])
    assert model.output_kind == PredictionKind.EPOCH
    assert (path / "model.pt").read_bytes() == before


@pytest.mark.parametrize(("classes", "kind"), [(["wake", "n1"], PredictionKind.EPOCH), (["no_arousal", "arousal"], PredictionKind.EVENT), (["apnea", "hypopnea", "regular breathing"], PredictionKind.EVENT)])
def test_manifest_detects_event_tasks_by_their_background_class(make_model_package, classes, kind):
    path, metadata = make_model_package()
    metadata["classification"]["classes"] = classes
    save_manifest(path, metadata)
    assert read_model_manifest(path).output_kind == kind


@pytest.mark.parametrize(("sequence_len", "target_resolution", "expected"), [(1, "30s", 30), (40, "40s", 1), (8, "40s", 5), (8, "80s", 10)])
def test_manifest_uses_resolution_per_output_step(make_model_package, sequence_len, target_resolution, expected):
    path, _ = make_model_package(sequence_len=sequence_len, target_resolution=target_resolution)
    assert read_model_manifest(path).output_resolution == expected


@pytest.mark.parametrize(("field", "value"), [("name", ""), ("name", "../escape"), ("task", None), ("input_channels", "eeg"), ("sha256", "bad"), ("payload", "../weights.pt"), ("payload", "/tmp/weights.pt"), ("config", None)])
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


def test_catalog_reads_model_subdirectories_and_skips_joint(make_model_package, tmp_path, caplog):
    make_model_package("z")
    make_model_package("a")
    joint, metadata = make_model_package("joint")
    metadata["classification"] = {"type": "multitask", "tasks": {}}
    save_manifest(joint, metadata)
    (tmp_path / "old.swmodel").write_bytes(b"retired format")
    (tmp_path / "notes.txt").write_text("not a model")
    catalog = ModelCatalog(tmp_path)
    assert [model.id for model in catalog.list()] == ["a", "z"]
    assert "Multi-task package is not enabled yet: joint" in caplog.text
    with pytest.raises(UnsupportedModelPackageError):
        read_model_manifest(joint)


def test_catalog_from_empty_directory(tmp_path):
    assert ModelCatalog(tmp_path).list() == []


def test_catalog_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="Model directory does not exist"):
        ModelCatalog(tmp_path / "missing")


def test_catalog_skips_broken_package_and_logs_it_once(make_model_package, tmp_path, caplog):
    make_model_package("good")
    broken, _ = make_model_package("broken")
    (broken / "manifest.json").write_text("{}")
    catalog = ModelCatalog(tmp_path)
    assert [model.id for model in catalog.list()] == ["good"]
    assert [model.id for model in catalog.list()] == ["good"]
    assert caplog.text.count("Skipping model package") == 1


def test_catalog_sees_new_packages_without_restart(make_model_package, tmp_path):
    catalog = ModelCatalog(tmp_path)
    assert catalog.list() == []
    make_model_package("new-model")
    assert catalog.get("new-model") is not None


def test_catalog_uses_newest_package_of_a_retrained_model(make_model_package, tmp_path):
    older, metadata = make_model_package("2026-11-01")
    newer, _ = make_model_package("2026-11-02")
    metadata["name"] = "2026-11-02"
    save_manifest(older, metadata)
    os.utime(older / "manifest.json", ns=(1, 1))
    assert ModelCatalog(tmp_path).get("2026-11-02").package_path == newer.resolve()


def test_provenance_identifies_the_exact_package(make_model_package, tmp_path):
    path, metadata = make_model_package("sleep")
    provenance = read_model_manifest(path).provenance()
    assert provenance["id"] == "sleep"
    assert provenance["version"] == f"sha256:{metadata['sha256']}"
