import hashlib
import json

import pytest


def pytest_addoption(parser):
    parser.addoption("--inference-python", help="Separate framework Python for opt-in real-model API tests.")
    parser.addoption("--inference-models", help="Directory containing trusted real-model package subdirectories.")


@pytest.fixture
def make_model_package(tmp_path):
    def create(name="test-model", *, task="sleep", sequence_len=1, target_resolution="30s"):
        path = tmp_path / name
        path.mkdir()
        payload = b"test payload; never deserialized by metadata tests"
        (path / "model.pt").write_bytes(payload)
        metadata = {
            "format_version": "sleepwalker-packaged-model-v1", "name": name, "task": task,
            "payload": "model.pt", "sha256": hashlib.sha256(payload).hexdigest(),
            "classification": {"type": "single-head-multiclass", "classes": ["wake", "n1"], "sequence_len": sequence_len, "target_resolution": target_resolution},
            "input_channels": ["eeg"], "capabilities": ["classification"],
            "config": {"data": {"channels": [{"logical_name": "eeg", "physical_names": ["C3-M2", "C4-M1"]}]}},
        }
        (path / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path, metadata
    return create


@pytest.fixture(autouse=True)
def isolate_default_server_storage(request, monkeypatch):
    name = request.node.path.name
    if not name.startswith("test_server_") and name not in {
        "test_workspace_import.py", "test_storage_lifecycle.py", "test_job_events.py",
    }:
        return

    from sleepwalker.server.storage.temporary_storage import TemporaryStorage

    isolated_root = request.getfixturevalue("tmp_path") / "default-server-storage"
    original_init = TemporaryStorage.__init__

    def isolated_init(self, root=None, **kwargs):
        original_init(self, isolated_root if root is None else root, **kwargs)

    monkeypatch.setattr(TemporaryStorage, "__init__", isolated_init)
