import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


@pytest.fixture
def worker_module(monkeypatch):
    # The worker imports the separately installed framework. Stub only those
    # imports here; exercise its real error handling without loading any model.
    deployment = ModuleType("sleepwalker.deployment")
    deployment.load_packaged_model = lambda *args, **kwargs: None
    cache = ModuleType("sleepwalker.datasets.EDFCache")
    cache.EDFCache = object
    monkeypatch.setitem(sys.modules, deployment.__name__, deployment)
    monkeypatch.setitem(sys.modules, cache.__name__, cache)
    path = Path(__file__).parents[1] / "sleepwalker/server/inference_worker.py"
    spec = importlib.util.spec_from_file_location("inference_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "error,code",
    [
        (MemoryError("private allocator details"), "RESOURCE_EXHAUSTED"),
        (torch.OutOfMemoryError("private CUDA allocator details"), "RESOURCE_EXHAUSTED"),
        (RuntimeError("out of memory is only untrusted error text"), "INFERENCE_FAILED"),
        (RuntimeError("private framework failure"), "INFERENCE_FAILED"),
        (ValueError("private model contract"), "MODEL_INCOMPATIBLE"),
        (OSError("private disk error"), "MODEL_INCOMPATIBLE"),
    ],
)
def test_worker_returns_typed_errors_without_private_details(worker_module, tmp_path, monkeypatch, error, code):
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inference_worker.py", str(request_path)])

    def fail(_request, _output):
        raise error

    monkeypatch.setattr(worker_module, "predict", fail)
    with pytest.raises(SystemExit) as exit_error:
        worker_module.main()

    assert exit_error.value.code == 1
    assert json.loads((tmp_path / "response.json").read_text()) == {"version": 1, "ok": False, "code": code}


def test_worker_returns_success(worker_module, tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inference_worker.py", str(request_path)])
    monkeypatch.setattr(worker_module, "predict", lambda _request, _output: None)

    worker_module.main()

    assert json.loads((tmp_path / "response.json").read_text()) == {"version": 1, "ok": True}
