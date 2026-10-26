import json
from pathlib import Path
import sys
import time

import pytest

from sleepwalker.server.services import subprocess_inference
from sleepwalker.server.services.job_executor import JobCancelledError, JobExecutionError
from sleepwalker.server.services.model_registry import read_model_manifest
from sleepwalker.server.services.subprocess_inference import SubprocessInferenceClient


@pytest.fixture
def stub_worker(tmp_path, monkeypatch):
    def create(body):
        path = tmp_path / "stub_worker.py"
        path.write_text(
            "import json, os, subprocess, sys, time\nfrom pathlib import Path\n"
            "request_path = Path(sys.argv[1])\nroot = request_path.parent\n"
            "request = json.loads(request_path.read_text())\n" + body,
            encoding="utf-8",
        )
        monkeypatch.setattr(subprocess_inference, "WORKER_PATH", path)
    return create


@pytest.fixture
def worker_temporary_paths(tmp_path, monkeypatch):
    paths = []
    original = subprocess_inference.TemporaryDirectory
    def create(**kwargs):
        directory = original(dir=tmp_path, **kwargs)
        paths.append(Path(directory.name))
        return directory
    monkeypatch.setattr(subprocess_inference, "TemporaryDirectory", create)
    return paths


def test_client_runs_without_api_pythonpath_and_cleans_transport_files(make_model_package, tmp_path, stub_worker, worker_temporary_paths, monkeypatch):
    path, _ = make_model_package()
    stub_worker(
        "assert 'PYTHONPATH' not in os.environ\n"
        "assert 'PYTHONHOME' not in os.environ\n"
        "assert 'LD_LIBRARY_PATH' not in os.environ\n"
        "assert 'LD_PRELOAD' not in os.environ\n"
        "assert request['task'] == 'sleep' and request['batch_size'] == 4\n"
        "assert request['edf_path'].endswith('recording with spaces.edf')\n"
        "(root / 'predictions.csv').write_text('time,task,valid,prob__wake,prob__n1\\n2025-01-01 00:05:00,sleep,True,0.9,0.1\\n')\n"
        "(root / 'response.json').write_text(json.dumps({'version': 1, 'ok': True}))\n"
    )
    monkeypatch.setenv("PYTHONPATH", "/invalid/api/import/path")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/invalid/system/torch/lib")
    monkeypatch.setenv("LD_PRELOAD", "/invalid/injected-library.so")
    client = SubprocessInferenceClient(Path(sys.executable))
    frame = client.predict(read_model_manifest(path), tmp_path / "recording with spaces.edf", device="cpu", batch_size=4, check_cancelled=lambda: None)
    assert frame["task"].tolist() == ["sleep"]
    assert frame["prob__wake"].tolist() == [0.9]
    assert len(worker_temporary_paths) == 1
    assert all(not path.exists() for path in worker_temporary_paths)


@pytest.mark.parametrize("outcome", ["typed_error", "crash", "invalid_json", "invalid_csv"])
def test_client_hides_worker_errors_and_cleans_files(make_model_package, tmp_path, stub_worker, worker_temporary_paths, outcome):
    path, _ = make_model_package()
    code = "print('private/patient/path and internal traceback', file=sys.stderr)\n"
    if outcome == "typed_error":
        code += "(root / 'response.json').write_text(json.dumps({'version': 1, 'ok': False, 'code': 'MODEL_INCOMPATIBLE'}))\nsys.exit(1)\n"
    elif outcome == "crash":
        code += "sys.exit(2)\n"
    elif outcome == "invalid_json":
        code += "(root / 'response.json').write_text('broken')\n"
    else:
        code += "(root / 'response.json').write_text(json.dumps({'version': 1, 'ok': True}))\n(root / 'predictions.csv').write_text('')\n"
    stub_worker(code)
    with pytest.raises(JobExecutionError) as error:
        SubprocessInferenceClient(Path(sys.executable)).predict(read_model_manifest(path), tmp_path / "recording.edf", device="cpu", batch_size=1, check_cancelled=lambda: None)
    assert error.value.code == ("MODEL_INCOMPATIBLE" if outcome == "typed_error" else "COMPUTE_UNAVAILABLE")
    assert "private/patient" not in str(error.value)
    assert all(not path.exists() for path in worker_temporary_paths)


def test_client_reports_missing_inference_python(make_model_package, tmp_path, worker_temporary_paths):
    path, _ = make_model_package()
    with pytest.raises(JobExecutionError) as error:
        SubprocessInferenceClient(tmp_path / "missing-python").predict(read_model_manifest(path), tmp_path / "recording.edf", device="cpu", batch_size=1, check_cancelled=lambda: None)
    assert error.value.code == "COMPUTE_UNAVAILABLE"
    assert all(not path.exists() for path in worker_temporary_paths)


def test_client_preserves_virtual_environment_executable_symlink(tmp_path):
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    assert SubprocessInferenceClient(executable).python_executable == executable


def test_cancellation_stops_worker_and_its_child_process(make_model_package, tmp_path, stub_worker, worker_temporary_paths):
    path, _ = make_model_package()
    marker = tmp_path / "processes.json"
    heartbeat = tmp_path / "child-heartbeat"
    child_code = f"import time\nfrom pathlib import Path\npath = Path({str(heartbeat)!r})\nwhile True:\n path.write_text(str(time.monotonic_ns()))\n time.sleep(0.01)\n"
    stub_worker(
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()
    def check_cancelled():
        if marker.exists() and heartbeat.exists():
            raise JobCancelledError
        assert time.monotonic() - started < 5, "Stub worker did not start"
    with pytest.raises(JobCancelledError):
        SubprocessInferenceClient(Path(sys.executable)).predict(read_model_manifest(path), tmp_path / "recording.edf", device="cpu", batch_size=1, check_cancelled=check_cancelled)
    # Do not inspect /proc/<pid>: a sandbox may expose host /proc but use a
    # different PID namespace. A stopped heartbeat proves the child stopped.
    time.sleep(0.1)
    last_heartbeat = heartbeat.read_text()
    time.sleep(0.15)
    assert heartbeat.read_text() == last_heartbeat
    assert all(not path.exists() for path in worker_temporary_paths)
