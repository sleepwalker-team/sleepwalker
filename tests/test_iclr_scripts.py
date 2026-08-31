from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_run_script():
    spec = importlib.util.spec_from_file_location("iclr2026_run", REPO_ROOT / "iclr2026" / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shell_expansions_cover_all_expert_configs():
    train_configs = sorted((REPO_ROOT / "iclr2026" / "configs" / "experts" / "train").glob("*.yml"))
    test_configs = sorted((REPO_ROOT / "iclr2026" / "configs" / "experts" / "test").glob("*.yml"))

    assert len(train_configs) == 13
    assert len(test_configs) == 13


def test_run_all_covers_configs_once_and_orders_model_families():
    script = (REPO_ROOT / "iclr2026" / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    references = re.findall(r"^  (iclr2026/configs/(?:experts/(?:train|test)|main)/[^ ]+\.yml)$", script, re.MULTILINE)
    expected = {
        *(str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "iclr2026" / "configs" / "experts" / "train").glob("*.yml")),
        *(str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "iclr2026" / "configs" / "experts" / "test").glob("*.yml")),
        *(str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "iclr2026" / "configs" / "main").glob("*.yml") if path.name != "test.yml"),
    }

    assert len(references) == len(set(references))
    assert set(references) == expected
    assert script.index("run_family sleepwalker") < script.index("run_family osf") < script.index("run_family sleepfm") < script.index('echo "[final]')


def test_entrypoint_arguments_are_forwarded_after_separator():
    run_script = load_run_script()

    runner, entrypoint = run_script.split_arguments(["tools/train.py", "one.yml", "two.yml", "--", "dry"])

    assert runner == ["tools/train.py", "one.yml", "two.yml"]
    assert entrypoint == ["dry"]


def test_gpu_and_session_names_are_validated():
    run_script = load_run_script()

    assert run_script.parse_gpus("0, 1,2") == ["0", "1", "2"]
    assert run_script.session_name("paper run", 0, Path("sleep_sleepwalker.yml")) == "iclr26-paper-run-01-sleep-sleepwalker"
    with pytest.raises(ValueError, match="without duplicates"):
        run_script.parse_gpus("0,0")


def test_queue_waits_until_tmux_reports_the_dead_pane_status(monkeypatch):
    run_script = load_run_script()
    responses = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=0, stdout="1\t\n")]
    monkeypatch.setattr(run_script.subprocess, "run", lambda *args, **kwargs: responses.pop(0))

    assert run_script.pane_state("session") == (False, None)


def test_queue_builds_entrypoint_command_without_reading_configs(monkeypatch, tmp_path):
    run_script = load_run_script()
    entrypoint = tmp_path / "entrypoint.py"
    config = tmp_path / "config.yml"
    python = tmp_path / "python"
    launched = {}

    def pane_state(session):
        return (True, 0) if session in launched else None

    def launch_job(session, gpu, command):
        launched[session] = (gpu, command)

    monkeypatch.setattr(run_script, "pane_state", pane_state)
    monkeypatch.setattr(run_script, "launch_job", launch_job)
    monkeypatch.setattr(run_script.time, "sleep", lambda _seconds: None)

    sessions = run_script.run_queue(entrypoint, [config], ["dry"], python, ["2"], "test", 0.01)

    assert launched[sessions[0]] == ("2", [str(python), str(entrypoint), "dry", str(config)])
