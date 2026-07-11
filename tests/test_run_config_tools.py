from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from generate_haml_from_yaml import build_haml_config
from run_config import build_command


def test_run_config_builds_cli_command_from_mapping():
    cfg = {
        "script": "train_arousal_hsp.py",
        "args": {
            "config": "configs/experts/arousal_hsp_dev.yml",
            "id": "smoke",
            "dry": True,
            "channels": ["eeg", "eog"],
            "no_mlflow": True,
        },
    }

    command = build_command(cfg, "python")

    assert command == [
        "python",
        "train_arousal_hsp.py",
        "--config",
        "configs/experts/arousal_hsp_dev.yml",
        "--id",
        "smoke",
        "--dry",
        "--channels",
        "eeg",
        "eog",
        "--no_mlflow",
    ]


def test_generate_haml_from_yaml_replaces_sweep_paths():
    cfg = {
        "script": "train_arousal_hsp.py",
        "args": {"id": "base", "config": "cfg.yml"},
        "sweep": {
            "args.id": ["run-a", "run-b"],
            "args.config": ["a.yml", "b.yml"],
        },
    }

    haml_cfg = build_haml_config(cfg)

    assert haml_cfg["script"] == "train_arousal_hsp.py"
    assert haml_cfg["args"]["id"] == '{{ "run-a" || "run-b" }}'
    assert haml_cfg["args"]["config"] == '{{ "a.yml" || "b.yml" }}'
