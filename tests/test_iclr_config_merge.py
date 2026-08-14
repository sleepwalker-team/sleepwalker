from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_merge_tool():
    spec = importlib.util.spec_from_file_location("iclr_config_merge", REPO_ROOT / "iclr2026" / "configs" / "generate_configs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_merge_resolves_model_channels_from_dataset_catalog():
    merge_tool = load_merge_tool()
    dataset = {
        "data": {
            "dataset": {
                "channel_catalog": {
                    "chin_emg": {"physical_names": ["1-2", "1-F"]},
                    "left_leg_emg": {"physical_names": ["Left Leg"]},
                },
            },
        },
        "test": {"batch_size": 64},
    }
    model = {
        "data": {
            "dataset": {
                "channels": [
                    {"source": "chin_emg", "logical_name": "EMG_Chin", "normalizer": None},
                    {"source": "left_leg_emg", "logical_name": "EMG_LLeg", "normalizer": None},
                ],
            },
        },
        "test": {"batch_size": 32},
    }

    resolved = merge_tool.resolve_channel_catalogs(merge_tool.merge(dataset, model))

    assert resolved["data"]["dataset"]["channels"] == [
        {"physical_names": ["1-2", "1-F"], "logical_name": "EMG_Chin", "normalizer": None},
        {"physical_names": ["Left Leg"], "logical_name": "EMG_LLeg", "normalizer": None},
    ]
    assert "channel_catalog" not in resolved["data"]["dataset"]
    assert resolved["test"]["batch_size"] == 32


def test_merge_rejects_unknown_channel_source():
    merge_tool = load_merge_tool()
    config = {
        "data": {
            "channel_catalog": {"eeg": {"physical_names": ["C3-M2"]}},
            "channels": [{"source": "missing", "logical_name": "EEG"}],
        },
    }

    with pytest.raises(ValueError, match="unknown channel source 'missing'"):
        merge_tool.resolve_channel_catalogs(config)


def test_single_task_train_configs_have_three_sources_and_an_optimizer():
    paths = sorted((REPO_ROOT / "iclr2026" / "configs" / "train" / "generated").glob("*.yml"))

    for path in paths:
        if path.stem == "multitask_sleepwalker":
            continue
        provenance, contents = path.read_text(encoding="utf-8").split("\n", 1)
        config = yaml.safe_load(contents)
        assert provenance.count("iclr2026/configs/train/") == 3
        assert config["trainer"]["optimizer"]["name"].startswith("torch.optim.")


def test_generated_train_configs_use_fixed_onecycle_policy():
    paths = sorted((REPO_ROOT / "iclr2026" / "configs" / "train" / "generated").glob("*.yml"))

    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8").split("\n", 1)[1])
        trainer = config["trainer"]
        is_sleepwalker = "sleepwalker" in path.stem
        assert trainer["epochs"] == (100 if is_sleepwalker else 25)
        assert trainer["lr_scheduler"]["name"] == "torch.optim.lr_scheduler.OneCycleLR"
        assert trainer["save_every"] == 10
        assert trainer["return_best"] is False
        assert "early_stopping" not in trainer
        assert config["run"]["n_samples"] == 250000
        assert "patients_per_epoch" not in config["run"]
        assert "patient_group_size" not in config["run"]


def test_mirror_config_covers_every_paper_model_channel_source():
    hsp = yaml.safe_load((REPO_ROOT / "iclr2026" / "configs" / "train" / "hsp.yml").read_text(encoding="utf-8"))
    catalog = hsp["data"]["channel_catalog"]
    mirror = yaml.safe_load((REPO_ROOT / "iclr2026" / "configs" / "mirror.yaml").read_text(encoding="utf-8"))
    mirrored_physical_names = {
        physical_name
        for channel in mirror["data"]["channels"]
        for physical_name in channel["physical_names"]
    }

    for path in sorted((REPO_ROOT / "iclr2026" / "configs" / "train" / "models").glob("*.yml")):
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        for channel in model["data"]["channels"]:
            missing = set(catalog[channel["source"]]["physical_names"]) - mirrored_physical_names
            assert not missing, f"{path.name} source {channel['source']} is missing mirror channels {sorted(missing)}"


def test_sleepwalker_train_and_test_channel_fragments_match():
    train_root = REPO_ROOT / "iclr2026" / "configs" / "train" / "models"
    test_root = REPO_ROOT / "iclr2026" / "configs" / "test" / "models"

    for train_path in sorted(train_root.glob("sleepwalker_*.yml")):
        train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
        test = yaml.safe_load((test_root / train_path.name).read_text(encoding="utf-8"))
        assert test["data"]["dataset"]["channels"] == train["data"]["channels"]


def test_sleepwalker_task_configs_use_portable_channel_sets():
    root = REPO_ROOT / "iclr2026" / "configs" / "train" / "models"
    expected = {
        "arousal": ["EEG", "EOG", "Chin EMG", "ECG"],
        "breathing": ["Abdomen", "Chest", "Airflow", "SpO2"],
        "desaturation": ["SpO2"],
    }

    for task, logical_names in expected.items():
        config = yaml.safe_load((root / f"sleepwalker_{task}.yml").read_text(encoding="utf-8"))
        assert [channel["logical_name"] for channel in config["data"]["channels"]] == logical_names


def test_saturation_ranges_match_post_normalization_channel_contracts():
    task_root = REPO_ROOT / "iclr2026" / "configs" / "train" / "tasks"
    for task in ["breathing", "desaturation"]:
        config = yaml.safe_load((task_root / f"{task}.yml").read_text(encoding="utf-8"))
        assert config["data"]["prepare_sample"]["valid_ranges"] == {"SpO2": [-5, 5], "SaO2": [50, 100]}

    multitask = yaml.safe_load((REPO_ROOT / "iclr2026" / "configs" / "train" / "models" / "sleepwalker_multitask.yml").read_text(encoding="utf-8"))
    assert multitask["data"]["prepare_sample"]["valid_ranges"] == {"SpO2": [-5, 5]}


def test_foundation_train_and_test_channel_fragments_match():
    train_root = REPO_ROOT / "iclr2026" / "configs" / "train" / "models"
    test_root = REPO_ROOT / "iclr2026" / "configs" / "test" / "models"

    for name in ["sleepfm.yml", "sleepgpt.yml", "osf.yml"]:
        train = yaml.safe_load((train_root / name).read_text(encoding="utf-8"))
        test = yaml.safe_load((test_root / name).read_text(encoding="utf-8"))
        assert test["data"]["dataset"]["channels"] == train["data"]["channels"]


def test_sleepgpt_uses_the_six_channel_physio_subset():
    config = yaml.safe_load((REPO_ROOT / "iclr2026" / "configs" / "train" / "models" / "sleepgpt.yml").read_text(encoding="utf-8"))

    assert [channel["logical_name"] for channel in config["data"]["channels"]] == ["C3", "C4", "EMG", "EOG", "F3", "O1"]
    assert {channel["unit"] for channel in config["data"]["channels"]} == {"uV"}


def test_osf_uses_recording_z_normalization_for_training_and_testing():
    train = yaml.safe_load((REPO_ROOT / "iclr2026" / "configs" / "train" / "models" / "osf.yml").read_text(encoding="utf-8"))
    test = yaml.safe_load((REPO_ROOT / "iclr2026" / "configs" / "test" / "models" / "osf.yml").read_text(encoding="utf-8"))

    assert train["data"]["z_normalize"] is True
    assert test["data"]["dataset"]["z_normalize"] is True
