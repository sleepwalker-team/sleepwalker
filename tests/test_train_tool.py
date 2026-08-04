from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch
import yaml

from sleepwalker.datasets import ChannelConfig
from sleepwalker.training import files as training_files


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_train_tool():
    spec = importlib.util.spec_from_file_location(
        "sleepwalker_train_tool",
        REPO_ROOT / "tools" / "train.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def dataset_context(dataset):
    input_channels = dataset.get_input_channels()
    return {
        "classes": dataset.get_classes(),
        "input_channels": input_channels,
        "n_channels": len(input_channels),
        "ts_len": dataset.get_timeseries_len(),
        "sampling_frequency": dataset.sample_frequency,
    }


def build_trainer(train_tool, spec, context):
    spec = copy.deepcopy(spec)
    spec["optimizer"] = train_tool.build_factory(spec["optimizer"], "model")
    if spec.get("lr_scheduler") is not None:
        spec["lr_scheduler"] = train_tool.build_factory(
            spec["lr_scheduler"], "optimizer"
        )
    if "loss_function" in spec:
        spec["loss_function"] = train_tool.build_value(spec["loss_function"])
    return train_tool.build_component(spec, context)


def test_active_training_component_graphs_construct():
    train_tool = load_train_tool()
    configs = sorted((REPO_ROOT / "configs" / "train").rglob("*.yml"))
    assert configs

    for path in configs:
        config = train_tool.read_yaml(path)
        entries = config["data"] if isinstance(config["data"], list) else [config["data"]]
        datasets = [train_tool.build_dataset(entry) for entry in entries]
        context = dataset_context(datasets[0])
        train_tool.build_component(config["model"], context)
        build_trainer(train_tool, config["trainer"], context)


def test_recursive_multimodel_construction_uses_entry_channels():
    train_tool = load_train_tool()
    config = train_tool.read_yaml(
        REPO_ROOT / "configs" / "train" / "arousal_hsp_multimodel.yml"
    )
    dataset = train_tool.build_dataset(config["data"])
    model = train_tool.build_component(config["model"], dataset_context(dataset))

    assert model.input_channels == ["EEG", "EOG", "Chin EMG", "EKG"]
    assert model.model_entries[0]["model"].n_channels == 1
    assert model.model_entries[1]["model"].n_channels == 3


def test_channel_config_builds_direct_and_per_physical_normalizers():
    train_tool = load_train_tool()
    direct = train_tool.build_channel(
        {"logical_name": "eeg", "physical_names": ["EEG"], "unit": "uV"}
    )
    per_physical = train_tool.build_channel(
        {
            "logical_name": "eeg",
            "physical_names": ["C3-M2", "C4-M1"],
            "normalizer": {
                "C3-M2": {
                    "name": "sleepwalker.datasets.normalizer.EEGFilterNormalizer.EEGFilterNormalizer",
                    "fs": 100,
                },
                "C4-M1": None,
            },
        }
    )

    assert direct == ChannelConfig("eeg", ["EEG"], unit="uV")
    assert per_physical.normalizer_for("C3-M2").__class__.__name__ == "EEGFilterNormalizer"
    assert per_physical.normalizer_for("C4-M1") is None


def test_nested_fully_qualified_callable_is_resolved():
    train_tool = load_train_tool()
    value = train_tool.build_value(
        {
            "task": {
                "labels": ["event", "no event"],
                "loss_function": "torch.nn.functional.cross_entropy",
            }
        }
    )
    assert value["task"]["loss_function"] is torch.nn.functional.cross_entropy


def test_stacking_config_freezes_only_nested_experts():
    train_tool = load_train_tool()
    config = train_tool.read_yaml(
        REPO_ROOT / "configs" / "train" / "multilabel_stacking.yml"
    )
    dataset = train_tool.build_dataset(config["data"])
    model = train_tool.build_component(config["model"], dataset_context(dataset))

    assert all(
        not parameter.requires_grad
        for entry in model.model_entries
        for parameter in entry["model"].parameters()
    )
    assert all(parameter.requires_grad for parameter in model.heads.parameters())


def test_fraction_split_is_visible_and_disjoint():
    train_tool = load_train_tool()
    files = {
        "train": [f"patient-{idx}.edf" for idx in range(10)],
        "validation_fraction": 0.2,
        "test_fraction": 0.25,
    }
    split = train_tool.load_patient_split(files, dataset=object(), seed=17)

    assert {name: len(paths) for name, paths in split.items()} == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_precomputed_split_is_loaded_without_selection(tmp_path):
    train_tool = load_train_tool()
    expected = {
        "train": ["train.edf"],
        "validation": ["validation.edf"],
        "test": ["test.edf"],
    }
    split_path = tmp_path / "hsp_split.yml"
    split_path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    assert train_tool.load_patient_split(split_path.as_posix(), object(), 17) == expected


def test_dry_run_changes_training_not_just_validation(monkeypatch):
    train_tool = load_train_tool()
    config = train_tool.read_yaml(
        REPO_ROOT / "configs" / "train" / "desaturation_hsp.yml"
    )
    dataset = train_tool.build_dataset(config["data"][0])
    monkeypatch.setattr(
        train_tool,
        "initialize_datasets",
        lambda _config, _dry_run: ([dataset], [], [], dataset),
    )

    run_config = train_tool.build_run_config(config, dry_run=True)

    assert run_config.trainer.epochs == 1
    assert run_config.n_samples == 512
    assert run_config.n_samples_test == 256
    assert run_config.batch_size == 16
    assert run_config.num_workers_dataloader == 0
    assert run_config.use_mlflow is False


def test_file_selector_combines_channel_and_pap_filtering(monkeypatch):
    candidates = ["regular.edf", "pap.edf", "missing.edf"]
    signals = {
        "regular.edf": ["EEG", "Chest"],
        "pap.edf": ["EEG", "Chest", "Mask Pressure"],
        "missing.edf": ["EEG"],
    }
    reads = []
    monkeypatch.setattr(
        training_files,
        "get_edf_files_in_repo",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(
        training_files,
        "read_edf_meta",
        lambda path: reads.append(path) or {"signals": signals[path]},
    )
    dataset = type(
        "Dataset",
        (),
        {"channels": [ChannelConfig("EEG", ["EEG"]), ChannelConfig("Chest", ["Chest"])]},
    )()

    selected = training_files.select_edf_files(
        "/unused",
        dataset=dataset,
        require_channels=True,
        include_pap=False,
    )

    assert selected == ["regular.edf"]
    assert reads == candidates
