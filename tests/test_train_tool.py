from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import sleepwalker.datasets.HSP as hsp_dataset
import sleepwalker.cli.train as train_tool
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.NumpyDataset import NumpyDataset
from sleepwalker.training.callbacks import prepare_patient_events
from sleepwalker.training import files as training_files


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_train_tool():
    return train_tool


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


def test_read_yaml_parses_scientific_notation_without_decimal_point(tmp_path):
    train_tool = load_train_tool()
    path = tmp_path / "scientific.yml"
    path.write_text("small: 1e-5\nlarge: 2E+4\nnegative: -3e2\nduration: 30min\nquoted: '1e-5'\n", encoding="utf-8")

    config = train_tool.read_yaml(path)

    assert config == {"small": 1e-5, "large": 2e4, "negative": -3e2, "duration": "30min", "quoted": "1e-5"}


def test_cli_uses_explicit_train_resume_and_dry_commands():
    train_tool = load_train_tool()
    parser = train_tool.build_parser()

    train_args = parser.parse_args(["train", "config.yml", "--fold", "fold_2"])
    resume_args = parser.parse_args(["resume", "results/run/20/checkpoint.pt"])
    dry_args = parser.parse_args(["dry", "config.yml"])

    assert (train_args.command, train_args.config, train_args.fold) == ("train", "config.yml", "fold_2")
    assert (resume_args.command, resume_args.checkpoint) == ("resume", "results/run/20/checkpoint.pt")
    assert (dry_args.command, dry_args.config, dry_args.fold) == ("dry", "config.yml", None)


def test_resume_command_uses_checkpoint_run_config(monkeypatch):
    train_tool = load_train_tool()
    run_config = object()
    calls = []
    monkeypatch.setattr(train_tool, "runcfg_from_checkpoint", lambda path: calls.append(("load", path)) or run_config)
    monkeypatch.setattr(train_tool, "run", lambda current_config: calls.append(("run", current_config)))

    train_tool.main(["resume", "checkpoint.pt"])

    assert calls == [("load", "checkpoint.pt"), ("run", run_config)]


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
    split_path.write_text(yaml.safe_dump({"folds": {"holdout": expected}}), encoding="utf-8")

    assert train_tool.load_patient_split(split_path.as_posix(), object(), 17) == expected


def test_precomputed_cross_validation_fold_is_selected(tmp_path):
    train_tool = load_train_tool()
    expected = {
        "train": ["train.edf"],
        "validation": ["validation.edf"],
        "test": ["test.edf"],
    }
    split_path = tmp_path / "cross_validation.yml"
    split_path.write_text(yaml.safe_dump({"folds": {"fold_0": expected, "fold_1": {"train": ["other-train.edf"], "validation": ["other-validation.edf"], "test": ["other-test.edf"]}}}), encoding="utf-8")

    assert train_tool.load_patient_split(split_path.as_posix(), object(), 17, fold="fold_0") == expected


def test_configured_patient_filter_receives_existing_split(monkeypatch):
    train_tool = load_train_tool()
    dataset = object()
    calls = []

    def keep_matching(*, patients, dataset, marker):
        calls.append((list(patients), dataset, marker))
        return [path for path in patients if marker in path]

    monkeypatch.setattr(training_files, "keep_matching", keep_matching, raising=False)

    selected = train_tool.apply_patient_filter(
        {
            "name": "sleepwalker.training.files.keep_matching",
            "marker": "keep",
        },
        ["keep.edf", "drop.edf"],
        dataset,
    )

    assert selected == ["keep.edf"]
    assert calls == [(["keep.edf", "drop.edf"], dataset, "keep")]


def test_configured_patient_filter_receives_worker_count(monkeypatch):
    train_tool = load_train_tool()
    calls = []

    def keep_all(*, patients, num_workers):
        calls.append((list(patients), num_workers))
        return patients

    monkeypatch.setattr(training_files, "keep_all", keep_all, raising=False)

    selected = train_tool.apply_patient_filter(
        {"name": "sleepwalker.training.files.keep_all"},
        ["one.edf", "two.edf"],
        object(),
        num_workers=7,
    )

    assert selected == ["one.edf", "two.edf"]
    assert calls == [(["one.edf", "two.edf"], 7)]


def test_dataset_initialization_filters_each_precomputed_partition(monkeypatch):
    train_tool = load_train_tool()
    initialized = {}

    class DummyDataset:
        def initialize(self, patients, workers, strict):
            initialized[id(self)] = (list(patients), workers, strict)

    built = {}

    def build_dataset(_entry):
        dataset = DummyDataset()
        built.setdefault("datasets", []).append(dataset)
        return dataset

    monkeypatch.setattr(train_tool, "build_dataset", build_dataset)
    monkeypatch.setattr(
        train_tool,
        "load_patient_split",
        lambda *_args, **_kwargs: {
            "train": ["train-keep.edf", "train-drop.edf"],
            "validation": ["validation-keep.edf", "validation-drop.edf"],
            "test": ["test-keep.edf", "test-drop.edf"],
        },
    )
    monkeypatch.setattr(
        train_tool,
        "apply_patient_filter",
        lambda _spec, patients, _dataset, num_workers: [
            path for path in patients if "keep" in path
        ],
    )
    config = {
        "data": {
            "files": "split.yml",
            "patient_filter": {"name": "unused"},
            "num_workers": 3,
        }
    }

    train, validation, test = train_tool.initialize_datasets(config)

    assert initialized[id(train[0])] == (["train-keep.edf"], 3, False)
    assert initialized[id(validation[0])] == (["validation-keep.edf"], 3, False)
    assert initialized[id(test[0][1])] == (["test-keep.edf"], 3, False)


def write_numpy_cache(path, patients):
    path.mkdir()
    metadata = {"sample_frequency": 100, "resample_type": "nearest", "total_input": "30s", "stride": "30s", "classes": ["wake", "rem"], "input_channels": ["eeg"]}
    (path / "meta.json").write_text(json.dumps(metadata))
    np.save(path / "data.npy", np.zeros((len(patients), 3000, 1), dtype=np.float32))
    np.save(path / "target.npy", np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(patients), 1)))
    np.save(path / "patient.npy", np.array(patients))
    np.save(path / "time.npy", np.arange(len(patients), dtype=np.int64))
    return path


@pytest.mark.parametrize("role", ["train", "validation", "test"])
@pytest.mark.parametrize("in_memory", [False, True])
def test_numpy_partition_uses_constructor_loaded_cache(tmp_path, role, in_memory):
    cache = write_numpy_cache(tmp_path / "cache", ["patient-a.edf", "patient-a.edf", "patient-b.edf"])
    config = {"data": {"name": "sleepwalker.datasets.NumpyDataset.NumpyDataset", "cache_path": str(cache), "in_memory": in_memory, "files": {role: ["patient-a.edf", "patient-b.edf"]}, "label": "cached"}}

    train, validation, test = train_tool.initialize_datasets(config)

    partitions = {"train": train, "validation": validation, "test": [dataset for _, dataset in test]}
    assert {name: len(datasets) for name, datasets in partitions.items()} == {name: int(name == role) for name in partitions}
    dataset = partitions[role][0]
    assert isinstance(dataset, NumpyDataset)
    assert dataset.initialized
    assert len(dataset) == 3
    assert dataset[2]["patient"] == "patient-b.edf"
    assert isinstance(dataset.data_shards[0], np.memmap) == (not in_memory)
    if role == "test":
        assert test[0][0] == "cached"


def test_numpy_cache_cannot_bypass_patient_partition(tmp_path):
    cache = write_numpy_cache(tmp_path / "cache", ["training.edf", "held-out.edf"])
    config = {"data": {"name": "sleepwalker.datasets.NumpyDataset.NumpyDataset", "cache_path": str(cache), "files": {"train": ["training.edf"]}}}

    with pytest.raises(ValueError, match="outside the train partition"):
        train_tool.initialize_datasets(config)


def test_dry_run_changes_training_not_just_validation(monkeypatch):
    train_tool = load_train_tool()
    config = train_tool.read_yaml(
        REPO_ROOT / "configs" / "train" / "desaturation_hsp.yml"
    )
    dataset = train_tool.build_dataset(config["data"][0])
    monkeypatch.setattr(
        train_tool,
        "initialize_datasets",
        lambda _config, dry_run, fold=None, model=None: ([dataset], [], []),
    )

    run_config = train_tool.runcfg_from_dict(config, dry_run=True, fold="fold_2")

    assert run_config.trainer.epochs == 1
    assert run_config.trainer.eval_every == 1
    assert run_config.experiment_name.endswith("-fold_2-dry-run")
    assert run_config.tags["fold"] == "fold_2"
    assert run_config.meta_data["fold"] == "fold_2"
    assert run_config.n_samples == 512
    assert run_config.n_samples_test == 256
    assert run_config.batch_size == 16
    assert run_config.num_workers_dataloader == 0
    assert run_config.use_mlflow is False


def test_runcfg_from_checkpoint_reuses_serialized_trainer_and_model(tmp_path, monkeypatch):
    train_tool = load_train_tool()
    config = train_tool.read_yaml(REPO_ROOT / "configs" / "train" / "desaturation_hsp.yml")
    dataset = train_tool.build_dataset(config["data"][0])
    model = object()
    trainer = object()
    checkpoint_path = tmp_path / "run" / "10" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    (checkpoint_path.parent.parent / "hparams.yml").write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(train_tool.BaseTrainer, "load_checkpoint", lambda path: (trainer, model))
    monkeypatch.setattr(train_tool, "initialize_datasets", lambda current_config, dry_run=False, fold=None: ([dataset], [], []))
    monkeypatch.setattr(train_tool, "build_component", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resume must not build a model or trainer")))

    run_config = train_tool.runcfg_from_checkpoint(checkpoint_path)

    assert run_config.trainer is trainer
    assert run_config.model is model
    assert run_config.train_datasets == [dataset]


def test_runcfg_from_checkpoint_requires_run_configuration(tmp_path, monkeypatch):
    train_tool = load_train_tool()
    checkpoint_path = tmp_path / "run" / "10" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    monkeypatch.setattr(train_tool.BaseTrainer, "load_checkpoint", lambda path: (object(), object()))

    with pytest.raises(FileNotFoundError, match="hparams.yml"):
        train_tool.runcfg_from_checkpoint(checkpoint_path)


def test_patient_filter_combines_required_and_excluded_channels(monkeypatch):
    candidates = ["regular.edf", "pap.edf", "missing.edf"]
    signals = {
        "regular.edf": ["EEG", "Chest"],
        "pap.edf": ["EEG", "Chest", "Mask Pressure"],
        "missing.edf": ["EEG"],
    }
    reads = []
    monkeypatch.setattr(
        training_files,
        "read_edf_meta",
        lambda path: reads.append(path) or {"signals": signals[path], "duration_s": 3600},
    )
    dataset = type(
        "Dataset",
        (),
        {"channels": [ChannelConfig("EEG", ["EEG"]), ChannelConfig("Chest", ["Chest"])]},
    )()

    selected = training_files.filter_edf_files(
        candidates,
        dataset=dataset,
        excluded_channels=["Mask Pressure"],
    )

    assert selected == ["regular.edf"]
    assert reads == candidates


def test_patient_filters_separate_edf_headers_and_hsp_annotations(monkeypatch):
    train_tool = load_train_tool()
    patients = ["usable.edf", "short.edf", "missing.edf", "awake.edf"]
    metadata = {
        "usable.edf": {"signals": ["EEG", "SpO2"], "duration_s": 3600},
        "short.edf": {"signals": ["EEG", "SpO2"], "duration_s": 1200},
        "missing.edf": {"signals": ["EEG"], "duration_s": 3600},
        "awake.edf": {"signals": ["EEG", "SpO2"], "duration_s": 3600},
    }
    labels = {
        "usable.edf.csv": ["stage - n2"],
        "awake.edf.csv": ["stage - w"],
    }
    monkeypatch.setattr(training_files, "read_edf_meta", lambda path: metadata[path])
    monkeypatch.setattr("sleepwalker.datasets.HSP.get_hsp_annotation_path", lambda path: f"{path}.csv")
    monkeypatch.setattr("sleepwalker.datasets.HSP.pd.read_csv", lambda path, **_kwargs: pd.DataFrame({"event": labels[path], "duration": [30.0] * len(labels[path])}))
    dataset = type(
        "Dataset",
        (),
        {
            "channels": [
                ChannelConfig("EEG", ["EEG"]),
                ChannelConfig("SpO2", ["SaO2", "SpO2"]),
            ]
        },
    )()

    selected = train_tool.apply_patient_filter(
        [
            {"name": "sleepwalker.training.files.filter_edf_files", "min_duration": "30min"},
            {"name": "sleepwalker.datasets.HSP.filter_hsp_annotation_labels", "required_labels": ["n1", "n2", "n3", "rem"]},
        ],
        patients,
        dataset,
    )

    assert selected == ["usable.edf"]


def test_hsp_annotation_filter_uses_multiple_workers(tmp_path):
    patients = []
    annotations = {
        "sub-one_task-psg_eeg.edf": "stage - n2",
        "sub-two_task-psg_eeg.edf": "arousal",
    }
    for name, label in annotations.items():
        edf_path = tmp_path / name
        annotation_path = edf_path.with_name(name.replace("eeg", "annotations").replace(".edf", ".csv"))
        annotation_path.write_text(f"event,time,duration\n{label},00:00:00,30\n", encoding="utf-8")
        patients.append(str(edf_path))

    selected = hsp_dataset.filter_hsp_annotation_labels(
        patients,
        required_labels=["n1", "n2", "n3", "rem"],
        num_workers=2,
    )

    assert selected == [patients[0]]


def test_hsp_sane_labels_only_map_labels():
    annotations = pd.DataFrame({
        "Label": ["rem", "rem", "sleep_stage_r", "stage - n2", "arousal"],
        "Duration": [0.004, 30.0, 0.004, 30.0, 0.004],
    })

    mapped = hsp_dataset.map_hsp_sane_labels(annotations)

    assert mapped["Label"].tolist() == ["rem", "rem", "rem", "n2", "arousal"]


def test_prepare_patient_events_filters_mapped_labels_before_optional_trimming():
    start = pd.Timestamp("2024-01-01")
    annotations = pd.DataFrame({
        "Label": ["sleep", "arousal", "sleep", "arousal"],
        "Starttime": [start, start + pd.Timedelta(seconds=5), start + pd.Timedelta(seconds=10), start + pd.Timedelta(seconds=30)],
        "Endtime": [start + pd.Timedelta(seconds=5), start + pd.Timedelta(seconds=6), start + pd.Timedelta(seconds=30), start + pd.Timedelta(seconds=31)],
    })

    filtered, _ = prepare_patient_events(annotations, None, min_seconds=15, labels=["sleep"])
    trimmed, _ = prepare_patient_events(annotations, None, min_seconds=15, labels=["sleep"], keep_events=["sleep"])

    assert filtered["Label"].tolist() == ["arousal", "sleep", "arousal"]
    assert trimmed["Label"].tolist() == ["sleep"]


def test_hsp_annotation_filter_only_checks_mapped_label_presence(tmp_path):
    short_edf = tmp_path / "short_task-psg_eeg.edf"
    valid_edf = tmp_path / "valid_task-psg_eeg.edf"
    short_edf.with_name("short_task-psg_annotations.csv").write_text("event,time,duration\nREM,00:00:00,0.004\n", encoding="utf-8")
    valid_edf.with_name("valid_task-psg_annotations.csv").write_text("event,time,duration\nREM,00:00:00,30\n", encoding="utf-8")

    selected = hsp_dataset.filter_hsp_annotation_labels([str(short_edf), str(valid_edf)], required_labels=["rem"], num_workers=1)

    assert selected == [str(short_edf), str(valid_edf)]
