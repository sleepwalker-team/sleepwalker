from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
import yaml

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.Stages import Stages
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel
from sleepwalker import prediction_transforms
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel
from iclr2026.pipeline import pair_datasets
from sleepwalker.models.ModelGraphClassifier import GraphNode, ModelGraphClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]


class MeanClassifier(BaseModel, ClassifierModel):
    def __init__(self, n_classes: int, ts_len: int = 300):
        super().__init__()
        self.linear = torch.nn.Linear(1, n_classes)
        self.ts_len = int(ts_len)

    def compute(self, x):
        return self.linear(x.mean(dim=1)).unsqueeze(1)

    def input_spec(self):
        return (1, self.ts_len, 1), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": 1}


def load_test_tool():
    spec = importlib.util.spec_from_file_location("sleepwalker_evaluate_tool", REPO_ROOT / "tools" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_package():
    dataset = UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["source-eeg"], unit="uV")],
        sample_frequency=100,
        total_input="60s",
        stride="5s",
    )
    return PackagedModel(name="package", task="event", model=MeanClassifier(2, ts_len=6000), dataset=dataset, classification_contract={"type": "single-head-multiclass", "classes": ["event", "no event"], "sequence_len": 2, "target_resolution": "20s"})


def write_unsplit_manifest(path: Path, files: list[str]) -> Path:
    path.write_text(yaml.safe_dump({"files": files}), encoding="utf-8")
    return path


def test_paper_evaluation_configs_are_valid():
    test_tool = load_test_tool()
    paths = sorted((REPO_ROOT / "iclr2026" / "configs" / "experts" / "test").glob("*.yml"))
    tasks = {"sleep", "arousal", "breathing", "desaturation"}
    expected = {
        *(f"{task}_{model}" for task in tasks for model in ("sleepwalker", "osf")),
        *(f"{task}_sleepfm" for task in tasks),
    }

    assert {path.stem for path in paths} == expected
    for path in paths:
        config = test_tool.read_config(path)
        assert isinstance(config["data"], dict)
        assert config["data"]["label"] == "HSP-test"
        assert "channel_catalog" not in str(config)
        assert "source" not in str(config)
        assert test_tool.build_analyses(config["analyses"])


def test_dataset_inherits_package_geometry_and_accepts_regular_channels(tmp_path):
    test_tool = load_test_tool()
    package = make_package()
    manifest = write_unsplit_manifest(tmp_path / "files.yml", ["patient.edf"])
    entry = {
        "label": "external",
        "files": str(manifest),
        "dataset": {
            "name": "sleepwalker.datasets.Stages.Stages",
            "event_mapping": {"event": "event", "sleep": "sleep"},
            "channels": [{"logical_name": "EEG", "physical_names": ["external-eeg"], "normalizer": None, "unit": "uV"}],
        },
    }

    dataset, patients, selected_fold = test_tool.prepare_dataset(package, entry)

    assert isinstance(dataset, Stages)
    assert dataset.sample_frequency == 100
    assert dataset.total_input == pd.Timedelta("60s")
    assert dataset.stride == pd.Timedelta("5s")
    assert dataset.channels == [ChannelConfig("EEG", ["external-eeg"], normalizer=None, unit="uV")]
    assert dataset.rejection_strategy == "none"
    assert patients == ["patient.edf"]
    assert selected_fold is None


def test_package_fold_selects_matching_manifest_fold(tmp_path):
    test_tool = load_test_tool()
    package = make_package()
    package.config = {"fold": "fold_1"}
    manifest = {
        "folds": {
            "fold_0": {"train": ["a.edf"], "validation": ["b.edf"], "test": ["c.edf"]},
            "fold_1": {"train": ["b.edf"], "validation": ["c.edf"], "test": ["a.edf"]},
        },
    }
    path = tmp_path / "split.yml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    files, selected_fold = test_tool.resolve_entry_files(package, {"label": "internal", "split": {"file": str(path)}})

    assert files == ["a.edf"]
    assert selected_fold == "fold_1"


def test_annotation_and_patient_filters_compose():
    test_tool = load_test_tool()
    frame = pd.DataFrame({
        "time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "target": [0, 1],
        "prob__event": [0.9, 0.1],
        "prob__no event": [0.1, 0.9],
        "annotation__sleep": [0.75, 0.25],
    })
    pipeline = [
        lambda current, patient, signals: prediction_transforms.filter_patient_signals(current, patient, signals, include_any=["Mask Pressure"]),
        lambda current, patient, signals: prediction_transforms.filter_annotation(current, patient, signals, labels=["sleep"], minimum_coverage=0.5),
    ]

    selected = prediction_transforms.apply_pipeline(frame, pipeline, patient="patient.edf", signals=["EEG", "Mask Pressure"])
    excluded = prediction_transforms.apply_pipeline(frame, pipeline, patient="patient.edf", signals=["EEG"])

    assert selected["target"].tolist() == [0]
    assert excluded.empty


def test_annotation_filter_uses_union_of_requested_labels():
    test_tool = load_test_tool()
    frame = pd.DataFrame({
        "annotation__n1": [0.3, 0.1],
        "annotation__n2": [0.3, 0.2],
    })

    selected = prediction_transforms.filter_annotation(frame, "patient.edf", [], labels=["n1", "n2"], minimum_coverage=0.5)

    assert selected.index.tolist() == [0]


def test_overlap_resolution_smoothing_and_metrics():
    test_tool = load_test_tool()
    frame = pd.DataFrame({
        "time": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
        "target": [0, 0, 1],
        "prob__negative": [0.8, 0.6, 0.2],
        "prob__positive": [0.2, 0.4, 0.8],
        "annotation__sleep": [1.0, 1.0, 1.0],
    })

    resolved = prediction_transforms.resolve_overlaps(frame, "patient.edf", [], method="mean_probability")
    smoothed = prediction_transforms.smooth_probabilities(resolved, "patient.edf", [], window=1)
    metrics = test_tool.patient_classification_metrics(smoothed, ["negative", "positive"])

    assert len(resolved) == 2
    assert resolved.iloc[0]["prob__negative"] == pytest.approx(0.7)
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["per_class"]["positive"]["sensitivity"] == 1.0


def test_patient_confusion_matrices_can_be_aggregated_directly():
    classes = ["negative", "positive"]
    frame = pd.DataFrame({
        "target": [0, 0, 1, 1],
        "prob__negative": [0.9, 0.8, 0.2, 0.1],
        "prob__positive": [0.1, 0.2, 0.8, 0.9],
    })
    patient_metrics = load_test_tool().patient_classification_metrics(frame, classes)
    aggregate = load_test_tool().confusion_metrics(patient_metrics["confusion_matrix"], classes)

    assert patient_metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert aggregate["accuracy"] == 1.0
    assert "auroc_macro" not in aggregate


def test_empty_confusion_matrix_reports_zero_windows():
    metrics = load_test_tool().confusion_metrics([[0, 0], [0, 0]], ["negative", "positive"])

    assert metrics["n_windows"] == 0
    assert metrics["cohen_kappa"] == 0.0


def test_cli_overwrite_updates_test_config(monkeypatch):
    test_tool = load_test_tool()
    config = {"test": {"package": "package", "overwrite": False}}
    calls = []
    monkeypatch.setattr(test_tool, "read_config", lambda path: config)
    monkeypatch.setattr(test_tool, "execute", lambda package, parsed: calls.append((package, parsed)))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--overwrite", "config.yml"])

    test_tool.main()

    assert config["test"]["overwrite"] is True
    assert calls == [("package", config)]


def test_packaged_model_evaluates_to_jsonl(tmp_path, monkeypatch):
    test_tool = load_test_tool()
    progress_statuses = []
    monkeypatch.setattr(test_tool.logger, "progress_status", progress_statuses.append)
    classes = ["wake", "n1", "n2", "n3", "rem"]
    dataset = UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["EEG"], unit="uV")],
        sample_frequency=10,
        total_input="30s",
        stride="30s",
    )
    package_path = PackagedModel(name="sleep", task="sleep", model=MeanClassifier(len(classes)), dataset=dataset, classification_contract={"type": "single-head-multiclass", "classes": classes, "sequence_len": 1, "target_resolution": "30s"}).save(tmp_path / "package")
    manifest = write_unsplit_manifest(tmp_path / "files.yml", [str(REPO_ROOT / "tests" / "data" / "signals_01.edf")])
    output = tmp_path / "metrics.jsonl"
    config = {
        "seed": 17,
        "data": {
            "label": "synthetic",
            "files": str(manifest),
            "dataset": {
                "name": "sleepwalker.datasets.SyntheticDataset.SyntheticDataset",
                "event_mapping": {label: label for label in classes},
            },
            "num_workers": 0,
            "strict": True,
        },
        "analyses": [{"name": "all"}],
        "test": {"device": "cpu", "batch_size": 64, "num_workers_dataloader": 0, "output": str(output)},
    }

    result = test_tool.execute(package_path, config)
    records = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]

    assert [record["record_type"] for record in records] == ["run", "patient", "aggregate"]
    assert records[2]["dataset"] == "synthetic"
    assert records[2]["n_patients_evaluated"] == 1
    assert records[2]["metrics"]["n_windows"] > 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
    assert any("windows" in status and "failed batches" in status and "inference" not in status and "predictions" not in status for status in progress_statuses)
    assert any("analysis | patient" in status and "kept" in status for status in progress_statuses)


def test_paired_graph_package_uses_the_regular_labelled_evaluator(tmp_path):
    sleep_classes = ["wake", "n2", "n3", "rem", "other"]
    event_classes = ["event", "no event"]
    slow = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=10, total_input="60s", stride="30s")
    fast = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=5, total_input="20s", stride="20s")
    nodes = {
        "long": GraphNode(MeanClassifier(len(sleep_classes), ts_len=600), {"long": {"classes": sleep_classes, "sequence_len": 1}}),
        "short": GraphNode(MeanClassifier(len(event_classes), ts_len=100), {"short": {"classes": event_classes, "sequence_len": 1}}),
    }
    package = PackagedModel(
        name="paired",
        task="multitask",
        model=ModelGraphClassifier(nodes=nodes, method="probability"),
        dataset=pair_datasets({"long": slow, "short": fast}),
        classification_contract={
            "type": "multitask",
            "tasks": {
                "long": {"classes": sleep_classes, "n_steps": 1, "target_resolution": "60s", "target_offset": "0s", "default": "other", "percentage": 0.5, "soft_boundaries": False},
                "short": {"classes": event_classes, "n_steps": 1, "target_resolution": "20s", "target_offset": "20s", "default": "no event", "percentage": 0.5, "soft_boundaries": False},
            },
        },
    )
    manifest = write_unsplit_manifest(tmp_path / "files.yml", [str(REPO_ROOT / "tests" / "data" / "signals_01.edf")])
    output = tmp_path / "paired.jsonl"
    config = {
        "seed": 17,
        "data": {
            "label": "synthetic",
            "files": str(manifest),
            "dataset": {
                "name": "sleepwalker.datasets.SyntheticDataset.SyntheticDataset",
                "event_mapping": {"wake": "wake", "n1": "event", "n2": "n2", "n3": "n3", "rem": "rem"},
            },
            "num_workers": 0,
            "strict": True,
        },
        "analyses": [{"name": "all"}],
        "test": {"device": "cpu", "batch_size": 32, "num_workers_dataloader": 0, "output": str(output)},
    }

    result = load_test_tool().execute(package, config)
    records = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]

    aggregates = [record for record in records if record["record_type"] == "aggregate"]
    assert {record["task"] for record in aggregates} == {"long", "short"}
    assert all(record["n_patients_evaluated"] == 1 for record in aggregates)
