from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.Stages import Stages
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel
from sleepwalker.models.ModelGraphClassifier import GraphNode, ModelGraphClassifier, pair_datasets


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
    spec = importlib.util.spec_from_file_location("sleepwalker_test_tool", REPO_ROOT / "tools" / "test.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_package():
    dataset = UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["source-eeg"], unit="uV")],
        sample_frequency=100,
        total_input="60s",
        target_resolution="20s",
        stride="5s",
    )
    return PackagedModel(name="package", task="event", model=MeanClassifier(2, ts_len=6000), dataset=dataset, classification_contract={"type": "single-head-multiclass", "classes": ["event", "no event"], "sequence_len": 2})


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
        "multitask_sleepwalker",
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
    assert dataset.target_resolution == pd.Timedelta("20s")
    assert dataset.stride == pd.Timedelta("5s")
    assert dataset.channels == [ChannelConfig("EEG", ["external-eeg"], normalizer=None, unit="uV")]
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
        lambda current, patient, signals: test_tool.filter_patient_signals(current, patient, signals, include_any=["Mask Pressure"]),
        lambda current, patient, signals: test_tool.filter_annotation(current, patient, signals, labels=["sleep"], minimum_coverage=0.5),
    ]

    selected = test_tool.apply_pipeline(frame, pipeline, patient="patient.edf", signals=["EEG", "Mask Pressure"])
    excluded = test_tool.apply_pipeline(frame, pipeline, patient="patient.edf", signals=["EEG"])

    assert selected["target"].tolist() == [0]
    assert excluded.empty


def test_annotation_filter_uses_union_of_requested_labels():
    test_tool = load_test_tool()
    frame = pd.DataFrame({
        "annotation__n1": [0.3, 0.1],
        "annotation__n2": [0.3, 0.2],
    })

    selected = test_tool.filter_annotation(frame, "patient.edf", [], labels=["n1", "n2"], minimum_coverage=0.5)

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

    resolved = test_tool.resolve_overlaps(frame, "patient.edf", [], method="mean_probability")
    smoothed = test_tool.smooth_probabilities(resolved, "patient.edf", [], window=1)
    metrics = test_tool.patient_classification_metrics(smoothed, ["negative", "positive"])

    assert len(resolved) == 2
    assert resolved.iloc[0]["prob__negative"] == pytest.approx(0.7)
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["per_class"]["positive"]["sensitivity"] == 1.0


def test_accumulator_keeps_confusion_matrices_not_predictions():
    test_tool = load_test_tool()
    classes = ["negative", "positive"]
    frame = pd.DataFrame({
        "target": [0, 0, 1, 1],
        "prob__negative": [0.9, 0.8, 0.2, 0.1],
        "prob__positive": [0.1, 0.2, 0.8, 0.9],
    })
    patient_metrics = test_tool.patient_classification_metrics(frame, classes)
    accumulator = test_tool.MetricAccumulator(classes)

    accumulator.update("patient.edf", patient_metrics)
    aggregate = accumulator.finalize()

    assert aggregate["patient_confusion_matrices"] == {"patient.edf": [[2, 0], [0, 2]]}
    assert aggregate["metrics"]["accuracy"] == 1.0
    assert "auroc_macro" not in aggregate["metrics"]
    assert not hasattr(accumulator, "predictions")


def test_empty_accumulator_reports_zero_evaluated_windows():
    aggregate = load_test_tool().MetricAccumulator(["negative", "positive"]).finalize()

    assert aggregate["n_patients_evaluated"] == 0
    assert aggregate["metrics"]["n_windows"] == 0
    assert aggregate["metrics"]["cohen_kappa"] is None


def test_packaged_model_evaluates_to_jsonl(tmp_path):
    test_tool = load_test_tool()
    classes = ["wake", "n1", "n2", "n3", "rem"]
    dataset = UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["EEG"], unit="uV")],
        sample_frequency=10,
        total_input="30s",
        target_resolution="30s",
        stride="30s",
    )
    package_path = PackagedModel(name="sleep", task="sleep", model=MeanClassifier(len(classes)), dataset=dataset, classification_contract={"type": "single-head-multiclass", "classes": classes, "sequence_len": 1}).save(tmp_path / "package")
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

    records = test_tool.execute(package_path, config)

    assert [record["record_type"] for record in records] == ["run", "patient", "aggregate"]
    assert records[2]["dataset"] == "synthetic"
    assert records[2]["n_patients_evaluated"] == 1
    assert records[2]["metrics"]["n_windows"] > 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_paired_graph_package_uses_the_regular_labelled_evaluator(tmp_path):
    sleep_classes = ["wake", "n2", "n3", "rem", "other"]
    event_classes = ["event", "no event"]
    slow = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=10, total_input="60s", target_resolution="60s", stride="30s")
    fast = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=5, total_input="20s", target_resolution="20s", stride="20s")
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

    records = load_test_tool().execute(package, config)

    aggregates = [record for record in records if record["record_type"] == "aggregate"]
    assert {record["task"] for record in aggregates} == {"long", "short"}
    assert all(record["n_patients_evaluated"] == 1 for record in aggregates)
