#!/usr/bin/env python3
"""Evaluate the packaged classifier named by a test config on labelled EDF manifests."""

from __future__ import annotations

import argparse
import copy
from functools import partial
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sleepwalker-matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.deployment import load_packaged_model
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.Run import seed_everything
from sleepwalker.trainer.utils.disk import NumpyEncoder, json_ready
from sleepwalker.trainer.utils.splits import fold_names, load_files
from sleepwalker.trainer.utils.targets import annotation_coverage, prepare_multiclass_target, prepare_single_target
from sleepwalker.training.execution import RepeatedViewModel, execute_batches
from sleepwalker.training.loader import build_loader
from sleepwalker.utils import logger
from tools.train import apply_patient_filter, build_callback, build_channel, build_value, import_name, read_yaml


def read_config(path: str | Path) -> dict[str, Any]:
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    if not isinstance(config["data"], (dict, list)) or not config["data"]:
        raise ValueError("data must be a dataset mapping or list of mappings.")
    if not isinstance(config["analyses"], list) or not config["analyses"]:
        raise ValueError("analyses must be a non-empty list.")
    if not str(config["test"]["output"]).endswith(".jsonl"):
        raise ValueError("test.output must end in .jsonl.")
    if not isinstance(config["test"].get("package"), str) or not config["test"]["package"]:
        raise ValueError("test.package must name the trained package directory.")
    return config


def execute(package_path: str | Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run the visible patient-streaming evaluation workflow."""
    seed_everything(int(config.get("seed", 17)))
    test_options = config["test"]
    output = Path(test_options["output"])
    if output.exists() and not bool(test_options.get("overwrite", False)):
        raise FileExistsError(f"Output already exists: {output}. Set test.overwrite: true to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    device = str(test_options.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    package = load_packaged_model(package_path, map_location=device)
    if package.classification_contract is None:
        raise TypeError(f"Package '{package.name}' does not contain a classifier.")
    package.model.to(device).eval()
    analyses = build_analyses(config["analyses"])

    records = []
    with temporary.open("w", encoding="utf-8") as handle:
        run_record = {
            "record_type": "run",
            "package": package.name,
            "package_path": str(package_path),
            "task": package.task,
            "classification_contract": json_ready(package.classification_contract),
            "git_commit": package.git_commit,
            "fold": package.config.get("fold") if isinstance(package.config, dict) else None,
            "seed": int(config.get("seed", 17)),
            "config": json_ready(config),
        }
        write_record(handle, records, run_record)

        entries = config["data"] if isinstance(config["data"], list) else [config["data"]]
        for entry in entries:
            dataset, patients, selected_fold = prepare_dataset(package, entry)
            logger.info(f"Initializing '{entry['label']}' with {len(patients)} requested patients.")
            dataset.initialize(patients, num_workers=int(entry.get("num_workers", 4)), strict=bool(entry.get("strict", True)))
            if dataset.get_n_patients() == 0 or len(dataset) == 0:
                raise ValueError(f"Dataset '{entry['label']}' produced no evaluable windows.")

            accumulators: dict[tuple[str, str], MetricAccumulator] = {}
            prediction_options = {**test_options, "device": device, "seed": int(config.get("seed", 17))}
            for patient, signals, task_frames in predict_patients(package, dataset, prediction_options):
                for analysis_name, pipeline in analyses:
                    for task, (classes, patient_frame) in task_frames.items():
                        analyzed = apply_pipeline(patient_frame, pipeline, patient=patient, signals=signals)
                        if analyzed.empty:
                            continue
                        key = (analysis_name, task)
                        accumulator = accumulators.setdefault(key, MetricAccumulator(classes))
                        patient_metrics = patient_classification_metrics(analyzed, classes)
                        accumulator.update(patient, patient_metrics)
                        write_record(handle, records, {
                            "record_type": "patient",
                            "package": package.name,
                            "task": task,
                            "dataset": str(entry["label"]),
                            "fold": selected_fold,
                            "analysis": analysis_name,
                            "patient": patient,
                            "classes": classes,
                            "metrics": patient_metrics,
                        })

            expected_tasks = [package.task] if package.classification_contract["type"] == "single-head-multiclass" else list(package.classification_contract["tasks"])
            if any(task is None for task in expected_tasks):
                raise ValueError("A single-head package must define task.")
            for analysis_name, _ in analyses:
                for task in expected_tasks:
                    key = (analysis_name, task)
                    if key not in accumulators:
                        raise ValueError(f"Analysis '{analysis_name}' produced no rows for task '{task}' on dataset '{entry['label']}'.")
                    write_record(handle, records, {
                        "record_type": "aggregate",
                        "package": package.name,
                        "task": task,
                        "dataset": str(entry["label"]),
                        "fold": selected_fold,
                        "analysis": analysis_name,
                        "classes": accumulators[key].classes,
                        "n_patients_requested": len(patients),
                        "n_patients_initialized": dataset.get_n_patients(),
                        **accumulators[key].finalize(),
                    })

    os.replace(temporary, output)
    return records


def write_record(handle, records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n")
    records.append(record)


def build_analyses(specs: list[dict[str, Any]]) -> list[tuple[str, list[Any]]]:
    analyses = []
    names = set()
    for spec in specs:
        name = str(spec["name"])
        if name in names:
            raise ValueError(f"Duplicate analysis name '{name}'.")
        names.add(name)
        analyses.append((name, [build_callback(step) for step in spec.get("pipeline", [])]))
    return analyses


def apply_pipeline(frame: pd.DataFrame, pipeline: list[Any], *, patient: str, signals: Sequence[str]) -> pd.DataFrame:
    current = frame.copy()
    for transform in pipeline:
        current = transform(current, patient=patient, signals=signals)
        if not isinstance(current, pd.DataFrame):
            raise TypeError(f"Analysis transform {transform} returned {type(current).__name__}, expected DataFrame.")
        if current.empty:
            break
    return current


def filter_patient_signals(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, include_all: Sequence[str] | None = None, include_any: Sequence[str] | None = None, exclude_any: Sequence[str] | None = None) -> pd.DataFrame:
    del patient
    available = set(signals)
    keep = (not include_all or set(include_all).issubset(available)) and (not include_any or bool(set(include_any).intersection(available))) and (not exclude_any or not set(exclude_any).intersection(available))
    return frame if keep else frame.iloc[0:0]


def filter_annotation(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, labels: Sequence[str], minimum_coverage: float = 0.5) -> pd.DataFrame:
    del patient, signals
    columns = [f"annotation__{label}" for label in labels]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Annotation filter refers to unavailable labels: {missing}.")
    coverage = frame[columns].sum(axis=1).clip(upper=1.0)
    return frame.loc[coverage >= float(minimum_coverage)]


def resolve_overlaps(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, method: str = "mean_probability") -> pd.DataFrame:
    del patient, signals
    if method not in {"mean_probability", "majority_vote"}:
        raise ValueError(f"Unknown overlap method '{method}'.")
    probability_columns = [column for column in frame if column.startswith("prob__")]
    annotation_columns = [column for column in frame if column.startswith("annotation__")]
    if int(frame.groupby("time")["target"].nunique().max()) != 1:
        raise ValueError("Overlapping predictions disagree on the target at one timestamp.")
    work = frame.copy()
    if method == "majority_vote":
        votes = work[probability_columns].to_numpy().argmax(axis=1)
        for class_index, column in enumerate(probability_columns):
            work[column] = (votes == class_index).astype(float)
    aggregation = {"target": "first", **{column: "mean" for column in probability_columns + annotation_columns}}
    return work.groupby("time", sort=True, as_index=False).agg(aggregation)


def smooth_probabilities(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, window: int = 3) -> pd.DataFrame:
    del patient, signals
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd integer.")
    if window == 1 or len(frame) < 2:
        return frame
    probability_columns = [column for column in frame if column.startswith("prob__")]
    work = frame.sort_values("time").copy()
    differences = work["time"].diff()
    positive = differences[differences > pd.Timedelta(0)]
    expected = positive.min() if not positive.empty else None
    segments = pd.Series(0, index=work.index) if expected is None else (differences > expected * 1.5).cumsum()
    work[probability_columns] = work.groupby(segments, sort=False)[probability_columns].transform(lambda values: values.rolling(window, center=True, min_periods=1).mean())
    probabilities = work[probability_columns].to_numpy(dtype=float)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    work[probability_columns] = probabilities
    return work


def package_fold(package) -> str | None:
    if not isinstance(package.config, dict):
        return None
    return package.config.get("fold")


def resolve_entry_files(package, entry: Mapping[str, Any]) -> tuple[list[str], str | None]:
    if ("files" in entry) == ("split" in entry):
        raise ValueError(f"Dataset '{entry['label']}' must use exactly one of files or split.")
    if "files" in entry:
        path = entry["files"]
        return load_files(path), None
    split = entry["split"]
    path = split["file"]
    selected_fold = split.get("fold", package_fold(package))
    available_folds = fold_names(path)
    if selected_fold is None and len(available_folds) == 1:
        selected_fold = available_folds[0]
    files = load_files(path, fold=selected_fold, partition=str(split.get("partition", "test")))
    return files, selected_fold


def build_dataset_arguments(spec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    values = copy.deepcopy(dict(spec))
    name = str(values.pop("name"))
    arguments = {}
    for key, value in values.items():
        if key == "channels":
            arguments[key] = [build_channel(channel) for channel in value]
        elif key.startswith("prepare_"):
            arguments[key] = build_callback(value)
        else:
            arguments[key] = build_value(value)
    return name, arguments


def prepare_dataset(package, entry: Mapping[str, Any]):
    dataset_name, overrides = build_dataset_arguments(entry["dataset"])
    if "event_mapping" not in overrides or not overrides["event_mapping"]:
        raise ValueError(f"Dataset '{entry['label']}' must provide event_mapping.")
    arguments = package.dataset.dataset_kwargs()
    arguments.update(overrides)
    arguments["prepare_target"] = build_evaluation_target(package, entry, sorted(set(arguments["event_mapping"].values())))
    dataset = import_name(dataset_name)(**arguments)
    package.assert_compatible(dataset, allow_preprocessing_override=True)
    patients, selected_fold = resolve_entry_files(package, entry)
    if entry.get("patient_filter") is not None:
        patients = apply_patient_filter(entry["patient_filter"], patients, dataset, num_workers=int(entry.get("num_workers", 4)))
    if not patients:
        raise ValueError(f"Dataset '{entry['label']}' has no patients after filtering.")
    return dataset, patients, selected_fold


def build_evaluation_target(package, entry: Mapping[str, Any], annotation_labels: list[str]):
    options = copy.deepcopy(entry.get("target", {}))
    contract = package.classification_contract
    if contract["type"] == "single-head-multiclass":
        classes = list(contract["classes"])
        missing = sorted(set(classes) - set(annotation_labels))
        if len(missing) > 1:
            raise ValueError(f"Dataset '{entry['label']}' event_mapping is missing multiple output classes {missing}.")
        return partial(prepare_single_target, target_classes=classes, sequence_len=int(contract["sequence_len"]), annotation_labels=annotation_labels, **options)
    task_config = {
        task: {
            "labels": list(spec["classes"]),
            "sequence_len": int(spec["n_steps"]),
            "target_resolution": spec["target_resolution"],
            "default": spec["default"],
            "percentage": float(spec["percentage"]),
            "soft_boundaries": bool(spec["soft_boundaries"]),
            "step_mask": None,
        }
        for task, spec in contract["tasks"].items()
    }
    unknown_tasks = sorted(set(options) - set(task_config))
    if unknown_tasks:
        raise ValueError(f"Dataset '{entry['label']}' target overrides refer to unknown tasks {unknown_tasks}.")
    for task_name, task_options in options.items():
        task_config[task_name].update(task_options)
    task_config = MultiLabelTrainer.normalize_task_config(task_config)
    return partial(prepare_multitask_target, task_config=task_config, annotation_labels=annotation_labels)


def task_target_slice(target: pd.DataFrame, task: Mapping[str, Any]) -> pd.DataFrame:
    frequency = target.index.freq or pd.infer_freq(target.index)
    if frequency is None:
        raise ValueError("Multitask annotations require a regular time index.")
    samples = int(round(pd.to_timedelta(task["target_span"]) / pd.to_timedelta(frequency)))
    start = (len(target) - samples) // 2
    return target.iloc[start:start + samples]


def prepare_multitask_target(target, target_extra=None, patient=None, time=None, *, task_config: Mapping[str, Mapping[str, Any]], annotation_labels: Sequence[str]):
    prepared = MultiLabelTrainer.prepare_target(target, target_extra=target_extra, patient=patient, time=time, task_config=task_config)
    if prepared is None:
        return None
    tasks = list(task_config.values())
    max_steps = max(int(task["n_steps"]) for task in tasks)
    annotations = torch.zeros((len(tasks), max_steps, len(annotation_labels)), dtype=torch.float32)
    for task_index, task in enumerate(tasks):
        n_steps = int(task["n_steps"])
        annotations[task_index, :n_steps] = annotation_coverage(task_target_slice(target, task), n_steps, annotation_labels)
    prepared["annotation"] = annotations
    return prepared


def single_frame(package, dataset, outputs, batch) -> dict[str, tuple[list[str], pd.DataFrame]]:
    classes = list(package.classification_contract["classes"])
    sequence_len = int(package.classification_contract["sequence_len"])
    probabilities = torch.softmax(outputs.detach().cpu(), dim=-1).numpy()
    expected = (len(batch["patient"]), sequence_len, len(classes))
    if probabilities.shape != expected or tuple(batch["target"].shape) != expected:
        raise ValueError(f"Expected outputs and targets shaped {expected}, got {probabilities.shape} and {tuple(batch['target'].shape)}.")
    targets = batch["target"].argmax(dim=-1).numpy().reshape(-1)
    valid = batch.get("target_mask", torch.ones(expected[:2], dtype=torch.bool)).numpy().reshape(-1)
    annotations = batch["annotation"].numpy().reshape(-1, batch["annotation"].shape[-1])
    step = pd.to_timedelta(dataset.target_resolution) / sequence_len
    frame = pd.DataFrame({"patient": [str(patient) for patient in batch["patient"] for _ in range(sequence_len)], "time": [pd.Timestamp(time) + index * step for time in batch["time"] for index in range(sequence_len)], "target": targets})
    for index, name in enumerate(classes):
        frame[f"prob__{name}"] = probabilities.reshape(-1, len(classes))[:, index]
    for index, name in enumerate(dataset.label_classes):
        frame[f"annotation__{name}"] = annotations[:, index]
    return {package.task: (classes, frame.loc[valid].reset_index(drop=True))}


def multitask_frames(package, dataset, outputs, batch) -> dict[str, tuple[list[str], pd.DataFrame]]:
    frames = {}
    task_specs = [{"task": task, **spec, "labels": spec["classes"]} for task, spec in package.classification_contract["tasks"].items()]
    for task_index, task in enumerate(task_specs):
        name = task["task"]
        classes = list(task["labels"])
        n_steps = int(task["n_steps"])
        probabilities = torch.softmax(outputs[name].detach().cpu(), dim=-1).numpy()
        targets = batch["target"][:, task_index, :n_steps, :len(classes)].argmax(dim=-1).numpy().reshape(-1)
        valid = batch["target_mask"][:, task_index, :n_steps].numpy().reshape(-1)
        annotations = batch["annotation"][:, task_index, :n_steps].numpy().reshape(-1, batch["annotation"].shape[-1])
        resolution = pd.to_timedelta(task["target_resolution"])
        offset = pd.to_timedelta(task["target_offset"])
        frame = pd.DataFrame({"patient": [str(patient) for patient in batch["patient"] for _ in range(n_steps)], "time": [pd.Timestamp(time) + offset + index * resolution for time in batch["time"] for index in range(n_steps)], "target": targets})
        for index, class_name in enumerate(classes):
            frame[f"prob__{class_name}"] = probabilities.reshape(-1, len(classes))[:, index]
        for index, annotation_name in enumerate(dataset.label_classes):
            frame[f"annotation__{annotation_name}"] = annotations[:, index]
        frames[name] = (classes, frame.loc[valid].reset_index(drop=True))
    return frames


def predict_patients(package, dataset, test_options: Mapping[str, Any]):
    workers = int(test_options.get("num_workers_dataloader", 0))
    n_repeat = int(test_options.get("n_repeat", 1))
    loader = build_loader(dataset, batch_size=int(test_options.get("batch_size", 64)), num_workers=workers, n_samples=None, collate_fn=batch_collate, shuffle=False, seed=int(test_options.get("seed", 0)), n_repeat=n_repeat)
    model = RepeatedViewModel(package.model) if n_repeat > 1 else package.model
    current_patient = None
    current_chunks: dict[str, list[pd.DataFrame]] = {}
    current_classes: dict[str, list[str]] = {}
    signal_map = {str(file.path): list(read_edf_meta(file.path)["signals"]) for file in dataset.edf_files}
    for outputs, batch in execute_batches(model, loader, test_options["device"]):
        if package.classification_contract["type"] == "single-head-multiclass":
            batch_frames = single_frame(package, dataset, outputs, batch)
        else:
            batch_frames = multitask_frames(package, dataset, outputs, batch)
        ordered_patients = list(dict.fromkeys(str(patient) for patient in batch["patient"]))
        for patient in ordered_patients:
            if current_patient is not None and patient != current_patient:
                yield current_patient, signal_map[current_patient], {task: (current_classes[task], pd.concat(chunks, ignore_index=True)) for task, chunks in current_chunks.items()}
                current_chunks = {}
                current_classes = {}
            current_patient = patient
            for task, (classes, frame) in batch_frames.items():
                selected = frame.loc[frame["patient"] == patient].drop(columns="patient")
                if not selected.empty:
                    current_chunks.setdefault(task, []).append(selected)
                    current_classes[task] = classes
    if current_patient is not None:
        yield current_patient, signal_map[current_patient], {task: (current_classes[task], pd.concat(chunks, ignore_index=True)) for task, chunks in current_chunks.items()}


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def binary_curve_metrics(true_positive: float, false_positive: float, false_negative: float, true_negative: float) -> tuple[float | None, float | None]:
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    if positives == 0:
        return None, None
    recall = safe_divide(true_positive, positives)
    precision = safe_divide(true_positive, true_positive + false_positive)
    prevalence = safe_divide(positives, positives + negatives)
    auprc = recall * precision + (1.0 - recall) * prevalence
    auroc = None if negatives == 0 else 0.5 * (recall + safe_divide(true_negative, negatives))
    return auroc, auprc


def confusion_metrics(matrix: np.ndarray, classes: list[str]) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix).astype(float)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    true_negative = matrix.sum() - true_positive - false_positive - false_negative
    precision = np.divide(true_positive, true_positive + false_positive, out=np.zeros_like(true_positive), where=true_positive + false_positive > 0)
    recall = np.divide(true_positive, true_positive + false_negative, out=np.zeros_like(true_positive), where=true_positive + false_negative > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)
    total = int(matrix.sum())
    accuracy = safe_divide(float(true_positive.sum()), total)
    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)
    expected = safe_divide(float(np.dot(row_sums, column_sums)), total * total)
    kappa = None if total == 0 or math.isclose(expected, 1.0) else float((accuracy - expected) / (1.0 - expected))
    result = {
        "n_windows": total,
        "accuracy": accuracy,
        "f1_micro": accuracy,
        "f1_macro": float(f1.mean()),
        "cohen_kappa": kappa,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "sensitivity_macro": float(recall.mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": {name: {"precision": float(precision[index]), "recall": float(recall[index]), "sensitivity": float(recall[index]), "f1": float(f1[index]), "support": int(row_sums[index])} for index, name in enumerate(classes)},
    }
    aurocs = []
    auprcs = []
    for index, name in enumerate(classes):
        auroc, auprc = binary_curve_metrics(true_positive[index], false_positive[index], false_negative[index], true_negative[index])
        result["per_class"][name].update({"auroc": auroc, "auprc": auprc})
        if auroc is not None:
            aurocs.append(auroc)
        if auprc is not None:
            auprcs.append(auprc)
    micro_auroc, micro_auprc = binary_curve_metrics(true_positive.sum(), false_positive.sum(), false_negative.sum(), true_negative.sum())
    result.update({"auroc_micro": micro_auroc, "auroc_macro": float(np.mean(aurocs)) if aurocs else None, "auprc_micro": micro_auprc, "auprc_macro": float(np.mean(auprcs)) if auprcs else None})
    return result


def patient_classification_metrics(frame: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    probabilities = frame[[f"prob__{name}" for name in classes]].to_numpy(dtype=float)
    target = frame["target"].to_numpy(dtype=int)
    prediction = probabilities.argmax(axis=1)
    return confusion_metrics(confusion_matrix(target, prediction, labels=range(len(classes))), classes)


def mean_patient_metrics(values: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    fields = ["accuracy", "f1_micro", "f1_macro", "cohen_kappa", "auroc_micro", "auroc_macro", "auprc_micro", "auprc_macro", "precision_macro", "recall_macro", "sensitivity_macro"]
    result = {field: float(np.mean([value[field] for value in values if value[field] is not None])) if any(value[field] is not None for value in values) else None for field in fields}
    result["per_class"] = {}
    for name in classes:
        result["per_class"][name] = {}
        for field in ["precision", "recall", "sensitivity", "f1", "auroc", "auprc"]:
            current = [value["per_class"][name][field] for value in values if value["per_class"][name].get(field) is not None]
            result["per_class"][name][field] = float(np.mean(current)) if current else None
    return result


class MetricAccumulator:
    def __init__(self, classes: list[str]):
        self.classes = list(classes)
        self.confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        self.patient_metrics: list[dict[str, Any]] = []
        self.patient_confusion_matrices: dict[str, list[list[int]]] = {}

    def update(self, patient: str, metrics: dict[str, Any]) -> None:
        self.confusion_matrix += np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        self.patient_metrics.append(metrics)
        self.patient_confusion_matrices[patient] = metrics["confusion_matrix"]

    def finalize(self) -> dict[str, Any]:
        metrics = confusion_metrics(self.confusion_matrix, self.classes)
        return {
            "n_patients_evaluated": len(self.patient_metrics),
            "metrics": metrics,
            "patient_mean_metrics": mean_patient_metrics(self.patient_metrics, self.classes),
            "curve_method": "summed_patient_confusion_matrices",
            "patient_confusion_matrices": self.patient_confusion_matrices,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Evaluation YAML under configs/test/.")
    args = parser.parse_args()
    config = read_config(args.config)
    execute(config["test"]["package"], config)


if __name__ == "__main__":
    main()
