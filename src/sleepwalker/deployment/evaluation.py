"""Shared dataset, metric, coverage, and dependency evaluation helpers."""

import copy
from functools import partial
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch

from sleepwalker.config import apply_patient_filter, build_callback, build_channel, build_value, import_name
from sleepwalker.datasets.PairedDataset import PairedDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment.package import PackagedModel
from sleepwalker.metrics import accuracy_from_confusion_matrix, cohen_kappa_from_confusion_matrix, f1_per_class_from_confusion_matrix, precision_from_confusion_matrix, recall_from_confusion_matrix, support_from_confusion_matrix
from sleepwalker.trainer.utils.disk import NumpyEncoder
from sleepwalker.trainer.utils.splits import fold_names, load_files
from sleepwalker.trainer.utils.targets import normalize_multitask_config, prepare_multitask_target, prepare_single_target
from sleepwalker.utils import logger


def write_prediction_feather(path: str | Path, predictions: pd.DataFrame) -> None:
    """Write one raw prediction table to Feather."""
    required = {"patient", "task", "time"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing Feather columns {missing}.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(predictions):,} prediction rows to {path}.")
    predictions.to_feather(path, compression="zstd")
    logger.info(f"Wrote prediction file {path} ({path.stat().st_size / 1024**3:.2f} GiB).")


def read_prediction_feather(path: str | Path, patients: list[str] | None = None) -> pd.DataFrame:
    """Read a raw prediction table, optionally retaining selected patients."""
    predictions = pd.read_feather(path)
    if patients is None:
        return predictions
    selected = set(map(str, patients))
    available = set(predictions["patient"].astype(str))
    missing = sorted(selected - available)
    if missing:
        raise KeyError(f"Prediction table is missing patients {missing[:5]}.")
    return predictions.loc[predictions["patient"].astype(str).isin(selected)].reset_index(drop=True)


def package_fold(package: PackagedModel) -> str | None:
    if not isinstance(package.config, dict):
        return None
    return package.config.get("fold")


def resolve_entry_files(package: PackagedModel, entry: Mapping[str, Any]) -> tuple[list[str], str | None]:
    if ("files" in entry) == ("split" in entry):
        raise ValueError(f"Dataset '{entry['label']}' must use exactly one of files or split.")
    if "files" in entry:
        return load_files(entry["files"]), None
    split = entry["split"]
    selected_fold = split.get("fold", package_fold(package))
    available_folds = fold_names(split["file"])
    if selected_fold is None and len(available_folds) == 1:
        selected_fold = available_folds[0]
    files = load_files(split["file"], fold=selected_fold, partition=str(split.get("partition", "test")))
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


def build_evaluation_target(package: PackagedModel, entry: Mapping[str, Any], annotation_labels: list[str]):
    options = copy.deepcopy(entry.get("target", {}))
    contract = package.classification_contract
    if contract is None:
        raise TypeError(f"Package '{package.name}' does not contain a classifier.")
    if contract["type"] == "single-head-multiclass":
        classes = list(contract["classes"])
        missing = sorted(set(classes) - set(annotation_labels))
        if len(missing) > 1:
            raise ValueError(f"Dataset '{entry['label']}' event_mapping is missing multiple output classes {missing}.")
        return partial(prepare_single_target, target_classes=classes, sequence_len=int(contract["sequence_len"]), target_resolution=contract["target_resolution"], target_offset=contract.get("target_offset", "0s"), annotation_labels=annotation_labels, allow_invalid=True, **options)
    task_config = {
        task: {
            "labels": list(spec["classes"]),
            "sequence_len": int(spec["n_steps"]),
            "target_resolution": spec["target_resolution"],
            "target_offset": spec["target_offset"],
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
    return partial(prepare_multitask_target, task_config=normalize_multitask_config(task_config), annotation_labels=annotation_labels, allow_partial=True)


def validate_dataset_entry(entry: Mapping[str, Any]) -> None:
    """Require one explicit dataset form without global/task fallback rules."""
    if any(key in entry for key in ("channels_by_task", "overrides", "task_overrides")):
        raise ValueError("Use complete dataset or datasets specifications, not preprocessing overrides.")
    if ("dataset" in entry) == ("datasets" in entry):
        raise ValueError("Provide exactly one of data.dataset or data.datasets.")
    if "dataset" in entry and "reference" in entry:
        raise ValueError("data.reference is only used with task datasets for a composed package.")
    if "datasets" in entry and (not isinstance(entry["datasets"], Mapping) or not entry["datasets"]):
        raise ValueError("data.datasets must be a non-empty mapping of tasks to complete specifications.")


def build_explicit_dataset(spec: Mapping[str, Any]):
    """Construct YAML preprocessing directly; never inherit package settings."""
    required = {"name", "channels", "sample_frequency", "total_input", "stride", "resample_type"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Evaluation dataset must explicitly specify {missing}.")
    for channel in spec["channels"]:
        if not {"logical_name", "physical_names", "read_mode", "preprocessors"} <= set(channel):
            raise ValueError("Evaluation channels must explicitly specify logical_name, physical_names, read_mode and preprocessors.")
    dataset_name, arguments = build_dataset_arguments(spec)
    if not arguments.get("event_mapping"):
        raise ValueError("Evaluation dataset must provide event_mapping.")
    if "prepare_target" in arguments:
        raise ValueError("Evaluation targets come from the package contract and data.target; omit dataset.prepare_target.")
    arguments["rejection_strategy"] = "none"
    return import_name(dataset_name)(**arguments)


def prepare_dataset(package: PackagedModel, entry: Mapping[str, Any]):
    validate_dataset_entry(entry)
    if isinstance(package.dataset, PairedDataset):
        if "datasets" not in entry or set(entry["datasets"]) != set(package.dataset.datasets):
            raise ValueError("A composed package requires data.datasets naming every expert exactly once.")
        if "reference" not in entry:
            raise ValueError("A composed package requires data.reference for annotation timing and window anchors.")
        base = build_explicit_dataset(entry["reference"])
        inputs = {task: UnlabelledDataset.from_dataset(build_explicit_dataset(spec)) for task, spec in entry["datasets"].items()}
        dataset = PairedDataset(inputs, base=base, input_offsets=package.dataset.input_offsets)
    else:
        if "reference" in entry:
            raise ValueError("A single-input package does not use data.reference.")
        spec = entry["dataset"] if "dataset" in entry else entry["datasets"][package.task]
        dataset = build_explicit_dataset(spec)
        base = dataset
    annotation_labels = sorted(set(base.event_mapping.values()))
    base.prepare_target_callback = build_evaluation_target(package, entry, annotation_labels)
    if not hasattr(dataset, "set_rejection_strategy"):
        raise TypeError(f"Evaluation dataset {dataset.__class__.__name__} does not support rejection strategies.")
    dataset.set_rejection_strategy("none")
    package.assert_compatible(dataset)
    patients, selected_fold = resolve_entry_files(package, entry)
    if entry.get("patient_filter") is not None:
        original_count = len(patients)
        patients = apply_patient_filter(entry["patient_filter"], patients, dataset, num_workers=int(entry.get("num_workers", 4)))
        logger.info(f"Dataset '{entry['label']}' patient filter kept {len(patients)}/{original_count} paths.")
    if not patients:
        raise ValueError(f"Dataset '{entry['label']}' has no patients after filtering.")
    return dataset, patients, selected_fold


def write_record(handle, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n")


def confusion_metrics(matrix: np.ndarray, classes: list[str]) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)):
        raise ValueError(f"Expected a {len(classes)}x{len(classes)} confusion matrix, got {matrix.shape}.")
    precision = precision_from_confusion_matrix(matrix)
    recall = recall_from_confusion_matrix(matrix)
    f1 = f1_per_class_from_confusion_matrix(matrix)
    support = support_from_confusion_matrix(matrix)
    accuracy = accuracy_from_confusion_matrix(matrix)
    return {
        "n_windows": int(matrix.sum()),
        "accuracy": accuracy,
        "f1_micro": accuracy,
        "f1_macro": float(f1.mean()),
        "cohen_kappa": cohen_kappa_from_confusion_matrix(matrix),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "sensitivity_macro": float(recall.mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": {name: {"precision": float(precision[index]), "recall": float(recall[index]), "sensitivity": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index, name in enumerate(classes)},
    }


def patient_classification_metrics(frame: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    required = {"target", *(f"prob__{name}" for name in classes)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot calculate classification metrics; missing columns {missing}.")
    probabilities = frame[[f"prob__{name}" for name in classes]].to_numpy(dtype=float)
    target = frame["target"].to_numpy(dtype=int)
    prediction = probabilities.argmax(axis=1)
    return confusion_metrics(confusion_matrix(target, prediction, labels=range(len(classes))), classes)


def mean_patient_metrics(values: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    fields = ["accuracy", "f1_micro", "f1_macro", "cohen_kappa", "precision_macro", "recall_macro", "sensitivity_macro"]
    result = {field: float(np.mean([value[field] for value in values])) if values else None for field in fields}
    result["per_class"] = {
        name: {
            field: float(np.mean([value["per_class"][name][field] for value in values])) if values else None
            for field in ["precision", "recall", "sensitivity", "f1"]
        }
        for name in classes
    }
    return result


def task_classes(package: PackagedModel) -> dict[str, list[str]]:
    contract = package.classification_contract
    if contract is None:
        raise TypeError(f"Package '{package.name}' does not contain a classifier.")
    if contract["type"] == "single-head-multiclass":
        if package.task is None:
            raise ValueError("A single-head package must define task.")
        return {package.task: list(contract["classes"])}
    return {task: list(spec["classes"]) for task, spec in contract["tasks"].items()}


def task_resolutions(package: PackagedModel) -> dict[str, pd.Timedelta]:
    contract = package.classification_contract
    if contract is None:
        raise TypeError(f"Package '{package.name}' has no classification contract.")
    if contract["type"] == "single-head-multiclass":
        return {str(package.task): pd.to_timedelta(contract["target_resolution"]) / int(contract["sequence_len"])}
    return {task: pd.to_timedelta(spec["target_resolution"]) for task, spec in contract["tasks"].items()}


def expected_windows_by_patient(dataset) -> dict[str, int]:
    return {str(file.path): int(upper - lower) for file, lower, upper in zip(dataset.edf_files, dataset.lower_bounds, dataset.upper_bounds)}


def predict_package(package: PackagedModel, data: Mapping[str, Any], test: Mapping[str, Any], *, device: str, seed: int, progress_label: str, prediction_output: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, dict], dict[str, pd.Timedelta], list[str]]:
    dataset, patients, _ = prepare_dataset(package, data)
    dataset.initialize(patients, num_workers=int(data.get("num_workers", 4)), strict=bool(data.get("strict", False)))
    if dataset.get_n_patients() == 0 or len(dataset) == 0:
        raise ValueError(f"Package '{package.name}' produced no evaluable windows.")
    expected = expected_windows_by_patient(dataset)
    predictions, received = package.predict_dataset(dataset, batch_size=int(test.get("batch_size", 64)), num_workers=int(test.get("num_workers_dataloader", 0)), n_repeat=1, device=device, seed=seed, rejection_strategy="none", progress=True, progress_label=progress_label, return_received_windows=True)
    if prediction_output is not None:
        write_prediction_feather(prediction_output, predictions)
    coverage = {}
    for patient in patients:
        patient = str(patient)
        expected_count = int(expected.get(patient, 0))
        received_count = int(received.get(patient, 0))
        missing = max(0, expected_count - received_count) if expected_count else 0
        coverage[patient] = {
            "initialized": patient in expected,
            "expected_windows": expected_count,
            "received_windows": received_count,
            "missing_windows": missing,
            "missing_fraction": float(missing / expected_count) if expected_count else 1.0,
        }
    package.model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, coverage, task_resolutions(package), [str(patient) for patient in patients]


def merged_active_intervals(frame: pd.DataFrame, resolution: pd.Timedelta, active_classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    probability_columns = [column for column in frame if column.startswith("prob__")]
    classes = [column.removeprefix("prob__") for column in probability_columns]
    predictions = np.asarray(classes, dtype=object)[frame[probability_columns].to_numpy(dtype=float).argmax(axis=1)]
    active = np.isin(predictions, active_classes)
    starts = frame.loc[active, "time"].astype("int64").to_numpy()
    ends = starts + int(resolution.value)
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    merged_starts = [int(starts[0])]
    merged_ends = [int(ends[0])]
    for start, end in zip(starts[1:], ends[1:]):
        if int(start) <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], int(end))
        else:
            merged_starts.append(int(start))
            merged_ends.append(int(end))
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_ends, dtype=np.int64)


def active_duration_at(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if len(starts) == 0:
        return np.zeros(len(values), dtype=np.int64)
    durations = ends - starts
    prefix = np.concatenate(([0], np.cumsum(durations, dtype=np.int64)))
    completed = np.searchsorted(ends, values, side="right")
    total = prefix[completed].copy()
    current = completed < len(starts)
    current_indices = completed[current]
    partial = np.maximum(0, values[current] - starts[current_indices])
    partial = np.minimum(partial, durations[current_indices])
    total[current] += partial
    return total


def condition_endpoint(child: pd.DataFrame, parent: pd.DataFrame, *, child_resolution: pd.Timedelta, parent_resolution: pd.Timedelta, active_classes: list[str], target_default: str, minimum_coverage: float) -> pd.DataFrame:
    result = child.copy()
    starts, ends = merged_active_intervals(parent, parent_resolution, active_classes)
    child_starts = result["time"].astype("int64").to_numpy()
    child_ends = child_starts + int(child_resolution.value)
    active_duration = active_duration_at(child_ends, starts, ends) - active_duration_at(child_starts, starts, ends)
    active = active_duration >= int(child_resolution.value * minimum_coverage)
    probability_columns = [column for column in result if column.startswith("prob__")]
    default_column = f"prob__{target_default}"
    result.loc[~active, probability_columns] = 0.0
    result.loc[~active, default_column] = 1.0
    return result


def validate_dependencies(specs: list[Mapping[str, Any]], classes_by_task: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    tasks = list(classes_by_task)
    known_tasks = set(tasks)
    normalized = []
    edges = []
    for spec in specs:
        required = {"source", "target", "active_classes", "target_default"}
        unknown_keys = set(spec) - required - {"minimum_coverage"}
        if not required.issubset(spec) or unknown_keys:
            raise ValueError(f"Each dependency must define {sorted(required)} and optional minimum_coverage, got {sorted(spec)}.")
        source, target = str(spec["source"]), str(spec["target"])
        if source not in known_tasks or target not in known_tasks:
            raise ValueError(f"Dependency {source!r}->{target!r} refers to an unknown task.")
        active_classes = list(spec["active_classes"])
        invalid_active = sorted(set(active_classes) - set(classes_by_task[source]))
        if not active_classes or invalid_active:
            raise ValueError(f"Dependency {source!r}->{target!r} has invalid active classes {invalid_active or active_classes}.")
        target_default = str(spec["target_default"])
        if target_default not in classes_by_task[target]:
            raise ValueError(f"Dependency {source!r}->{target!r} default {target_default!r} is not a target class.")
        minimum_coverage = float(spec.get("minimum_coverage", 0.5))
        if not 0 <= minimum_coverage <= 1:
            raise ValueError("Dependency minimum_coverage must be between zero and one.")
        edges.append((source, target))
        normalized.append({"source": source, "target": target, "active_classes": active_classes, "target_default": target_default, "minimum_coverage": minimum_coverage})
    if len(edges) != len(set(edges)):
        raise ValueError("Dependency graph contains duplicate edges.")

    children = {task: [] for task in tasks}
    indegree = {task: 0 for task in tasks}
    for source, target in edges:
        children[source].append(target)
        indegree[target] += 1
    ready = [task for task in tasks if indegree[task] == 0]
    order = []
    while ready:
        task = ready.pop(0)
        order.append(task)
        for child in children[task]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(tasks):
        raise ValueError("Dependency graph must be acyclic.")
    positions = {task: index for index, task in enumerate(order)}
    return sorted(normalized, key=lambda dependency: positions[dependency["target"]])


def apply_dependency_graph(frames: Mapping[str, pd.DataFrame], resolutions: Mapping[str, pd.Timedelta], dependencies: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    result = {task: frame.copy() for task, frame in frames.items()}
    for dependency in dependencies:
        source, target = dependency["source"], dependency["target"]
        result[target] = condition_endpoint(result[target], result[source], child_resolution=resolutions[target], parent_resolution=resolutions[source], active_classes=dependency["active_classes"], target_default=dependency["target_default"], minimum_coverage=dependency["minimum_coverage"])
    return result
