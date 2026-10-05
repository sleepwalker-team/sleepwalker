#!/usr/bin/env python3
"""Evaluate a packaged classifier on labelled EDF manifests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sleepwalker-matplotlib")

import numpy as np
import pandas as pd
import torch

from sleepwalker.config import build_callback, read_yaml
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.deployment.evaluation import confusion_metrics, mean_patient_metrics, package_fold, patient_classification_metrics, prepare_dataset, task_classes, validate_dataset_entry, write_record
from sleepwalker.prediction_transforms import apply_pipeline
from sleepwalker.telemetry import telemetry_run
from sleepwalker.trainer.Run import seed_everything
from sleepwalker.trainer.utils.disk import json_ready
from sleepwalker.utils import logger


def read_config(path: str | Path) -> dict[str, Any]:
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    if not isinstance(config["data"], (dict, list)) or not config["data"]:
        raise ValueError("data must be a dataset mapping or list of mappings.")
    for entry in config["data"] if isinstance(config["data"], list) else [config["data"]]:
        validate_dataset_entry(entry)
    if not isinstance(config["analyses"], list) or not config["analyses"]:
        raise ValueError("analyses must be a non-empty list.")
    if not str(config["test"]["output"]).endswith(".jsonl"):
        raise ValueError("test.output must end in .jsonl.")
    if not isinstance(config["test"].get("package"), str) or not config["test"]["package"]:
        raise ValueError("test.package must name the trained package directory.")
    return config


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


def evaluate_predictions(handle, package: PackagedModel, dataset, predictions: pd.DataFrame, analyses: list[tuple[str, list[Any]]], *, dataset_label: str, fold: str | None, n_patients_requested: int) -> None:
    classes_by_task = task_classes(package)
    aggregate_matrices = {(analysis_name, task): np.zeros((len(classes), len(classes)), dtype=np.int64) for analysis_name, _ in analyses for task, classes in classes_by_task.items()}
    patient_metrics = {(analysis_name, task): [] for analysis_name, _ in analyses for task in classes_by_task}
    patient_matrices = {(analysis_name, task): {} for analysis_name, _ in analyses for task in classes_by_task}
    signal_map = {str(file.path): list(read_edf_meta(file.path)["signals"]) for file in dataset.edf_files}
    required = {"patient", "task", "valid"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing evaluation columns {missing}.")

    grouped = predictions.groupby(["patient", "task"], sort=False)
    patient_order = list(dict.fromkeys(predictions["patient"].astype(str)))
    patient_indices = {patient: index for index, patient in enumerate(patient_order, start=1)}
    total_steps = grouped.ngroups * len(analyses)
    logger.progress_start(total=total_steps, desc=f"{dataset_label} | analysis", leave=True)
    try:
        for (raw_patient, raw_task), patient_frame in grouped:
            patient = str(raw_patient)
            task = str(raw_task)
            if task not in classes_by_task:
                raise ValueError(f"Prediction table contains unknown task '{task}'.")
            if patient not in signal_map:
                raise ValueError(f"Prediction table contains unknown patient '{patient}'.")
            original_rows = len(patient_frame)
            frame = patient_frame.drop(columns=["patient", "task"]).dropna(axis=1, how="all")
            frame = frame.loc[frame["valid"]].drop(columns="valid").reset_index(drop=True)
            for analysis_name, pipeline in analyses:
                analyzed = apply_pipeline(frame, pipeline, patient=patient, signals=signal_map[patient])
                logger.progress_status(f"{dataset_label} | analysis | patient {patient_indices[patient]}/{len(patient_order)} | {analysis_name} | {task} | kept {len(analyzed):,}/{original_rows:,}")
                logger.progress_advance(1)
                if analyzed.empty:
                    continue
                classes = classes_by_task[task]
                metrics = patient_classification_metrics(analyzed, classes)
                key = (analysis_name, task)
                matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
                aggregate_matrices[key] += matrix
                patient_metrics[key].append(metrics)
                patient_matrices[key][patient] = metrics["confusion_matrix"]
                write_record(handle, {
                    "record_type": "patient",
                    "package": package.name,
                    "task": task,
                    "dataset": dataset_label,
                    "fold": fold,
                    "analysis": analysis_name,
                    "patient": patient,
                    "classes": classes,
                    "metrics": metrics,
                })
    finally:
        logger.progress_close()

    for analysis_name, _ in analyses:
        for task, classes in classes_by_task.items():
            key = (analysis_name, task)
            metrics = confusion_metrics(aggregate_matrices[key], classes)
            logger.info(f"{dataset_label}/{analysis_name}/{task}: {metrics['n_windows']} windows, accuracy {metrics['accuracy']:.4f}, macro F1 {metrics['f1_macro']:.4f}, kappa {metrics['cohen_kappa']:.4f}.")
            write_record(handle, {
                "record_type": "aggregate",
                "package": package.name,
                "task": task,
                "dataset": dataset_label,
                "fold": fold,
                "analysis": analysis_name,
                "classes": classes,
                "n_patients_requested": n_patients_requested,
                "n_patients_initialized": dataset.get_n_patients(),
                "n_patients_evaluated": len(patient_metrics[key]),
                "metrics": metrics,
                "patient_mean_metrics": mean_patient_metrics(patient_metrics[key], classes),
                "patient_confusion_matrices": patient_matrices[key],
            })


def execute(package_path: str | Path | PackagedModel, config: Mapping[str, Any]) -> Path:
    seed = int(config.get("seed", 17))
    seed_everything(seed)
    test_options = config["test"]
    output = Path(test_options["output"])
    if output.exists() and not bool(test_options.get("overwrite", False)):
        raise FileExistsError(f"Output already exists: {output}. Set test.overwrite: true to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    device = str(test_options.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    with telemetry_run(test_options.get("telemetry"), output=output.with_suffix(".telemetry"), run_name=output.stem, tags={"command": "evaluate"}, overwrite=bool(test_options.get("overwrite", False))):
        package = package_path if isinstance(package_path, PackagedModel) else load_packaged_model(package_path, map_location=device)
        package_reference = package.name if isinstance(package_path, PackagedModel) else str(package_path)
        analyses = build_analyses(config["analyses"])
        entries = config["data"] if isinstance(config["data"], list) else [config["data"]]
        logger.info(f"Evaluating package '{package.name}' on {device}: {len(entries)} dataset(s), {len(analyses)} analysis pipeline(s), output {output}.")

        try:
            with temporary.open("w", encoding="utf-8") as handle:
                write_record(handle, {
                    "record_type": "run",
                    "package": package.name,
                    "package_path": package_reference,
                    "task": package.task,
                    "classification_contract": json_ready(package.classification_contract),
                    "git_commit": package.git_commit,
                    "fold": package_fold(package),
                    "seed": seed,
                    "config": json_ready(config),
                })
                for entry in entries:
                    dataset, patients, selected_fold = prepare_dataset(package, entry)
                    label = str(entry["label"])
                    logger.context(label)
                    try:
                        logger.info(f"Initializing {len(patients)} requested patients with rejection_strategy='none'.")
                        dataset.initialize(patients, num_workers=int(entry.get("num_workers", 4)), strict=bool(entry.get("strict", True)))
                        if dataset.get_n_patients() == 0 or len(dataset) == 0:
                            raise ValueError(f"Dataset '{label}' produced no evaluable windows.")
                        logger.info(f"Initialized {dataset.get_n_patients()} patients and {len(dataset)} evaluation windows.")
                        predictions = package.predict_dataset(dataset, batch_size=int(test_options.get("batch_size", 64)), num_workers=int(test_options.get("num_workers_dataloader", 0)), n_repeat=int(test_options.get("n_repeat", 1)), device=device, seed=seed, rejection_strategy="none", allow_preprocessing_override=True, progress=True, progress_label=label)
                        if predictions.empty:
                            raise ValueError(f"Dataset '{label}' produced no valid prediction rows.")
                        logger.info(f"Produced {len(predictions):,} prediction rows for {predictions['patient'].nunique() if not predictions.empty else 0} patients.")
                        evaluate_predictions(handle, package, dataset, predictions, analyses, dataset_label=label, fold=selected_fold, n_patients_requested=len(patients))
                    finally:
                        logger.uncontext()
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        logger.info(f"Evaluation complete: {output}.")
        return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    configs = parser.add_mutually_exclusive_group(required=True)
    configs.add_argument("config", nargs="?", help="Evaluation YAML under configs/test/.")
    configs.add_argument("--config", dest="config_option", help="Evaluation YAML supplied by an external runtime.")
    args = parser.parse_args(argv)
    config = read_config(args.config or args.config_option)
    if args.overwrite:
        config["test"]["overwrite"] = True
    execute(config["test"]["package"], config)


if __name__ == "__main__":
    main()
