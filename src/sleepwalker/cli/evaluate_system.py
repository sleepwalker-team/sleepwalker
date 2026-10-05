#!/usr/bin/env python3
"""Evaluate a dependency-aware system of packaged classifiers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from sleepwalker.config import build_callback, import_name, read_yaml
from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.deployment.evaluation import apply_dependency_graph, patient_classification_metrics, predict_package, task_classes, validate_dataset_entry, validate_dependencies, write_prediction_feather, write_record
from sleepwalker.prediction_transforms import apply_pipeline
from sleepwalker.telemetry import telemetry_run
from sleepwalker.trainer.Run import seed_everything
from sleepwalker.trainer.utils.disk import json_ready
from sleepwalker.utils import logger


def read_config(path: str | Path) -> dict[str, Any]:
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a top-level mapping.")
    system = config.get("system")
    if not isinstance(system, dict) or not isinstance(system.get("name"), str) or not system["name"]:
        raise ValueError(f"{path} must define a non-empty system.name.")
    if system.get("type") == "independent":
        if not isinstance(system.get("packages"), dict) or not system["packages"]:
            raise ValueError(f"{path} independent systems must define a non-empty package mapping.")
    elif system.get("type") == "package":
        if not isinstance(system.get("package"), str) or not system["package"]:
            raise ValueError(f"{path} package systems must define system.package.")
    elif system.get("type") == "factory":
        if not isinstance(system.get("factory"), str) or not system["factory"]:
            raise ValueError(f"{path} factory systems must define system.factory.")
    else:
        raise ValueError(f"{path} system.type must be independent, package, or factory.")
    if not isinstance(config.get("data"), dict):
        raise ValueError(f"{path} must define one data mapping.")
    validate_dataset_entry(config["data"])
    if not isinstance(config.get("pipeline"), list):
        raise ValueError(f"{path} pipeline must be a list.")
    if not isinstance(config.get("dependencies"), list):
        raise ValueError(f"{path} dependencies must be a list.")
    if not isinstance(config.get("test"), dict) or not str(config["test"].get("output", "")).endswith(".jsonl"):
        raise ValueError(f"{path} test.output must be a JSONL path.")
    prediction_output = config["test"].get("prediction_output")
    if prediction_output is not None and Path(prediction_output).suffix != ".feather":
        raise ValueError(f"{path} test.prediction_output must end in .feather.")
    return config


def load_system_packages(system: Mapping[str, Any]) -> dict[str, PackagedModel]:
    if system["type"] == "independent":
        packages = {name: load_packaged_model(path, map_location="cpu") for name, path in system["packages"].items()}
    elif system["type"] == "package":
        packages = {"system": load_packaged_model(system["package"], map_location="cpu")}
    else:
        factory = import_name(system["factory"])
        packages = factory(system)
        if isinstance(packages, PackagedModel):
            packages = {"system": packages}
    if not isinstance(packages, dict) or not packages or any(not isinstance(package, PackagedModel) for package in packages.values()):
        raise TypeError("A system must resolve to a non-empty mapping of PackagedModel instances.")
    return packages


def system_task_classes(packages: Mapping[str, PackagedModel]) -> dict[str, list[str]]:
    result = {}
    for package in packages.values():
        for task, classes in task_classes(package).items():
            if task in result:
                raise ValueError(f"System packages both predict task {task!r}.")
            result[task] = classes
    return result


def prepare_task_frames(predictions: pd.DataFrame, classes_by_task: Mapping[str, list[str]], pipeline: list, patient: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for task in classes_by_task:
        frame = predictions.loc[predictions["task"] == task].drop(columns=["patient", "task"]).dropna(axis=1, how="all")
        if frame.empty:
            raise ValueError(f"Patient predictions are missing task '{task}'.")
        frame = frame.loc[frame["valid"]].drop(columns="valid").reset_index(drop=True)
        frame = apply_pipeline(frame, pipeline, patient=patient, signals=[])
        if frame.empty:
            raise ValueError(f"Patient has no analyzable predictions for task '{task}'.")
        frames[task] = frame
    return frames


def analyze_patient(predictions: pd.DataFrame, resolutions: Mapping[str, pd.Timedelta], classes_by_task: Mapping[str, list[str]], pipeline: list, dependencies: list[dict[str, Any]], patient: str = "") -> dict[str, dict]:
    frames = prepare_task_frames(predictions, classes_by_task, pipeline, patient)
    frames = apply_dependency_graph(frames, resolutions, dependencies)
    return {
        task: {
            "classes": classes,
            "metrics": patient_classification_metrics(frames[task], classes),
            "n_endpoint_windows": len(frames[task]),
        }
        for task, classes in classes_by_task.items()
    }


def completed_output(output: Path, system_name: str) -> bool:
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [record for record in records if record.get("record_type") == "run"]
    patients = [record for record in records if record.get("record_type") == "patient"]
    patient_ids = {record.get("patient") for record in patients}
    return len(runs) == 1 and runs[0].get("system") == system_name and len(patients) == len(patient_ids) == int(runs[0].get("n_patients_requested", -1))


def execute(config: Mapping[str, Any], config_path: str | Path | None = None) -> Path:
    seed = int(config.get("seed", 17))
    seed_everything(seed)
    output = Path(config["test"]["output"])
    prediction_output = Path(config["test"]["prediction_output"]) if config["test"].get("prediction_output") is not None else None
    if output.exists() and not bool(config["test"].get("overwrite", False)):
        if not completed_output(output, config["system"]["name"]) or prediction_output is not None and not prediction_output.is_file():
            raise ValueError(f"Existing output is not complete for system {config['system']['name']!r}: {output}.")
        print(f"Skipping completed system {config['system']['name']}: {output}", flush=True)
        return output
    if prediction_output is not None and prediction_output.exists() and not bool(config["test"].get("overwrite", False)):
        raise FileExistsError(f"Prediction output already exists: {prediction_output}.")

    with telemetry_run(config["test"].get("telemetry"), output=output.with_suffix(".telemetry"), run_name=config["system"]["name"], tags={"command": "evaluate-system"}, overwrite=bool(config["test"].get("overwrite", False))):
        packages = load_system_packages(config["system"])
        classes_by_task = system_task_classes(packages)
        if "datasets" in config["data"] and set(config["data"]["datasets"]) != set(classes_by_task):
            raise ValueError("data.datasets must name every system task exactly once.")
        dependencies = validate_dependencies(config["dependencies"], classes_by_task)
        pipeline = [build_callback(step) for step in config["pipeline"]]
        device = str(config["test"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        prediction_frames = []
        model_coverages = {}
        resolutions = {}
        requested_patients = None
        prediction_temporary = prediction_output.with_suffix(prediction_output.suffix + ".tmp") if prediction_output is not None else None
        if prediction_temporary is not None:
            prediction_temporary.parent.mkdir(parents=True, exist_ok=True)
            prediction_temporary.unlink(missing_ok=True)
            if bool(config["test"].get("overwrite", False)):
                prediction_output.unlink(missing_ok=True)

        try:
            for model_key, package in packages.items():
                predictions, coverage, package_resolutions, patients = predict_package(package, config["data"], config["test"], device=device, seed=seed, progress_label=f"{config['system']['name']}:{model_key}")
                if requested_patients is None:
                    requested_patients = patients
                elif patients != requested_patients:
                    raise ValueError(f"System package '{model_key}' resolved a different patient cohort.")
                prediction_frames.append(predictions)
                model_coverages[model_key] = coverage
                for task, resolution in package_resolutions.items():
                    if task in resolutions and resolutions[task] != resolution:
                        raise ValueError(f"System packages disagree on the {task!r} resolution.")
                    resolutions[task] = resolution
        except BaseException:
            if prediction_temporary is not None:
                prediction_temporary.unlink(missing_ok=True)
            raise
        if set(resolutions) != set(classes_by_task):
            raise ValueError(f"System resolutions cover {sorted(resolutions)}, expected {sorted(classes_by_task)}.")

        predictions = pd.concat(prediction_frames, ignore_index=True)
        if prediction_temporary is not None:
            write_prediction_feather(prediction_temporary, predictions)
        requested_patients = requested_patients or []
        requested_patient_set = set(requested_patients)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        logger.info(f"Writing dependency-conditioned metrics for {len(requested_patients):,} patients to {temporary}.")
        logger.progress_start(total=len(requested_patients), desc=f"{config['system']['name']} | analysis", leave=True)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                write_record(handle, {"record_type": "run", "system": config["system"]["name"], "coverage_basis": "received_input_windows", "config_path": str(config_path) if config_path is not None else None, "seed": seed, "n_patients_requested": len(requested_patients), "config": json_ready(config)})
                analyzed_patients = set()
                for raw_patient, patient_predictions in predictions.groupby("patient", sort=False, observed=True):
                    patient = str(raw_patient)
                    if patient not in requested_patient_set:
                        raise ValueError(f"Predictions contain unrequested patient {patient!r}.")
                    analyzed_patients.add(patient)
                    logger.progress_status(f"{config['system']['name']} | analysis | patient {len(analyzed_patients)}/{len(requested_patients)} | {patient}")
                    coverage = {model_key: values[patient] for model_key, values in model_coverages.items()}
                    record = {"record_type": "patient", "system": config["system"]["name"], "dataset": config["data"]["label"], "patient": patient, "coverage": coverage, "tasks": {}}
                    if all(values["initialized"] for values in coverage.values()):
                        try:
                            record["tasks"] = analyze_patient(patient_predictions, resolutions, classes_by_task, pipeline, dependencies, patient)
                        except Exception as error:
                            record["analysis_error"] = f"{type(error).__name__}: {error}"
                    write_record(handle, record)
                    logger.progress_advance(1)
                for patient in requested_patients:
                    if patient in analyzed_patients:
                        continue
                    logger.progress_status(f"{config['system']['name']} | analysis | patient {len(analyzed_patients) + 1}/{len(requested_patients)} | {patient} | no predictions")
                    coverage = {model_key: values[patient] for model_key, values in model_coverages.items()}
                    write_record(handle, {"record_type": "patient", "system": config["system"]["name"], "dataset": config["data"]["label"], "patient": patient, "coverage": coverage, "tasks": {}})
                    analyzed_patients.add(patient)
                    logger.progress_advance(1)
            os.replace(temporary, output)
            if prediction_temporary is not None:
                os.replace(prediction_temporary, prediction_output)
            logger.info(f"Wrote evaluation file {output}.")
        except BaseException:
            temporary.unlink(missing_ok=True)
            if prediction_temporary is not None:
                prediction_temporary.unlink(missing_ok=True)
            raise
        finally:
            logger.progress_close()
        return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    configs = parser.add_mutually_exclusive_group(required=True)
    configs.add_argument("config", nargs="?", help="One system-evaluation YAML file.")
    configs.add_argument("--config", dest="config_option", help="Evaluation YAML supplied by an external runtime.")
    args = parser.parse_args(argv)
    config_path = args.config or args.config_option
    config = read_config(config_path)
    if args.overwrite:
        config["test"]["overwrite"] = True
    execute(config, config_path)


if __name__ == "__main__":
    main()
