#!/usr/bin/env python3
"""Evaluate a packaged classifier on labelled EDF manifests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sleepwalker-matplotlib")

import torch

from sleepwalker.config import build_callback, read_yaml
from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.deployment.evaluation import evaluate_predictions, package_fold, prepare_dataset, validate_dataset_entry, write_record
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
                        predictions = package.predict_dataset(dataset, batch_size=int(test_options.get("batch_size", 64)), num_workers=int(test_options.get("num_workers_dataloader", 0)), n_repeat=int(test_options.get("n_repeat", 1)), device=device, seed=seed, rejection_strategy="none", progress=True, progress_label=label)
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
