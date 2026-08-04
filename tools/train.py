#!/usr/bin/env python3
"""Run a standard Sleepwalker training configuration.

This tool is a small YAML adapter around the normal Python training API.  The
canonical example is ``configs/train/desaturation_hsp.yml``.  Components use a
fully qualified ``name`` and put constructor arguments directly beside it::

    data:
      name: sleepwalker.datasets.HSP.HSP
      channels:
        - logical_name: SpO2
          physical_names: [SaO2, SpO2, SPO2]
          normalizer:
            name: sleepwalker.datasets.normalizer.SaturationFilterNormalizer.SaturationFilterNormalizer
            fs: 100
      files:
        train:
          name: sleepwalker.training.files.select_edf_files
          root: /raid/sleepwalker/hsp
        validation_fraction: 0.1
        test_fraction: 0.1

    model:
      name: sleepwalker.models.UTime.UTime

    trainer:
      name: sleepwalker.trainer.MulticlassTrainer.MulticlassTrainer
      epochs: 30
      optimizer:
        name: torch.optim.Adam
        lr: 0.001

    run:
      experiment_name: desaturation_hsp
      batch_size: 128
      n_samples: 250000
      num_workers_dataloader: 8

``data`` may be a list for multi-dataset training.  Composite models are
ordinary nested named components; for example their entries name
``sleepwalker.models.MultiModel.MetaModelEntry`` explicitly.

``files`` either names selectors for a newly generated split, as above, or is
the path to a precomputed YAML split.  A precomputed split has exactly the
patient paths consumed by the run and needs only this shape::

    train: [/data/sub-1.edf]
    validation: [/data/sub-2.edf]
    test: [/data/sub-3.edf]

Callbacks remain ordinary Python functions.  ``prepare_patient``,
``prepare_target``, and ``prepare_sample`` are bound with ``functools.partial``;
optimizers and schedulers are delayed until their dependency exists.

Supported: standard RunCfg runs, custom filtering selectors, preparation
callbacks, multiple datasets, nested/composite models, MulticlassTrainer and
MultiLabelTrainer configurations.  Not supported: custom control flow,
multi-stage experiments, result aggregation, or defining Python expressions in
YAML.  Use a normal training script for those cases.

``--dry-run`` is the main validation path.  It performs a real one-epoch run
using at most 30 training patients and 10 validation/test patients (shared
across data entries), with small sample budgets and no MLflow logging.
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
import importlib
import inspect
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sleepwalker-matplotlib")

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.MultiDataset import combine_datasets
from sleepwalker.datasets.utils import random_split
from sleepwalker.trainer.Run import RunCfg, RunResult, run, seed_everything


DATA_FIELDS = {
    "files",
    "num_workers",
    "strict",
    "label",
    "inference",
    "output_classes",
    "split_overrides",
}
CONTEXT_FIELDS = {"classes", "input_channels", "n_channels", "ts_len", "sampling_frequency"}
DRY_RUN_PATIENTS = {"train": 30, "validation": 10, "test": 10}


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def import_name(name: str) -> Any:
    """Import a fully qualified module, class, method, or function name."""
    parts = name.split(".")
    for boundary in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:boundary]))
        except ModuleNotFoundError:
            continue
        for part in parts[boundary:]:
            value = getattr(value, part)
        return value
    raise ImportError(name)


def build_value(value: Any, context: Mapping[str, Any] | None = None) -> Any:
    """Build nested named components and import qualified callable values."""
    context = {} if context is None else context
    if isinstance(value, list):
        return [build_value(item, context) for item in value]
    if isinstance(value, dict):
        if "name" in value:
            return build_component(value, context)
        return {key: build_value(item, context) for key, item in value.items()}
    if isinstance(value, str) and value.startswith(("sleepwalker.", "torch.")):
        return import_name(value)
    return value


def build_component(spec: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> Any:
    """Instantiate one ``name`` component, injecting standard dataset context."""
    context = {} if context is None else dict(context)
    symbol = import_name(str(spec["name"]))
    raw_arguments = {key: value for key, value in spec.items() if key != "name"}

    local_context = dict(context)
    if isinstance(raw_arguments.get("input_channels"), list):
        local_context["input_channels"] = raw_arguments["input_channels"]
        local_context["n_channels"] = len(raw_arguments["input_channels"])
    arguments = {
        key: build_value(value, local_context)
        for key, value in raw_arguments.items()
    }

    parameters = inspect.signature(symbol).parameters
    for key in CONTEXT_FIELDS:
        if key in parameters and key in local_context and key not in arguments:
            arguments[key] = local_context[key]
    return symbol(**arguments)


def build_callback(spec: str | Mapping[str, Any] | None) -> Any:
    """Bind a dataset or RunCfg callback without calling it."""
    if spec is None or callable(spec):
        return spec
    if isinstance(spec, str):
        return import_name(spec)
    symbol = import_name(str(spec["name"]))
    arguments = {
        key: build_value(value)
        for key, value in spec.items()
        if key != "name"
    }
    return partial(symbol, **arguments) if arguments else symbol


def build_factory(spec: Mapping[str, Any], dependency: str) -> Any:
    """Delay optimizer/scheduler construction until its dependency exists."""
    symbol = import_name(str(spec["name"]))
    arguments = {
        key: build_value(value)
        for key, value in spec.items()
        if key != "name"
    }
    if dependency == "model":
        return lambda model: symbol(model.parameters(), **arguments)
    return lambda optimizer: symbol(optimizer, **arguments)


def build_channel(spec: Mapping[str, Any]) -> ChannelConfig:
    arguments = dict(spec)
    if "normalizer" in arguments:
        arguments["normalizer"] = build_value(arguments["normalizer"])
    return ChannelConfig(**arguments)


def build_dataset(entry: Mapping[str, Any], role: str | None = None):
    """Build the dataset shown in one ``data`` entry."""
    spec = {key: value for key, value in entry.items() if key not in DATA_FIELDS}
    spec.update(entry.get("split_overrides", {}).get(role, {}))
    spec["channels"] = [build_channel(channel) for channel in spec.get("channels", [])]
    for callback_name in ("prepare_patient", "prepare_target", "prepare_sample"):
        if callback_name in spec:
            spec[callback_name] = build_callback(spec[callback_name])
    dataset = build_component(spec)
    if "output_classes" in entry:
        dataset.classes = list(entry["output_classes"])
    return dataset


def select_patients(spec: Any, dataset) -> list[str]:
    """Resolve one explicit patient list or call one configured selector."""
    if isinstance(spec, list):
        return [str(path) for path in spec]
    selector = import_name(str(spec["name"]))
    arguments = {
        key: build_value(value)
        for key, value in spec.items()
        if key != "name"
    }
    return [str(path) for path in selector(dataset=dataset, **arguments)]


def load_patient_split(files: str | Mapping[str, Any], dataset, seed: int) -> dict[str, list[str]]:
    """Load an authoritative split or visibly create one from selectors."""
    if isinstance(files, str):
        split = read_yaml(files)
        return {role: [str(path) for path in split[role]] for role in DRY_RUN_PATIENTS}

    split_options = dict(files)
    validation_fraction = float(split_options.pop("validation_fraction", 0))
    test_fraction = float(split_options.pop("test_fraction", 0))
    patients = {
        role: select_patients(spec, dataset)
        for role, spec in split_options.items()
    }
    if validation_fraction and "validation" not in patients:
        patients["train"], patients["validation"] = random_split(
            patients["train"], validation_fraction, seed=seed
        )
    if test_fraction and "test" not in patients:
        patients["train"], patients["test"] = random_split(
            patients["train"], test_fraction, seed=seed + 1
        )
    return patients


def initialize_datasets(config: Mapping[str, Any], dry_run: bool = False):
    """Load patient paths and initialize every train/validation/test dataset."""
    entries = config["data"] if isinstance(config["data"], list) else [config["data"]]
    seed = int(config.get("seed", 17))
    train_datasets, validation_datasets, test_datasets = [], [], []
    inference_dataset = None

    for entry in entries:
        template = build_dataset(entry)
        split = load_patient_split(entry["files"], template, seed)
        workers = min(int(entry.get("num_workers", 4)), 2) if dry_run else int(entry.get("num_workers", 4))
        patients_per_entry = {
            role: max(1, limit // len(entries))
            for role, limit in DRY_RUN_PATIENTS.items()
        }

        for role, patients in split.items():
            if dry_run:
                patients = patients[: patients_per_entry[role]]
            if not patients:
                continue
            dataset = build_dataset(entry, role)
            dataset.initialize(patients, workers, strict=bool(entry.get("strict", False)))
            if role == "train":
                train_datasets.append(dataset)
                if inference_dataset is None or entry.get("inference", False):
                    inference_dataset = build_dataset(entry, "inference")
            elif role == "validation":
                validation_datasets.append(dataset)
            elif role == "test":
                label = str(entry.get("label", dataset.__class__.__name__))
                test_datasets.append((label, dataset))

    return train_datasets, validation_datasets, test_datasets, inference_dataset


def build_run_config(config: Mapping[str, Any], dry_run: bool = False) -> RunCfg:
    """Build datasets, model, trainer, and finally the normal API ``RunCfg``."""
    seed_everything(int(config.get("seed", 17)))
    train_datasets, validation_datasets, test_datasets, inference_dataset = initialize_datasets(
        config, dry_run
    )
    training_dataset = combine_datasets(train_datasets)
    input_channels = training_dataset.get_input_channels()
    context = {
        "classes": training_dataset.get_classes(),
        "input_channels": input_channels,
        "n_channels": len(input_channels),
        "ts_len": training_dataset.get_timeseries_len(),
        "sampling_frequency": training_dataset.sample_frequency,
    }

    model = build_component(config["model"], context)
    trainer_spec = copy.deepcopy(config["trainer"])
    trainer_spec["optimizer"] = build_factory(trainer_spec["optimizer"], "model")
    if trainer_spec.get("lr_scheduler") is not None:
        trainer_spec["lr_scheduler"] = build_factory(trainer_spec["lr_scheduler"], "optimizer")
    if "loss_function" in trainer_spec:
        trainer_spec["loss_function"] = build_value(trainer_spec["loss_function"])
    if dry_run:
        trainer_spec["epochs"] = 1
        trainer_spec["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        trainer_spec["warmup_device"] = "cpu"
    trainer = build_component(trainer_spec, context)

    run_options = copy.deepcopy(config["run"])
    run_options.setdefault("model_name", str(config["model"]["name"]).split(".")[-1])
    run_options["collate_fn"] = build_callback(run_options.get("collate_fn")) or batch_collate
    run_options.setdefault("meta_data", copy.deepcopy(dict(config)))
    if dry_run:
        run_options["experiment_name"] = f"{run_options['experiment_name']}-dry-run"
        run_options["batch_size"] = min(int(run_options["batch_size"]), 16)
        run_options["n_samples"] = min(int(run_options.get("n_samples") or 512), 512)
        run_options["n_samples_test"] = min(int(run_options.get("n_samples_test") or 256), 256)
        run_options["num_workers_dataloader"] = 0
        run_options["test_repeats"] = [1]
        run_options["use_mlflow"] = False
        run_options["log_path"] = tempfile.mkdtemp(prefix="sleepwalker-dry-run-")

    return RunCfg(
        **run_options,
        model=model,
        trainer=trainer,
        train_datasets=train_datasets,
        val_datasets=validation_datasets,
        test_datasets=test_datasets,
        inference_dataset=inference_dataset,
    )


def execute(config: Mapping[str, Any], dry_run: bool = False) -> RunResult:
    return run(build_run_config(config, dry_run))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Training YAML under configs/.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train for one epoch on a small patient and sample budget.",
    )
    args = parser.parse_args()
    execute(read_yaml(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
