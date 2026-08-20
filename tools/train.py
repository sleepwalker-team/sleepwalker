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
          name: sleepwalker.datasets.HSP.get_annotated_hsp_edf_files
          root: /raid/sleepwalker/hsp
        validation_fraction: 0.1
        test_fraction: 0.1
      patient_filter:
        - name: sleepwalker.training.files.filter_edf_files
          min_duration: 30min
        - name: sleepwalker.datasets.HSP.filter_hsp_annotation_labels
          required_labels: [n1, n2, n3, rem]

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

``data`` may be a list for multi-dataset training. Nested components such as
``ModelGraphClassifier`` nodes use the same fully qualified ``name`` syntax.

``files`` either names selectors for a newly generated split, as above, or is
the path to a precomputed YAML split.  ``tools/split.py`` creates holdout and
cross-validation manifests in this shape::

    folds:
      holdout:
        train: [/data/sub-1.edf]
        validation: [/data/sub-2.edf]
        test: [/data/sub-3.edf]

An optional ``patient_filter`` narrows each base partition for the current
experiment before dataset initialization without reassigning any patient. Use
``--fold fold_0`` to train one fold from a cross-validation manifest.

Callbacks remain ordinary Python functions.  ``prepare_patient``,
``prepare_target``, and ``prepare_sample`` are bound with ``functools.partial``;
optimizers and schedulers are delayed until their dependency exists.

Supported: standard RunCfg runs, custom filtering selectors, preparation
callbacks, multiple datasets, nested/composite models, MulticlassTrainer and
MultiLabelTrainer configurations.  Not supported: custom control flow,
multi-stage experiments, result aggregation, or defining Python expressions in
YAML.  Use a normal training script for those cases.

The ``dry`` command is the main validation path.  It performs a real one-epoch run
using at most 30 training patients and 10 validation/test patients (shared
across data entries), with small sample budgets and no MLflow logging.

Use ``train CONFIG`` for a new run and ``resume CHECKPOINT`` to reconstruct a
run from the checkpoint and the ``hparams.yml`` stored at the run root.
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
import importlib
import inspect
import os
from pathlib import Path
import re
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
from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.Run import RunCfg, run, seed_everything
from sleepwalker.trainer.utils.splits import load_split
from sleepwalker.utils import logger

#TODO: Sometimes delayed output -> review
DATA_FIELDS = {
    "files", #TODO: REVIEW THIS PART
    "num_workers",
    "strict",
    "label",
    "output_classes",
    "patient_filter",
}
CONTEXT_FIELDS = {"classes", "input_channels", "n_channels", "ts_len", "sampling_frequency", "sequence_len"}
DRY_RUN_PATIENTS = {"train": 30, "validation": 10, "test": 10}


class ScientificNotationLoader(yaml.SafeLoader):
    """Safe YAML loader that recognizes exponent notation without a decimal point."""


ScientificNotationLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=ScientificNotationLoader) or {}


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
        if isinstance(value.get("name"), str) and "." in value["name"]:
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
        if key in {"classes", "sequence_len"} and "task_config" in arguments:
            continue
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
    arguments = { key: build_value(value) for key, value in spec.items() if key != "name"}
    if dependency == "model":
        return lambda model: symbol(model.parameters(), **arguments)
    return partial(symbol, **arguments)

def build_channel(spec: Mapping[str, Any]) -> ChannelConfig:
    arguments = dict(spec)
    if "normalizer" in arguments:
        arguments["normalizer"] = build_value(arguments["normalizer"])
    return ChannelConfig(**arguments)

def build_dataset(entry: Mapping[str, Any]):
    """Build the dataset shown in one ``data`` entry."""
    spec = {key: value for key, value in entry.items() if key not in DATA_FIELDS}
    if "channels" in spec:
        spec["channels"] = [build_channel(channel) for channel in spec["channels"]]
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
    if "dataset" in inspect.signature(selector).parameters:
        arguments["dataset"] = dataset
    return [str(path) for path in selector(**arguments)]


def apply_patient_filter(
    spec: str | Mapping[str, Any] | list,
    patients: list[str],
    dataset,
    num_workers: int = 1,
) -> list[str]:
    """Apply configured filters in order to paths already assigned to a split."""
    filters = spec if isinstance(spec, list) else [spec]
    filtered = list(patients)
    for filter_spec in filters:
        if isinstance(filter_spec, str):
            patient_filter = import_name(filter_spec)
            arguments = {}
        else:
            patient_filter = import_name(str(filter_spec["name"]))
            arguments = {key: build_value(value) for key, value in filter_spec.items() if key != "name"}
        parameters = inspect.signature(patient_filter).parameters
        if "dataset" in parameters:
            arguments["dataset"] = dataset
        if "num_workers" in parameters and "num_workers" not in arguments:
            arguments["num_workers"] = num_workers
        current = [str(path) for path in patient_filter(patients=filtered, **arguments)]
        unexpected = sorted(set(current) - set(filtered))
        if unexpected:
            raise ValueError(f"A patient filter may only remove paths from its existing partition; it added {unexpected[:5]}.")
        if len(current) != len(set(current)):
            raise ValueError("A patient filter returned duplicate paths.")
        filtered = current
    return filtered


def load_patient_split(files: str | Mapping[str, Any], dataset, seed: int, fold: str | None = None) -> dict[str, list[str]]:
    """Load an authoritative split or visibly create one from selectors."""
    if isinstance(files, str):
        return load_split(files, fold=fold)

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


def initialize_datasets(config: Mapping[str, Any], dry_run: bool = False, fold: str | None = None):
    """Load patient paths and initialize every train/validation/test dataset."""
    entries = config["data"] if isinstance(config["data"], list) else [config["data"]]
    seed = int(config.get("seed", 17))
    train_datasets, validation_datasets, test_datasets = [], [], []

    for entry in entries:
        template = build_dataset(entry)
        split = load_patient_split(entry["files"], template, seed, fold=fold)
        workers = (
            min(int(entry.get("num_workers", 4)), 2)
            if dry_run
            else int(entry.get("num_workers", 4))
        )
        patients_per_entry = {
            role: max(1, limit // len(entries))
            for role, limit in DRY_RUN_PATIENTS.items()
        }

        for role, patients in split.items():
            dataset = build_dataset(entry)
            if entry.get("patient_filter") is not None:
                filtered = apply_patient_filter(
                    entry["patient_filter"],
                    patients,
                    dataset,
                    num_workers=workers,
                )
                logger.info(
                    f"Configured patient filter for {role} kept "
                    f"{len(filtered)}/{len(patients)} paths."
                )
                patients = filtered
            if dry_run:
                patients = patients[: patients_per_entry[role]]
            if not patients:
                continue
            dataset.initialize(patients, workers, strict=bool(entry.get("strict", False)))
            if role == "train":
                train_datasets.append(dataset)
            elif role == "validation":
                validation_datasets.append(dataset)
            elif role == "test":
                label = str(entry.get("label", dataset.__class__.__name__))
                test_datasets.append((label, dataset))

    return train_datasets, validation_datasets, test_datasets


def run_options_from_dict(config: Mapping[str, Any], dry_run: bool, fold: str | None) -> dict[str, Any]:
    run_options = copy.deepcopy(config["run"])
    if fold is not None:
        run_options["experiment_name"] = f"{run_options['experiment_name']}-{fold}"
        run_options.setdefault("tags", {})["fold"] = fold
    run_options.setdefault("model_name", str(config["model"]["name"]).split(".")[-1])
    run_options["collate_fn"] = build_callback(run_options.get("collate_fn")) or batch_collate
    metadata = copy.deepcopy(dict(config))
    if fold is not None:
        metadata["fold"] = fold
    run_options["meta_data"] = metadata
    if dry_run:
        run_options["experiment_name"] = f"{run_options['experiment_name']}-dry-run"
        run_options["batch_size"] = min(int(run_options["batch_size"]), 16)
        run_options["n_samples"] = min(int(run_options.get("n_samples") or 512), 512)
        run_options["n_samples_test"] = min(int(run_options.get("n_samples_test") or 256), 256)
        run_options["num_workers_dataloader"] = 0
        run_options["test_repeats"] = [1]
        run_options["use_mlflow"] = False
        run_options["log_path"] = tempfile.mkdtemp(prefix="sleepwalker-dry-run-")
    return run_options


def runcfg_from_dict(config: Mapping[str, Any], dry_run: bool = False, fold: str | None = None) -> RunCfg:
    """Build a new model, trainer, datasets, and ``RunCfg`` from configuration."""
    selected_fold = fold if fold is not None else config.get("fold")
    seed_everything(int(config.get("seed", 17)))
    train_datasets, validation_datasets, test_datasets = initialize_datasets(config, dry_run=dry_run, fold=selected_fold)
    training_dataset = combine_datasets(train_datasets)
    input_channels = training_dataset.get_input_channels()
    sequence_len = int(config["trainer"].get("sequence_len", 1))
    context = {
        "classes": training_dataset.get_classes(),
        "input_channels": input_channels,
        "n_channels": len(input_channels),
        "ts_len": training_dataset.get_timeseries_len(),
        "sampling_frequency": training_dataset.sample_frequency,
        "sequence_len": sequence_len,
    }

    model = build_component(config["model"], context)
    validate_model_input(model, training_dataset)
    trainer_spec = copy.deepcopy(config["trainer"])
    trainer_spec["optimizer"] = build_factory(trainer_spec["optimizer"], "model")
    if trainer_spec.get("lr_scheduler") is not None:
        trainer_spec["lr_scheduler"] = build_factory(trainer_spec["lr_scheduler"], "optimizer")
    if "loss_function" in trainer_spec:
        trainer_spec["loss_function"] = build_value(trainer_spec["loss_function"])
    if dry_run:
        trainer_spec["epochs"] = 1
        trainer_spec["eval_every"] = 1
        trainer_spec["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        trainer_spec["warmup_device"] = "cpu"
    trainer = build_component(trainer_spec, context)
    run_options = run_options_from_dict(config, dry_run, selected_fold)

    return RunCfg(
        **run_options,
        model=model,
        trainer=trainer,
        train_datasets=train_datasets,
        val_datasets=validation_datasets,
        test_datasets=test_datasets,
    )


def validate_model_input(model, dataset) -> None:
    """Validate model input metadata against the dataset that supplies it."""

    shape, meta = model.input_spec()
    expected_shape = (1, dataset.get_timeseries_len(), len(dataset.get_input_channels()))
    if tuple(shape) != expected_shape:
        raise ValueError(f"Model input {tuple(shape)} does not match training dataset input {expected_shape}.")
    if meta.get("layout") != "BTC":
        raise ValueError(f"Training models must use BTC inputs, got {meta.get('layout')!r}.")
    if "input_channels" in meta and list(meta["input_channels"]) != list(dataset.get_input_channels()):
        raise ValueError(f"Model expects input channels {list(meta['input_channels'])}, got {list(dataset.get_input_channels())}.")
    if "sampling_frequency" in meta and float(meta["sampling_frequency"]) != float(dataset.sample_frequency):
        raise ValueError(f"Model expects sampling_frequency={meta['sampling_frequency']}, got {dataset.sample_frequency}.")


def runcfg_from_checkpoint(path: str | Path) -> RunCfg:
    """Restore trainer state and rebuild datasets from the run's serialized configuration."""
    checkpoint_path = Path(path)
    trainer, model = BaseTrainer.load_checkpoint(checkpoint_path)
    config_path = checkpoint_path.parent.parent / "hparams.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected the run configuration at {config_path}.")
    config = read_yaml(config_path)
    missing = [key for key in ("data", "model", "trainer", "run") if key not in config]
    if missing:
        raise ValueError(f"The checkpoint configuration is missing: {', '.join(missing)}.")

    fold = config.get("fold")
    seed_everything(int(config.get("seed", 17)))
    train_datasets, validation_datasets, test_datasets = initialize_datasets(config, fold=fold)
    run_options = run_options_from_dict(config, False, fold)
    return RunCfg(
        **run_options,
        model=model,
        trainer=trainer,
        train_datasets=train_datasets,
        val_datasets=validation_datasets,
        test_datasets=test_datasets,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Start a new training run from YAML.")
    train_parser.add_argument("config", help="Training YAML under configs/.")
    train_parser.add_argument("--fold", help="Fold name in a cross-validation manifest.")

    resume_parser = commands.add_parser("resume", help="Resume the run embedded in a training checkpoint.")
    resume_parser.add_argument("checkpoint", help="Training checkpoint file.")

    dry_parser = commands.add_parser("dry", help="Run one epoch with small patient and sample budgets.")
    dry_parser.add_argument("config", help="Training YAML under configs/.")
    dry_parser.add_argument("--fold", help="Fold name in a cross-validation manifest.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "resume":
        run(runcfg_from_checkpoint(args.checkpoint))
        return
    run(runcfg_from_dict(read_yaml(args.config), dry_run=args.command == "dry", fold=args.fold))


if __name__ == "__main__":
    main()
