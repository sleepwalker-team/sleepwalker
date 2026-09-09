"""Shared experiment runner used by task-specific training scripts.

The training scripts in the repository prepare datasets, models, trainers, and
experiment metadata, then hand the assembled configuration to ``run(...)``.
This module centralizes loader construction, training invocation, and canonical
per-experiment test-result artifacts.
"""

import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torchinfo import summary

from sleepwalker.deployment import save_packaged_model
from sleepwalker.models.ModelGraphClassifier import ModelGraphClassifier
from sleepwalker.trainer.utils.disk import write_json
from sleepwalker.datasets.MultiDataset import combine_datasets
from sleepwalker.training.execution import RepeatedViewModel
from sleepwalker.training.loader import build_loader
from sleepwalker.utils import LocalArtifactSink, MlflowSink, logger


@dataclass
class RunCfg:
    """Describe one fully prepared training run.

    ``RunCfg`` is the boundary between task-specific experiment assembly and
    the generic execution logic in :func:`run`. Training scripts are expected
    to fully decide dataset splits, model architecture, trainer behavior, and
    deployment/export metadata before constructing this object. The runner then
    handles loader construction, training, logging, evaluation, and expert
    package wiring in a uniform way.

    Training scripts fully assemble datasets, model, trainer, and run config.
    The runner only coordinates loaders, fitting, evaluation, logging, and the
    final expert export.
    """

    experiment_name: str
    model_name: str
    model: Any
    trainer: Any
    train_datasets: list[Any]
    val_datasets: list[Any]
    test_datasets: list[tuple[str, Any]]
    batch_size: int
    n_samples: int | None
    num_workers_dataloader: int
    n_samples_test: int | None = None
    patients_per_epoch: int | None = None
    patient_group_size: int | None = None
    test_repeats: list[int] = field(default_factory=lambda: [1])
    use_mlflow: bool = False
    log_path: str = "sleepwalker"
    tags: dict[str, str] = field(default_factory=dict)
    collate_fn: Any = None
    meta_data: dict[str, Any] = field(default_factory=dict)
    expert_task: str | None = None
    export_package: bool = True
    package_path: str | None = None


@dataclass
class RunResult:
    """
    Hold the main artifacts returned by :func:`run`.

    Attributes:
        experiment_name: Run name used during logging.
        model: Model instance after training and optional checkpoint reload.
        trainer: Trainer used for fitting and evaluation.
        train_result: Raw dictionary returned by ``trainer.fit(...)``.
        test_results: Test records generated during evaluation.
    """

    experiment_name: str
    model: Any
    trainer: Any
    train_result: dict[str, Any]
    test_results: list[dict[str, Any]]


def seed_everything(seed: int) -> None:
    """Seed the research runner before models, samplers, or workers are used."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def export_final_model(cfg: RunCfg, train_dataset) -> None:
    """Export the final inference package without making it trainer state."""
    with tempfile.TemporaryDirectory(prefix="sleepwalker_final_") as temporary_directory:
        package_path = os.path.join(temporary_directory, "package")
        package = save_packaged_model(
            package_path,
            name=cfg.experiment_name,
            task=cfg.expert_task or cfg.model_name,
            model=cfg.model,
            dataset=train_dataset,
            classification_contract=cfg.trainer.classification_contract(),
            config=cfg.meta_data,
        )
        logger.artifact(path=package_path, dest="final")
        if cfg.package_path is not None:
            package.save(cfg.package_path)
            logger.info(f"Exported model package to {cfg.package_path}")


def export_final_checkpoint(cfg: RunCfg) -> None:
    """Store the selected final model together with resumable trainer state."""
    with tempfile.TemporaryDirectory(prefix="sleepwalker_final_checkpoint_") as temporary_directory:
        checkpoint_path = cfg.trainer.save_checkpoint(os.path.join(temporary_directory, "checkpoint.pt"), cfg.model)
        logger.artifact(path=checkpoint_path, dest="final")


def model_statistics(model: torch.nn.Module) -> dict[str, int]:
    """Return structured parameter and graph communication counts."""
    statistics = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    if isinstance(model, ModelGraphClassifier):
        statistics["communication_scalars_per_window"] = model.communication_scalars()
    return statistics


def run(cfg: RunCfg) -> RunResult:
    """Execute a prepared training run.

    Args:
        cfg: Fully prepared run configuration.

    Returns:
        A :class:`RunResult` containing the trained model, trainer, raw
        training result, and per-dataset test records.

    Raises:
        ValueError: If required configuration such as ``collate_fn`` is
            missing.
    """
    if cfg.collate_fn is None:
        raise ValueError("RunCfg.collate_fn must not be None.")
    train_dataset = combine_datasets(cfg.train_datasets)
    val_dataset = combine_datasets(cfg.val_datasets) if len(cfg.val_datasets) > 0 else None
    loader_seed = int(torch.initial_seed())

    os.makedirs(cfg.log_path, exist_ok=True)
    local_artifact_path = os.path.join(cfg.log_path, cfg.experiment_name)
    result_path = os.path.join(local_artifact_path, "results.json")
    if os.path.exists(result_path):
        raise FileExistsError(f"Training result already exists: {result_path}.")
    logger.add_sink(LocalArtifactSink(local_artifact_path))

    if cfg.use_mlflow:
        logger.add_sink(MlflowSink(tracking_uri=f"sqlite:///{os.path.join(cfg.log_path, 'mlflow.sqlite')}", experiment=cfg.experiment_name, artifact_uri=None))
    logger.start_run(run_name=cfg.experiment_name, tags=cfg.tags)

    statistics = model_statistics(cfg.model)
    cfg.meta_data["model_statistics"] = statistics
    logger.hparams(cfg.meta_data)
    logger.info(f"Model parameters: {statistics['trainable_parameters']:,} trainable / {statistics['total_parameters']:,} total")
    if "communication_scalars_per_window" in statistics:
        logger.info(f"Graph communication: {statistics['communication_scalars_per_window']:,} scalars per window")

    # TODO: Remove testing from this ??
    logger.info(f"Loaded {train_dataset.get_n_patients()} for training")
    if val_dataset is not None:
        logger.info(f"Loaded {val_dataset.get_n_patients()} for validation")
    logger.info(f"Prepared {len(cfg.test_datasets)} test dataset(s)")
    summary_input, _ =  cfg.model.input_spec() 
    if isinstance(summary_input, dict):
        for input_name, input_shape in summary_input.items():
            if len(input_shape) == 4:
                logger.info(f"Input '{input_name}' is {input_shape[1]} native call(s) of {input_shape[2]} x {input_shape[3]}")
            else:
                logger.info(f"Input '{input_name}' is {input_shape[1]} x {input_shape[2]}")
    elif summary_input is not None:
        logger.info(f"Input data is {summary_input[1]} x {summary_input[2]}")
        summary(cfg.model, input_size=summary_input, depth=5, device="cpu", row_settings=["hide_recursive_layers"])

    train_loader = build_loader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers_dataloader,
        n_samples=cfg.n_samples,
        collate_fn=cfg.collate_fn,
        sampling="patient_balanced" if cfg.patients_per_epoch is not None else "random",
        seed=loader_seed,
        patients_per_epoch=cfg.patients_per_epoch,
        patient_group_size=cfg.patient_group_size,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = build_loader(
            val_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers_dataloader,
            n_samples=cfg.n_samples,
            collate_fn=cfg.collate_fn,
            sampling="sequential",
            seed=loader_seed + 1,
            rejection_strategy="none",
        )

    train_result = cfg.trainer.fit(cfg.model, train_loader, val_loader)
    if "best_model_state" in train_result:
        cfg.model.load_state_dict(train_result["best_model_state"])

    export_final_checkpoint(cfg)
    if cfg.export_package:
        export_final_model(cfg, train_dataset)

    test_records: list[dict[str, Any]] = []
    for dataset_name, test_dataset in cfg.test_datasets:
        for repeat in cfg.test_repeats:
            context_label = f"{dataset_name}:r={repeat}" if len(cfg.test_repeats) > 1 else dataset_name
            logger.context(context_label)
            test_loader = build_loader(
                test_dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers_dataloader,
                n_samples=cfg.n_samples_test,
                collate_fn=cfg.collate_fn,
                sampling="sequential",
                seed=loader_seed + 2,
                rejection_strategy="none",
                n_views=repeat,
            )
            execution_model = RepeatedViewModel(cfg.model) if repeat > 1 else cfg.model
            test_loss, test_cm = cfg.trainer.test(execution_model, test_loader)
            record = {
                "name":cfg.experiment_name,
                "model": cfg.model_name,
                "test_loss": test_loss,
                "test_cm": test_cm,
                "train_loss": train_result["losses"],
                "train_cm": train_result["outputs"],
                "dataset": dataset_name,
                "classes": train_dataset.get_classes(),
                "path": os.path.join(cfg.log_path, cfg.experiment_name),
            }
            if len(cfg.test_repeats) > 1:
                record["repeat"] = repeat
            if "best_model" in train_result:
                record["best_model"] = train_result["best_model"]
            test_records.append(record)
            logger.uncontext()

    result_value = test_records[0] if len(test_records) == 1 else test_records
    write_json(result_path, result_value)

    logger.end_run()
    return RunResult(
        experiment_name=cfg.experiment_name,
        model=cfg.model,
        trainer=cfg.trainer,
        train_result=train_result,
        test_results=test_records,
    )
