"""Shared experiment runner used by task-specific training scripts.

The training scripts in the repository prepare datasets, models, trainers, and
experiment metadata, then hand the assembled configuration to ``run(...)``.
This module centralizes loader construction, training invocation, and jsonl
logging of test results.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchinfo import summary

from sleepwalker.trainer.utils.disk import append_to_jsonl
from sleepwalker.trainer.utils.splits import combine_datasets
from sleepwalker.utils import LocalArtifactSink, MlflowSink, logger


@dataclass
class RunCfg:
    """
    Describe one fully prepared experiment invocation.

    The caller is responsible for assembling dataset splits, model, trainer,
    experiment naming, and task-specific behavior. :func:`run` only executes
    the shared mechanics.

    Attributes:
        experiment_name: Name used for logging and jsonl artifacts.
        model_name: Human-readable model identifier stored in test records.
        model: Model instance to fit and evaluate.
        trainer: Trainer instance implementing ``fit(...)`` and ``test(...)``.
        train_datasets: Training dataset parts that will be combined.
        val_datasets: Validation dataset parts that will be combined.
        test_datasets: Named test dataset pairs evaluated after training.
        batch_size: Batch size for train, validation, and test loaders.
        n_samples: Optional cap for random training/validation sampling.
        num_workers_dataloader: Worker count for all loaders built here.
        test_repeats: Repeat counts passed into trainer-side test evaluation.
        use_energy_tracker: Whether to enable the optional Lamarr energy
            tracker.
        tags: Logging tags forwarded to the repository logger.
        collate_fn: Required collate function used for all loaders.
        meta_data: Optional structured run metadata logged as hparams.
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
    test_repeats: list[int] = field(default_factory=lambda: [1])
    use_energy_tracker: bool = False
    use_mlflow: bool = False
    log_path: str = "sleepwalker"
    tags: dict[str, str] = field(default_factory=dict)
    collate_fn: Any = None
    meta_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """
    Hold the main artifacts returned by :func:`run`.

    Attributes:
        experiment_name: Run name used during logging.
        model: Model instance after training and optional checkpoint reload.
        trainer: Trainer used for fitting and evaluation.
        train_result: Raw dictionary returned by ``trainer.fit(...)``.
        test_records: Jsonl-style test records generated during evaluation.
    """

    experiment_name: str
    model: Any
    trainer: Any
    train_result: dict[str, Any]
    test_records: list[dict[str, Any]]


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    n_samples: int | None,
    collate_fn,
    shuffle_default: bool,
):
    """Construct a dataloader with optional random subsampling.

    Args:
        dataset: Dataset object for the loader.
        batch_size: Batch size passed to ``DataLoader``.
        num_workers: Dataloader worker count.
        n_samples: Optional random sample budget. When set and smaller than the
            dataset length, a ``RandomSampler`` is used.
        collate_fn: Collate function for batched items.
        shuffle_default: Whether to shuffle when no explicit sampler is used.

    Returns:
        A configured ``DataLoader``.
    """
    if n_samples is not None and len(dataset) > n_samples:
        sampler = RandomSampler(dataset, num_samples=n_samples)
    else:
        sampler = None
        
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle_default and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "drop_last": False,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        **loader_kwargs,
    )


def _infer_summary_input_size(train_dataset):
    try:
        ts_len = train_dataset.get_timeseries_len()
        n_channels = len(train_dataset.get_input_channels())
        return ts_len, n_channels
    except Exception:
        return None


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

    tracker = None
    if cfg.use_energy_tracker:
        from lamarr_energy_tracker import EnergyTracker

        tracker = EnergyTracker(project_name=cfg.experiment_name)
        tracker.start()

    os.makedirs(cfg.log_path, exist_ok=True)
    local_artifact_path = os.path.join(cfg.log_path, cfg.experiment_name)
    logger.add_sink(LocalArtifactSink(local_artifact_path))

    if cfg.use_mlflow:
        logger.add_sink(MlflowSink(tracking_uri=f"sqlite:///{os.path.join(cfg.log_path, 'mlflow.sqlite')}", experiment=cfg.experiment_name, artifact_uri=None))
    logger.start_run(run_name=cfg.experiment_name, tags=cfg.tags)
    
    if cfg.meta_data:
        logger.hparams(cfg.meta_data)

    logger.info(f"Loaded {train_dataset.get_n_patients()} for training")
    if val_dataset is not None:
        logger.info(f"Loaded {val_dataset.get_n_patients()} for validation")
    logger.info(f"Prepared {len(cfg.test_datasets)} test dataset(s)")
    summary_input = _infer_summary_input_size(train_dataset)
    if summary_input is not None:
        logger.info(f"Input data is {summary_input[0]} x {summary_input[1]}")
        summary(cfg.model, input_size=(1, summary_input[0], summary_input[1]), depth=5, row_settings=["hide_recursive_layers"])

    train_loader = build_loader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers_dataloader,
        n_samples=cfg.n_samples,
        collate_fn=cfg.collate_fn,
        shuffle_default=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = build_loader(
            val_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers_dataloader,
            n_samples=cfg.n_samples,
            collate_fn=cfg.collate_fn,
            shuffle_default=True,
        )

    train_result = cfg.trainer.fit(cfg.model, train_loader, val_loader)
    if "checkpoint" in train_result:
        logger.artifact(path=os.path.join(train_result["checkpoint"], "model.pt"), dest=f"final")
        logger.artifact(path=os.path.join(train_result["checkpoint"], "optimizer.pt"), dest=f"final")
        if Path(os.path.join(train_result["checkpoint"], "scheduler.pt")).is_file():
            logger.artifact(path=os.path.join(train_result["checkpoint"], "scheduler.pt"), dest=f"final")

        with torch.inference_mode():
            state_dict = torch.load(os.path.join(train_result["checkpoint"], "model.pt"), map_location="cpu")
            cfg.model.load_state_dict(state_dict)

    test_records: list[dict[str, Any]] = []
    for dataset_name, test_dataset in cfg.test_datasets:
        for repeat in cfg.test_repeats:
            context_label = f"{dataset_name}:r={repeat}" if len(cfg.test_repeats) > 1 else dataset_name
            logger.context(context_label)
            cfg.trainer.n_repeat_test = repeat

            test_loader = build_loader(
                test_dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers_dataloader,
                n_samples=None,
                collate_fn=cfg.collate_fn,
                shuffle_default=False,
            )
            test_loss, test_cm = cfg.trainer.test(cfg.model, test_loader)
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
            append_to_jsonl(os.path.join(cfg.log_path, "results"), record)
            test_records.append(record)
            logger.uncontext()

    if tracker is not None:
        tracker.stop()
    logger.end_run()
    return RunResult(
        experiment_name=cfg.experiment_name,
        model=cfg.model,
        trainer=cfg.trainer,
        train_result=train_result,
        test_records=test_records,
    )
