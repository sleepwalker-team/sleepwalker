from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchinfo import summary

from sleepwalker.trainer.utils.disk import append_to_jsonl
from sleepwalker.trainer.utils.splits import combine_datasets
from sleepwalker.utils import logger


@dataclass
class RunCfg:
    """
    Fully prepared run configuration consumed by `run(...)`.

    The training script is expected to prepare the semantic parts of the run:
    dataset splits, model, trainer, experiment naming, and any task-specific
    behavior. `run(...)` only executes the shared training mechanics.
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
    tags: dict[str, str] = field(default_factory=dict)
    collate_fn: Any = None


@dataclass
class RunResult:
    """
    Result object returned by `run(...)`.

    Fields
    ------
    experiment_name:
        Name of the run used for logging and jsonl artifacts.
    model:
        Model instance after training and optional checkpoint reload.
    trainer:
        Trainer instance used for fit/test.
    train_result:
        Raw dictionary returned by `trainer.fit(...)`.
    test_records:
        Records written to the experiment jsonl file during test evaluation.
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
    sampler = RandomSampler(dataset, num_samples=n_samples) if n_samples is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_default and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )


def _infer_summary_input_size(train_dataset):
    try:
        ts_len = train_dataset.get_timeseries_len()
    except Exception:
        return None

    if hasattr(train_dataset, "channels"):
        n_channels = len(train_dataset.channels)
    elif hasattr(train_dataset, "get_input_channels"):
        n_channels = len(train_dataset.get_input_channels())
    else:
        return None

    return ts_len, n_channels


def run(cfg: RunCfg) -> RunResult:
    if cfg.collate_fn is None:
        raise ValueError("RunCfg.collate_fn must not be None.")

    train_dataset = combine_datasets(cfg.train_datasets)
    val_dataset = combine_datasets(cfg.val_datasets) if len(cfg.val_datasets) > 0 else None

    tracker = None
    if cfg.use_energy_tracker:
        from lamarr_energy_tracker import EnergyTracker

        tracker = EnergyTracker(project_name=cfg.experiment_name)
        tracker.start()

    logger.start_run(run_name=cfg.experiment_name, tags=cfg.tags)

    if os.path.exists("sleepwalker.log"):
        os.remove("sleepwalker.log")

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
                "model": cfg.model_name,
                "test_loss": test_loss,
                "test_cm": test_cm,
                "train_loss": train_result["losses"],
                "train_cm": train_result["outputs"],
                "dataset": dataset_name,
                "classes": train_dataset.get_classes(),
                "artifact_root": train_result["checkpoint"] if "checkpoint" in train_result else "",
            }
            if len(cfg.test_repeats) > 1:
                record["repeat"] = repeat
            if "best_model" in train_result:
                record["best_model"] = train_result["best_model"]
            append_to_jsonl(cfg.experiment_name, record)
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
