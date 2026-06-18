"""Shared experiment runner used by task-specific training scripts.

The training scripts in the repository prepare datasets, models, trainers, and
experiment metadata, then hand the assembled configuration to ``run(...)``.
This module centralizes loader construction, training invocation, and jsonl
logging of test results.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchinfo import summary

from sleepwalker.trainer.utils.disk import append_to_jsonl
from sleepwalker.datasets.MultiDataset import combine_datasets
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

    The fields fall into three groups:

    1. Core execution fields:
       ``model``, ``trainer``, dataset lists, loader settings, and evaluation
       repeat counts. These define what is trained and how it is iterated.
    2. Logging fields:
       ``experiment_name``, ``model_name``, ``tags``, ``meta_data``,
       ``log_path``, and the feature flags for MLflow and energy tracking.
       These affect observability, not model semantics.
    3. Deployment fields:
       ``expert_name``, ``expert_task``, ``expert_builder``,
       ``expert_dataset_template``, and ``expert_metadata``. These define the
       expert package snapshot saved by the trainer and are the main link
       between a completed run and later reloadable inference/training usage.

    Expectations and non-goals:

    - ``run`` does not derive missing task semantics. If a trainer needs a
      specific collator, dataset template, or package builder, the caller must
      supply them.
    - ``train_datasets`` and ``val_datasets`` are combined in-memory via the
      repository dataset utilities; they are not treated as separate training
      domains once handed to the runner.
    - ``expert_dataset_template`` should represent the unlabelled inference
      shape expected by the saved model. When omitted, ``run`` falls back to
      the first training dataset if it exposes ``to_unlabelled()``.

    Field details:

    - ``experiment_name``: Stable run identifier used for local artifact
      folders, MLflow runs, JSONL test records, and checkpoint naming.
    - ``model_name``: Human-readable label written into metrics/test outputs.
      This is descriptive metadata and does not need to match the Python class
      name.
    - ``model``: Instantiated model object to train and later evaluate.
    - ``trainer``: Trainer object implementing ``fit(...)`` and ``test(...)``.
      The runner also injects expert package metadata into this object.
    - ``train_datasets`` / ``val_datasets``: Dataset objects combined into one
      loader per split.
    - ``test_datasets``: Named datasets evaluated after training. The tuple
      label becomes part of the logged test records.
    - ``batch_size``: Loader batch size shared across train/val/test.
    - ``n_samples``: Optional random sampling cap for train/val loaders. This
      is a loader-time budget, not a dataset truncation on disk.
    - ``num_workers_dataloader``: Worker count for loaders created here.
    - ``test_repeats``: Logical inference repeat counts, typically used with
      grouped/randomized inputs where evaluation is averaged over repeated
      views.
    - ``use_energy_tracker``: Enables the optional Lamarr energy tracker.
    - ``use_mlflow``: Enables the repository MLflow sink for this run.
    - ``log_path``: Root folder under which local run artifacts are written.
    - ``tags``: Logger/MLflow tags associated with the run.
    - ``collate_fn``: Required batch collator for every loader built here.
    - ``meta_data``: Structured run metadata logged as hyperparameters and
      copied into the expert package metadata.
    - ``expert_name`` / ``expert_task``: Export-facing identity for the saved
      package. Defaults fall back to ``experiment_name`` and ``model_name``.
    - ``expert_builder``: Reload specification used by the expert package to
      reconstruct model, trainer, and dataset template in a code-aware way.
    - ``expert_dataset_template``: Unlabelled dataset template that defines
      inference-time structure and feeds the package input contract.
    - ``expert_metadata``: Additional export-only metadata merged into package
      metadata alongside ``meta_data``.
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
    expert_name: str | None = None
    expert_task: str | None = None
    expert_builder: dict[str, Any] | None = None
    expert_dataset_template: Any = None
    expert_metadata: dict[str, Any] = field(default_factory=dict)


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
    package_dataset_template = cfg.expert_dataset_template
    if package_dataset_template is None and len(cfg.train_datasets) > 0:
        first_dataset = cfg.train_datasets[0]
        if hasattr(first_dataset, "to_unlabelled"):
            package_dataset_template = first_dataset.to_unlabelled()
        else:
            package_dataset_template = first_dataset

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

    cfg.trainer._expert_package_cfg = {
        "expert_name": cfg.expert_name or cfg.experiment_name,
        "task": cfg.expert_task or cfg.model_name,
        "builder": cfg.expert_builder,
        "dataset_template": package_dataset_template,
        "metadata": {
            "experiment_name": cfg.experiment_name,
            "model_name": cfg.model_name,
            **cfg.expert_metadata,
            **cfg.meta_data,
        },
    }

    logger.info(f"Loaded {train_dataset.get_n_patients()} for training")
    if val_dataset is not None:
        logger.info(f"Loaded {val_dataset.get_n_patients()} for validation")
    logger.info(f"Prepared {len(cfg.test_datasets)} test dataset(s)")
    summary_input, _ =  cfg.model.input_spec() 
    if summary_input is not None:
        logger.info(f"Input data is {summary_input[1]} x {summary_input[2]}")
        summary(cfg.model, input_size=summary_input, depth=5, row_settings=["hide_recursive_layers"])

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
        logger.artifact(path=train_result["checkpoint"], dest="final")

        with torch.inference_mode():
            state_dict = torch.load(os.path.join(train_result["checkpoint"], "model_state.pt"), map_location="cpu")
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
