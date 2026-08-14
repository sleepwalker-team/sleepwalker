"""Shared trainer lifecycle for Sleepwalker experiments.

This module provides the common mechanics used by concrete trainers: optional
preprocessor warmup, checkpoint creation, and the fit/test loop that higher-level scripts call through
``sleepwalker.trainer.Run``.

Concrete task logic lives in subclasses such as
``MulticlassTrainer`` and ``MultiLabelTrainer``.
"""

import inspect
import shutil
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from sleepwalker.deployment import save_packaged_model
from sleepwalker.utils import logger


def build_lr_scheduler(lr_scheduler_fn, optimizer, epochs: int, steps_per_epoch: int):
    """Build a scheduler once and supply missing training-length arguments."""
    if lr_scheduler_fn is None:
        return None, False
    if steps_per_epoch < 1:
        raise ValueError("Cannot build a learning-rate scheduler for an empty training loader.")

    parameters = inspect.signature(lr_scheduler_fn).parameters
    arguments = {}
    total_steps = parameters.get("total_steps")
    if total_steps is not None and total_steps.default in (None, inspect.Parameter.empty):
        scheduler_epochs = parameters.get("epochs")
        scheduler_steps = parameters.get("steps_per_epoch")
        if scheduler_epochs is not None and scheduler_steps is not None:
            if scheduler_epochs.default in (None, inspect.Parameter.empty):
                arguments["epochs"] = epochs
            if scheduler_steps.default in (None, inspect.Parameter.empty):
                arguments["steps_per_epoch"] = steps_per_epoch
        else:
            arguments["total_steps"] = epochs * steps_per_epoch

    scheduler = lr_scheduler_fn(optimizer, **arguments)
    return scheduler, isinstance(scheduler, (OneCycleLR, CyclicLR))


class BaseTrainer(ABC):
    """Abstract base class for model training and inference helpers.

    Subclasses implement task-specific epoch execution. This base class handles
    preprocessor warmup, checkpointing, and the train/test lifecycle.
    """

    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        return_best: bool = True,
        train_transform: Optional[list[Callable]] = None,
    ):
        if not isinstance(return_best, bool):
            raise TypeError("return_best must be a bool.")
        self.epochs = epochs
        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.device = device
        self.warmup_device = warmup_device
        self.save_every = save_every
        self.early_stopping_patience = early_stopping
        self.return_best = return_best
        self.train_transform = train_transform

    def export_model(self, model: torch.nn.Module, dataset, *, name: str, task: str, config: dict[str, Any], dest: str) -> None:
        """Export a directly testable model package through the active artifact sinks."""
        with tempfile.TemporaryDirectory(prefix=f"sleepwalker_{dest}_") as temporary_directory:
            package_path = os.path.join(temporary_directory, "package")
            save_packaged_model(
                package_path,
                name=name,
                task=task,
                model=model,
                dataset=dataset,
                classification_contract=self.classification_contract(),
                config=config,
            )
            logger.artifact(path=package_path, dest=dest)

    def apply_train_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-sample augmentation transforms to a batch tensor.

        Args:
            x: Input tensor shaped ``[B, T, C]``.

        Returns:
            A tensor with the same shape and dtype after sequentially applying
            the configured ``train_transform`` callables.
        """
        if self.train_transform is None or len(self.train_transform) == 0:
            return x

        x_device = x.device
        out = []
        for sample in x:
            sample_df = pd.DataFrame(sample.detach().cpu().numpy())
            for transform in self.train_transform:
                sample_df = transform(sample_df)
            out.append(torch.from_numpy(sample_df.to_numpy()).to(x_device, dtype=x.dtype))
        return torch.stack(out, dim=0)

    def warmup_preprocessor(self, model, data_loader, device: str = "cuda"):
        """Warm up model preprocessing steps that need streaming statistics.

        Args:
            model: Model instance exposing the ``BaseModel`` preprocessor API.
            data_loader: Loader that yields batches with a ``data`` field.
            device: Device used while warming preprocessors.

        Returns:
            The warmed model. The object is mutated in place.
        """
        if hasattr(model, "warmup_preprocessor"):
            return model.warmup_preprocessor(data_loader, device)

        model.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size

        if batch_size is None:
            raise ValueError("batch_size should not be None here.")

        steps = model.preprocessors
        for idx, step in enumerate(steps):
            logger.progress_start(total_batches * batch_size, desc=f" {idx}/{len(steps) - 1}", leave=True)
            if step.requires_warmup():
                for batch in data_loader:
                    x = batch["data"].to(device)
                    x = model.apply_preprocessors(x, idx)
                    step.update(x)
                    logger.progress_advance(batch_size)
            else:
                logger.progress_advance(total_batches * batch_size)
            logger.progress_close()

        return model

    def warmup_trainer(self, data_loader, device: str = "cuda"):
        """Run trainer-specific warmup before optimization starts.

        Notes:
            The base implementation is a no-op. Subclasses use this hook for
            class-count estimation, loss reweighting, and related setup.
        """
        return

    def set_loader_epoch(self, loader, epoch: int):
        sampler = loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def test(self, model, test_loader):
        """Run evaluation on one prepared loader."""
        model.eval()
        with torch.inference_mode():
            return self.run_epoch(test_loader, None, model, "TEST")

    def fit(self, model, train_loader, val_loader=None, *, package_name: Optional[str] = None, package_task: Optional[str] = None, package_config: Optional[dict[str, Any]] = None):
        """Train a model and optionally track validation checkpoints.

        Args:
            model: Model instance to optimize.
            train_loader: Training dataloader.
            val_loader: Optional validation dataloader.

        Returns:
            A dictionary containing at least ``losses`` and ``outputs``. When
            checkpointing occurs, the dictionary may also contain
            ``checkpoint`` and ``best_model``.
        """
        package_name = package_name or model.__class__.__name__
        package_task = package_task or package_name
        package_config = dict(package_config or {})
        self.set_loader_epoch(train_loader, 0)
        logger.context("Warmup trainer")
        self.warmup_trainer(train_loader)
        logger.uncontext()

        opt = self.optimizer_fn(model)

        lr_scheduler, scheduler_per_batch = build_lr_scheduler(self.lr_scheduler_fn, opt, self.epochs, len(train_loader))

        if self.early_stopping_patience and val_loader is None:
            logger.warning("early_stopping was set to true, but no validation dataset was given. Disabling early stopping")
            self.early_stopping_patience = None

        self.set_loader_epoch(train_loader, 0)
        logger.context("Warmup preprocessors")
        self.warmup_preprocessor(model, train_loader, self.warmup_device)
        logger.uncontext()

        model = model.to(self.device)
        val_losses: list[float] = []
        losses = []
        outputs = []

        self.best_model_idx = None
        self.best_checkpoint = None
        self.steps = {"train": 0, "val": 0, "test": 0}
        self.epoch_step = 0
        for epoch in range(self.epochs):
            self.set_loader_epoch(train_loader, epoch)
            model.train()
            loss, output = self.run_epoch(train_loader, opt, model, f"TRAIN [{epoch+1}/{self.epochs}]", lr_scheduler if scheduler_per_batch else None)
            outputs.append({"train": output})
            losses.append({"train": loss})

            epoch_number = epoch + 1
            if self.save_every > 0 and epoch_number % self.save_every == 0:
                logger.info(f"Logging intermediate model after {epoch_number} epochs.")
                checkpoint_config = dict(package_config)
                checkpoint_config["checkpoint_epoch"] = epoch_number
                self.export_model(model, train_loader.dataset, name=f"{package_name}-epoch-{epoch_number}", task=package_task, config=checkpoint_config, dest=str(epoch_number))

            if lr_scheduler is not None and not scheduler_per_batch:
                lr_scheduler.step()

            if val_loader is not None:
                model.eval()
                with torch.inference_mode():
                    val_loss, val_output = self.run_epoch(val_loader, None, model, f"VAL [{epoch+1}/{self.epochs}]")
                outputs[-1]["val"] = val_output
                losses[-1]["val"] = val_loss
                val_losses.append(val_loss)

                imin = int(np.argmin(val_losses))
                if self.best_model_idx is None or imin != self.best_model_idx:
                    self.best_model_idx = imin
                    if self.return_best:
                        if self.best_checkpoint is not None:
                            logger.info(f"Found old best model in {self.best_checkpoint}. Deleting it")
                            shutil.rmtree(os.path.dirname(self.best_checkpoint))

                        best_folder = tempfile.mkdtemp(prefix="sleepwalker_best_model_")
                        self.best_checkpoint = os.path.join(best_folder, "model.pt")
                        torch.save(model.state_dict(), self.best_checkpoint)

                if self.early_stopping_patience and (epoch - imin >= self.early_stopping_patience):
                    logger.info(f"Early stopping after {epoch} epochs - best epoch was {imin}")
                    result = {
                        "losses": losses,
                        "outputs": outputs,
                    }
                    if self.return_best:
                        result["best_model"] = imin
                        result["checkpoint"] = self.best_checkpoint
                    return result

            self.epoch_step += 1

        result = {"losses": losses, "outputs": outputs}
        if self.return_best and self.best_checkpoint is not None:
            result["best_model"] = self.best_model_idx
            result["checkpoint"] = self.best_checkpoint
        return result

    @abstractmethod
    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        pass
