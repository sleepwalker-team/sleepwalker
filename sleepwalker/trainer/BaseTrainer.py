"""Shared trainer lifecycle for Sleepwalker experiments.

This module provides the common mechanics used by concrete trainers: optional
preprocessor warmup, checkpoint creation, and the fit/test loop that higher-level scripts call through
``sleepwalker.trainer.Run``.

Concrete task logic lives in subclasses such as
``MulticlassTrainer`` and ``MultiLabelTrainer``.
"""

import inspect
import os
from pathlib import Path
import random
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR

from sleepwalker.deployment.package import CloudpickleAdapter
from sleepwalker.utils import logger

def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["bit_generator"], numpy_state["state"].cpu().numpy(), numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("The checkpoint contains CUDA RNG state, but CUDA is not available.")
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError(f"The checkpoint contains RNG state for {len(state['cuda'])} CUDA devices, but {torch.cuda.device_count()} are available.")
        torch.cuda.set_rng_state_all(state["cuda"])


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
        save_every: int = 10,
        eval_every: int = 10,
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
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping
        self.return_best = return_best
        self.train_transform = train_transform
        self.optimizer = None
        self.lr_scheduler = None
        self.scheduler_per_batch = None
        self.completed_epochs = 0
        self.steps_per_epoch = None
        self.losses = []
        self.outputs = []
        self.val_losses = []
        self.best_model_idx = None
        self.best_model_state = None
        self.steps = {"train": 0, "val": 0, "test": 0}
        self.epoch_step = 0
        self.rng_state = None

    def validate_training_state(self, model: torch.nn.Module) -> None:
        """Fail if the serialized model, optimizer, and scheduler graph is inconsistent."""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected a torch.nn.Module, got {type(model).__name__}.")
        if self.optimizer is None:
            if self.lr_scheduler is not None or self.scheduler_per_batch is not None or self.completed_epochs != 0 or self.steps_per_epoch is not None:
                raise RuntimeError("Trainer has partial training state without an optimizer.")
            return
        if self.steps_per_epoch is None or self.scheduler_per_batch is None:
            raise RuntimeError("Trainer has incomplete optimizer or scheduler state.")
        if self.lr_scheduler_fn is not None and self.lr_scheduler is None:
            raise RuntimeError("Trainer is configured with a scheduler but has no scheduler state.")
        if self.lr_scheduler_fn is None and self.lr_scheduler is not None:
            raise RuntimeError("Trainer has scheduler state but is not configured with a scheduler.")
        model_parameters = {id(parameter) for parameter in model.parameters()}
        optimizer_parameters = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        if any(id(parameter) not in model_parameters for parameter in optimizer_parameters):
            raise RuntimeError("Optimizer parameters do not belong to the supplied model.")
        if self.lr_scheduler is not None and self.lr_scheduler.optimizer is not self.optimizer:
            raise RuntimeError("Scheduler is not attached to trainer.optimizer.")
        if self.completed_epochs < 0 or self.completed_epochs > self.epochs:
            raise RuntimeError(f"Trainer completed_epochs={self.completed_epochs} is invalid for epochs={self.epochs}.")

    def save_checkpoint(self, path: str | os.PathLike, model: torch.nn.Module) -> Path:
        """Serialize this trainer and its complete training state."""
        self.validate_training_state(model)
        if self.optimizer is None:
            raise RuntimeError("Cannot save a checkpoint before training has been initialized.")
        self.rng_state = capture_rng_state()
        checkpoint_path = Path(path)
        torch.save({"trainer": self, "model": model}, checkpoint_path, pickle_module=CloudpickleAdapter)
        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, path: str | os.PathLike, map_location=None):
        """Load and validate a serialized trainer and model."""
        checkpoint = torch.load(path, map_location=map_location, pickle_module=CloudpickleAdapter, weights_only=False)
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"trainer", "model"}:
            raise TypeError("Expected a checkpoint containing exactly 'trainer' and 'model'.")
        trainer = checkpoint["trainer"]
        model = checkpoint["model"]
        if not isinstance(trainer, cls):
            raise TypeError(f"Expected a {cls.__name__} checkpoint, got {type(trainer).__name__}.")
        trainer.validate_training_state(model)
        return trainer, model

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
                    if batch is None:
                        continue
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

    def fit(self, model, train_loader, val_loader=None):
        """Initialize or continue this trainer's model and optimization state."""
        if len(train_loader) < 1:
            raise ValueError("Cannot train with an empty training loader.")

        self.validate_training_state(model)
        if self.optimizer is None:
            self.set_loader_epoch(train_loader, 0)
            logger.context("Warmup trainer")
            self.warmup_trainer(train_loader)
            logger.uncontext()

            logger.context("Warmup preprocessors")
            self.warmup_preprocessor(model, train_loader, self.warmup_device)
            logger.uncontext()

            model.to(self.device)
            self.optimizer = self.optimizer_fn(model)
            self.steps_per_epoch = len(train_loader)
            self.lr_scheduler, self.scheduler_per_batch = build_lr_scheduler(self.lr_scheduler_fn, self.optimizer, self.epochs, self.steps_per_epoch)
            self.validate_training_state(model)
        else:
            if len(train_loader) != self.steps_per_epoch:
                raise ValueError(f"The checkpoint used {self.steps_per_epoch} training steps per epoch, but the new loader has {len(train_loader)}.")
            if self.completed_epochs >= self.epochs:
                raise RuntimeError(f"Training is already complete after {self.completed_epochs} epochs.")
            model.to(self.device)
            if self.rng_state is not None:
                restore_rng_state(self.rng_state)
            logger.info(f"Continuing training after {self.completed_epochs} completed epochs.")

        if self.early_stopping_patience and val_loader is None:
            logger.warning("early_stopping was set to true, but no validation dataset was given. Disabling early stopping")
            self.early_stopping_patience = None

        stopped_early = False
        for epoch in range(self.completed_epochs, self.epochs):
            self.set_loader_epoch(train_loader, epoch)
            model.train()
            loss, output = self.run_epoch(train_loader, self.optimizer, model, f"TRAIN [{epoch+1}/{self.epochs}]", self.lr_scheduler if self.scheduler_per_batch else None)
            self.outputs.append({"train": output})
            self.losses.append({"train": loss})

            epoch_number = epoch + 1
            if self.lr_scheduler is not None and not self.scheduler_per_batch:
                self.lr_scheduler.step()

            if val_loader is not None and self.eval_every > 0 and epoch_number % self.eval_every == 0:
                model.eval()
                with torch.inference_mode():
                    val_loss, val_output = self.run_epoch(val_loader, None, model, f"VAL [{epoch+1}/{self.epochs}]")
                self.outputs[-1]["val"] = val_output
                self.losses[-1]["val"] = val_loss
                self.val_losses.append(val_loss)

                best_validation_idx = int(np.argmin(self.val_losses))
                if self.best_model_idx is None or best_validation_idx == len(self.val_losses) - 1:
                    self.best_model_idx = epoch
                    if self.return_best:
                        self.best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

                if self.early_stopping_patience and epoch - self.best_model_idx >= self.early_stopping_patience:
                    logger.info(f"Early stopping after {epoch} epochs - best epoch was {self.best_model_idx}")
                    stopped_early = True

            self.epoch_step += 1
            self.completed_epochs = epoch_number

            if self.save_every > 0 and epoch_number % self.save_every == 0:
                logger.info(f"Logging training checkpoint after {epoch_number} epochs.")
                with tempfile.TemporaryDirectory(prefix="sleepwalker_training_checkpoint_") as temporary_directory:
                    checkpoint_path = self.save_checkpoint(Path(temporary_directory) / "checkpoint.pt", model)
                    logger.artifact(path=checkpoint_path, dest=str(epoch_number))

            if stopped_early:
                break

        result = {"losses": self.losses, "outputs": self.outputs}
        if self.return_best and self.best_model_state is not None:
            result["best_model"] = self.best_model_idx
            result["best_model_state"] = self.best_model_state
        return result

    @abstractmethod
    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        pass
