import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.trainer.utils.disk import store_checkpoint
from sleepwalker.utils import logger


class BaseTrainer(ABC):
    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        train_transform: Optional[list[Callable]] = None,
        n_repeat_train: int = 1,
        n_repeat_test: int = 1,
    ):
        self.epochs = epochs
        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.device = device
        self.warmup_device = warmup_device
        self.save_every = save_every
        self.early_stopping_patience = early_stopping
        self.train_transform = train_transform
        self.n_repeat_train = n_repeat_train
        self.n_repeat_test = n_repeat_test

    def apply_train_transform(self, x: torch.Tensor) -> torch.Tensor:
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

    def warmup_preprocessors(self, model, data_loader, device: str = "cuda"):
        if hasattr(model, "_warmup_preprocessors"):
            return model._warmup_preprocessors(data_loader, device)

        model.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size

        if batch_size is None:
            raise ValueError("batch_size should not be None here.")

        for idx in range(len(model.preprocessors)):
            logger.progress_start(total_batches * batch_size, desc=f" {idx}/{len(model.preprocessors) - 1}", leave=True)
            if model.preprocessors[idx].requires_warmup():
                for batch in data_loader:
                    x = batch["data"].to(device)
                    x = model.apply_preprocessors(x, idx)
                    model.preprocessors[idx].update(x)
                    logger.progress_advance(batch_size)
            else:
                logger.progress_advance(total_batches * batch_size)
            logger.progress_close()

        return model

    def warmup_trainer(self, data_loader, device: str = "cuda"):
        return

    def _wrap_loader_with_repeats(self, loader, n_repeat: int, shuffle_default: bool):
        sampler = getattr(loader, "sampler", None)
        if n_repeat <= 1:
            return loader

        if isinstance(sampler, RepeatSampler):
            if sampler.n_repeat != n_repeat:
                logger.warning("Found a different n_repeat value in given sampler. Using supplied n_repeat")
            return loader

        if sampler is None:
            sampler = RandomSampler(loader.dataset) if shuffle_default else SequentialSampler(loader.dataset)
        sampler = RepeatSampler(sampler, n_repeat=n_repeat)

        loader_kwargs = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size * n_repeat,
            "shuffle": False,
            "sampler": sampler,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "drop_last": loader.drop_last,
            "pin_memory": loader.pin_memory,
            "persistent_workers": loader.persistent_workers,
        }
        if loader.num_workers > 0 and loader.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = loader.prefetch_factor
        return DataLoader(**loader_kwargs)

    def test(self, model, test_loader):
        test_loader = self._wrap_loader_with_repeats(test_loader, self.n_repeat_test, shuffle_default=False)
        model.eval()
        with torch.inference_mode():
            return self.run_epoch(test_loader, None, model, "TEST")

    def fit(self, model, train_loader, val_loader=None):
        train_loader = self._wrap_loader_with_repeats(train_loader, self.n_repeat_train, shuffle_default=True)
        if val_loader is not None:
            val_loader = self._wrap_loader_with_repeats(val_loader, self.n_repeat_test, shuffle_default=False)

        logger.context("Warmup trainer")
        self.warmup_trainer(train_loader)
        logger.uncontext()

        opt = self.optimizer_fn(model)

        if self.lr_scheduler_fn is not None:
            lr_scheduler = self.lr_scheduler_fn(opt)
            if isinstance(lr_scheduler, OneCycleLR):
                raise ValueError("OneCycleLR is currently not supported")
        else:
            lr_scheduler = None

        if self.early_stopping_patience and val_loader is None:
            logger.warning("early_stopping was set to true, but no validation dataset was given. Disabling early stopping")
            self.early_stopping_patience = None

        logger.context("Warmup preprocessors")
        self.warmup_preprocessors(model, train_loader, self.warmup_device)
        logger.uncontext()

        model = model.to(self.device)
        val_losses: list[float] = []
        losses = []
        outputs = []

        self.best_model_idx = None
        self.best_checkpoint = None
        self.steps = {"train": 0, "val": 0, "test": 0}
        self.epoch_step = 0
        self.last_folder = None

        for epoch in range(self.epochs):
            model.train()
            loss, output = self.run_epoch(train_loader, opt, model, f"TRAIN [{epoch+1}/{self.epochs}]")
            outputs.append({"train": output})
            losses.append({"train": loss})

            if self.save_every > 0 and (epoch % self.save_every == 0):
                logger.info(f"Logging intermediate model after {epoch} epochs.")
                self.last_folder = store_checkpoint(model, opt, lr_scheduler, tempfile.mkdtemp(prefix=f"checkpoint_epoch_{epoch}_"))
                logger.artifact(path=os.path.join(self.last_folder, "model.pt"), dest=f"{epoch}")
                logger.artifact(path=os.path.join(self.last_folder, "optimizer.pt"), dest=f"{epoch}")
                if lr_scheduler:
                    logger.artifact(path=os.path.join(self.last_folder, "scheduler.pt"), dest=f"{epoch}")

            if lr_scheduler is not None:
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
                    if self.best_checkpoint is not None:
                        logger.info(f"Found old best model in {self.best_checkpoint}. Deleting it")
                        shutil.rmtree(self.best_checkpoint)

                    self.best_checkpoint = store_checkpoint(model, opt, lr_scheduler, tempfile.mkdtemp(prefix="sleepwalker_best_model_"))
                    self.best_model_idx = imin

                if self.early_stopping_patience and (epoch - imin >= self.early_stopping_patience):
                    logger.info(f"Early stopping after {epoch} epochs - best epoch was {imin}")
                    return {
                        "losses": losses,
                        "outputs": outputs,
                        "best_model": imin,
                        "checkpoint": self.best_checkpoint,
                    }

            self.epoch_step += 1

        if self.last_folder is not None:
            return {"losses": losses, "outputs": outputs, "checkpoint": self.last_folder}
        return {"losses": losses, "outputs": outputs}

    @abstractmethod
    def run_epoch(self, loader, opt, model, prefix=""):
        pass
