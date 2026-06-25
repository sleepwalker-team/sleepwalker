from functools import partial
import random
from typing import Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import torch

from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.losses import class_weights_for_loss
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.datasets.utils import estimate_class_cnts
from sleepwalker.trainer.utils.display import format_confusion_table, render_confusion_table_grid
from sleepwalker.trainer.utils.metrics import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix


class DiagnosisTrainer(MulticlassTrainer):
    """Train and evaluate a single softmax classification head for patient-level
        diagnosis labels and several segments per sample.

    Args:
        epochs: Number of training epochs.
        optimizer: Factory that builds an optimizer for the model.
        classes: Ordered output labels used for targets, confusion matrices,
            and prediction tables.
        sequences_per_patient: Number of segments per sample.
        loss_function: Base loss function, usually cross entropy.
        device: Torch device used for training and inference.
        warmup_device: Device used during preprocessor warmup.
        save_every: Checkpoint cadence in epochs.
        lr_scheduler: Optional scheduler factory.
        early_stopping: Optional validation patience in epochs.
        train_transform: Optional list of per-sample transforms.
        n_repeat_train: Number of repeated views per training sample.
        n_repeat_test: Number of repeated views per evaluation sample.
        loss_mode: Loss reweighting mode understood by
            ``class_weights_for_loss``.
        class_weights: Optional manual per-class weights.
        balance_batches: Whether to reject overrepresented targets in
            ``prepare_target`` during training.
        balance_gamma: Exponent applied to the inverse-frequency acceptance
            probability during batch balancing. ``1.0`` reproduces the
            previous behavior, values above ``1.0`` make balancing more
            aggressive, and values between ``0`` and ``1`` make it weaker.
    """

    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        classes: list[str],
        sequences_per_patient: int,
        loss_function: Callable,
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        train_transform: Optional[list[Callable]] = None,
        n_repeat_train: int = 1,
        n_repeat_test: int = 1,
        loss_mode: str = "regular",
        class_weights: Optional[dict[str, float]] = None,
        balance_batches: bool = False,
        balance_gamma: float = 1.0,
    ):
        super().__init__(
            epochs=epochs,
            optimizer=optimizer,
            classes=classes,
            loss_function=loss_function,
            device=device,
            warmup_device=warmup_device,
            save_every=save_every,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            train_transform=train_transform,
            n_repeat_train=n_repeat_train,
            n_repeat_test=n_repeat_test,
            loss_mode=loss_mode,
            class_weights=class_weights,
            balance_batches=balance_batches,
            balance_gamma=balance_gamma,
        )

        self.sequences_per_patient = sequences_per_patient

    def warmup_preprocessors(self, model, data_loader, device: str = "cuda"):
        """Warm up model preprocessors that need streaming statistics.

        Args:
            model: Model instance that may expose ``preprocessors`` or a custom
                ``_warmup_preprocessors`` hook.
            data_loader: Loader that yields batches with a ``data`` field.
            device: Device used while warming preprocessors.

        Returns:
            The warmed model. The object is mutated in place.
        """
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
                    for i in range(self.sequences_per_patient):
                        x_seg = x[:, i]
                        x_seg= model.apply_preprocessors(x_seg, idx)
                        model.preprocessors[idx].update(x_seg)
                    logger.progress_advance(batch_size)
            else:
                logger.progress_advance(total_batches * batch_size)
            logger.progress_close()

        return model

    def run_epoch(self, loader, opt, model, prefix=""):
        """Run one train, validation, or test epoch.

        Args:
            loader: Dataloader producing batch dictionaries with ``data`` and
                one-hot ``target`` fields.
            opt: Optimizer for training epochs, or ``None`` during evaluation.
            model: Model instance to execute.
            prefix: Progress-label prefix used to infer the logging mode.

        Returns:
            A tuple ``(epoch_loss, confusion_matrix)``.
        """
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        nc = self.num_classes
        
        loss_sum = 0
        cm_sum = np.zeros((nc, nc), dtype=np.int64)
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = loader.sampler.n_repeat if isinstance(getattr(loader, "sampler", None), RepeatSampler) else 1

        for batch in loader:
            x = batch["data"].to(self.device)
            y = batch["target"].to(self.device)

            if opt is not None:
                x = self.apply_train_transform(x)
                opt.zero_grad(set_to_none=True)

            if n_repeat > 1:
                # TODO: Repeats not yet implmented for diagnosis classification
                if x.shape[0] % n_repeat != 0:
                    raise ValueError(f"Batch size {x.shape[0]} is not divisible by n_repeat={n_repeat}.")
                base_batch = x.shape[0] // n_repeat
                x_grouped = x.view(base_batch, n_repeat, *x.shape[1:])
                y = y.view(base_batch, n_repeat, *y.shape[1:])[:, 0]

                logits_sum = None 
                for repeat_idx in range(n_repeat):
                    current_logits = model(x_grouped[:, repeat_idx].contiguous())
                    logits_sum = current_logits if logits_sum is None else logits_sum + current_logits
                logits = logits_sum / n_repeat
            else:
                logits_sum = None
                for i in range(self.sequences_per_patient):
                    x_seg = x[:, i] # (batch, timesteps, channels)
                    current_logits = model(x_seg)
                    logits_sum = current_logits if logits_sum is None else logits_sum + current_logits
                logits = logits_sum / self.sequences_per_patient

            if logits.shape[1] == 1: # TODO add this to forward deployment!
                loss = self.loss_function(logits, y.argmax(axis=1, keepdim=True).float())
                pred_np = (logits >= 0.5).ravel().long().cpu().numpy()
            else:
                loss = self.loss_function(logits, y)
                pred_np = logits.argmax(axis=1).cpu().numpy()
            target_np = y.argmax(axis=1).cpu().numpy()
            
            if opt is not None:
                loss.backward()
                opt.step()

            cm = confusion_matrix(target_np, pred_np, labels=range(len(self.classes)))
            cm_sum += cm
            cnt += 1
            loss_sum += float(loss.item())

            step = self.steps[mode]
            self._log_from_cm(cm, float(loss.item()), mode=mode, scope="batch", step=step, show_cm = False) 

            accs = cm_sum.trace() / cm_sum.sum() * 100.0
            f1_micro = f1_score_from_confusion_matrix(cm_sum, macro=False)
            f1_macro = f1_score_from_confusion_matrix(cm_sum, macro=True)
            coehns_kappa = cohen_kappa_from_confusion_matrix(cm_sum)

            desc = f"{prefix:<12} {loss_sum/cnt:2.4f} acc {accs:2.3f} " \
                   f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} κ {coehns_kappa:2.3f}"
            
            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1) 
        self._log_from_cm(cm_sum, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step, show_cm = True)  
        

        return epoch_loss, cm_sum  
    
