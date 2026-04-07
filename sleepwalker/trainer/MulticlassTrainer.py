from typing import Callable, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.trainer.utils import (
    cohen_kappa_from_confusion_matrix,
    f1_score_from_confusion_matrix,
    format_confusion_table,
    render_confusion_table_grid,
)


class MulticlassTrainer(BaseTrainer):
    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        classes: list[str],
        loss_function: Callable,
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        train_transform: Optional[list[Callable]] = None,
        n_repeat_train: int = 1,
        n_repeat_test: int = 1,
    ):
        super().__init__(
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            warmup_device=warmup_device,
            save_every=save_every,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            train_transform=train_transform,
            n_repeat_train=n_repeat_train,
            n_repeat_test=n_repeat_test,
        )

        self.classes = classes
        self.num_classes = len(classes)
        self.loss_function = loss_function

    def _log_from_cm(self, cm: np.ndarray, loss_value: float, mode: str, scope: str = "batch", step:int = 0):
        """Centralized metric logging from confusion matrix."""  
        total = cm.sum()  
        if total == 0:  
            return  
        
        acc = cm.trace() / total * 100.0  
        f1_micro = f1_score_from_confusion_matrix(cm, macro=False)  
        f1_macro = f1_score_from_confusion_matrix(cm, macro=True)  
        kappa = cohen_kappa_from_confusion_matrix(cm)  
        logger.metric(f"{scope}/{mode}/accuracy", acc, step=step)  
        logger.metric(f"{scope}/{mode}/f1_micro", f1_micro, step=step)  
        logger.metric(f"{scope}/{mode}/f1_macro", f1_macro, step=step)  
        logger.metric(f"{scope}/{mode}/coehns_kappa", kappa, step=step)  
        logger.metric(f"{scope}/{mode}/loss", float(loss_value), step=step) 

    def run_epoch(self, loader, opt, model, prefix=""):
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
                if x.shape[0] % n_repeat != 0:
                    raise ValueError(f"Batch size {x.shape[0]} is not divisible by n_repeat={n_repeat}.")
                base_batch = x.shape[0] // n_repeat
                x_grouped = x.view(base_batch, n_repeat, *x.shape[1:])
                y = y.view(base_batch, n_repeat, *y.shape[1:])[:, 0]

                logits_sum = None
                for repeat_idx in range(n_repeat):
                    current_logits = model(x_grouped[:, repeat_idx])
                    logits_sum = current_logits if logits_sum is None else logits_sum + current_logits
                logits = logits_sum / n_repeat
            else:
                logits = model(x)

            loss = self.loss_function(logits, y)
            
            target_np = y.argmax(axis=1).cpu().numpy()
            pred_np = logits.argmax(axis=1).cpu().numpy()

            if opt is not None:
                loss.backward()
                opt.step()

            cm = confusion_matrix(target_np, pred_np, labels=range(len(self.classes)))
            cm_sum += cm
            cnt += 1
            loss_sum += float(loss.item())

            step = self.steps[mode]
            self._log_from_cm(cm, float(loss.item()), mode=mode, scope="batch", step=step) 

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
        self._log_from_cm(cm_sum, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)  
        logger.info(render_confusion_table_grid([format_confusion_table(self.classes, cm_sum)], header=f"{mode.upper()} confusion matrix", n_cols=1))

        return epoch_loss, cm_sum  
