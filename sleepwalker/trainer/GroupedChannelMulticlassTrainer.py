from typing import Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix
from sleepwalker.utils import logger


class GroupedChannelMulticlassTrainer(MulticlassTrainer):
    def __init__(
        self,
        groups: Optional[list[list[str]]] = None,
        n_repeat_train: int = 1,
        n_repeat_test: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.n_repeat_train = n_repeat_train
        self.n_repeat_test = n_repeat_test

    @staticmethod
    def get_target(target, target_extra=None, percentage: float = 0.5, class_cnts=None):
        return MulticlassTrainer.get_target(target, target_extra, percentage, class_cnts)

    def _wrap_loader_with_repeats(self, loader, n_repeat: int, shuffle_default: bool):
        sampler = getattr(loader, "sampler", None)
        if n_repeat <= 1:
            return loader

        if isinstance(sampler, RepeatSampler):
            if sampler.n_repeat != n_repeat:
                logger.warning("Found a different n_repeat value in given sampler. Using suppled n_repeat")
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

    def run_epoch(self, loader, opt, model, prefix=""):
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        nc = self.num_classes

        loss_sum = 0
        cm_sum = np.zeros((nc, nc), dtype=np.int64)
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = loader.sampler.n_repeat if isinstance(loader.sampler, RepeatSampler) else 1

        for batch in loader:
            x = batch["data"].to(self.device)
            y = batch["target"].to(self.device)

            if opt is not None:
                x = self.apply_train_transform(x)

            if n_repeat > 1:
                if x.shape[0] % n_repeat != 0:
                    raise ValueError(f"Batch size {x.shape[0]} is not divisible by n_repeat={n_repeat}.")
                base_batch = x.shape[0] // n_repeat
                y = y.view(base_batch, n_repeat, *y.shape[1:])[:, 0]
            else:
                base_batch = x.shape[0]

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            if n_repeat > 1:
                # Grouped runs evaluate each repeat explicitly and average the
                # resulting logits before loss/metrics so the reported numbers
                # reflect the repeated channel sampling policy.
                logits_sum = None
                x_grouped = x.view(base_batch, n_repeat, *x.shape[1:])
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

        return epoch_loss, cm_sum

    def test(self, model, test_loader):
        test_loader = self._wrap_loader_with_repeats(test_loader, self.n_repeat_test, shuffle_default=False)
        return super().test(model, test_loader)

    def fit(self, model, train_loader, val_loader=None):
        train_loader = self._wrap_loader_with_repeats(train_loader, self.n_repeat_train, shuffle_default=True)
        if val_loader is not None:
            val_loader = self._wrap_loader_with_repeats(val_loader, self.n_repeat_test, shuffle_default=False)
        return super().fit(model, train_loader, val_loader)
