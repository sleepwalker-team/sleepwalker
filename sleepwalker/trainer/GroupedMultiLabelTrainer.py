from typing import Optional

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.models.MetaModel import MetaModel
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.utils import logger


class GroupedMultiLabelTrainer(MultiLabelTrainer):
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
    def get_target(target, target_extra=None, class_cnts=None, task_config=None):
        return MultiLabelTrainer.get_target(target, target_extra, class_cnts, task_config)

    def _wrap_loader_with_repeats(self, loader, n_repeat: int, shuffle_default: bool):
        if n_repeat <= 1 or isinstance(getattr(loader, "sampler", None), RepeatSampler):
            return loader

        sampler = getattr(loader, "sampler", None)
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
        if not isinstance(model, MetaModel):
            raise ValueError("GroupedMultiLabelTrainer requires a MetaModel.")
        dataset_resolution = getattr(loader.dataset, "target_resolution", None)
        if dataset_resolution is None:
            raise ValueError("GroupedMultiLabelTrainer requires loader.dataset.target_resolution.")

        import pandas as pd

        dataset_resolution = pd.to_timedelta(dataset_resolution)
        if dataset_resolution != self.largest_task_resolution:
            raise ValueError(
                f"Dataset target_resolution={dataset_resolution} does not match largest_task_resolution={self.largest_task_resolution}."
            )

        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)

        loss_sum = 0
        epoch_cms = {
            cfg["task"]: torch.zeros((len(cfg["labels"]), len(cfg["labels"])), dtype=torch.int64, device=self.device)
            for cfg in self.task_specs
        }
        cnt = 0
        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = self.n_repeat_train if mode == "train" else self.n_repeat_test

        per_task_losses = [0 for _ in range(len(self.task_specs))]
        per_task_acc = [0 for _ in range(len(self.task_specs))]
        for batch in loader:
            x = batch["data"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)

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
                    if logits_sum is None:
                        logits_sum = {task: value.clone() for task, value in current_logits.items()}
                    else:
                        for task, value in current_logits.items():
                            logits_sum[task] += value
                logits = {task: value / n_repeat for task, value in logits_sum.items()}
            else:
                logits = model(x)

            losses = []
            batch_cms = {} if self.log_batches else None
            for task_idx, cfg in enumerate(self.task_specs):
                task = cfg["task"]
                n_classes = len(cfg["labels"])
                n_steps = cfg["n_steps"]
                y_task = y[:, task_idx, :n_steps]
                logits_task = logits[task]

                logits_flat = logits_task.reshape(-1, logits_task.shape[-1])
                y_flat = y_task.reshape(-1)
                if (y_flat < 0).any():
                    raise ValueError(f"Task '{task}' contains invalid targets. Unclear labels must be filtered in get_target().")
                if (y_flat >= n_classes).any():
                    raise ValueError(f"Task '{task}' contains out-of-range targets.")

                loss_task = self.loss_function(logits_flat, y_flat)
                per_task_losses[task_idx] += loss_task.item()

                pred_flat = logits_task.argmax(dim=-1).reshape(-1)
                per_task_acc[task_idx] += (pred_flat == y_flat).float().mean().item() * 100.0
                losses.append(loss_task)

                counts = torch.bincount(
                    y_flat.to(dtype=torch.int64) * n_classes + pred_flat.to(dtype=torch.int64),
                    minlength=n_classes * n_classes,
                ).reshape(n_classes, n_classes)
                epoch_cms[task] += counts
                if self.log_batches and batch_cms is not None:
                    batch_cms[task] = counts.detach().cpu().numpy()

            if len(losses) == 0:
                continue
            loss = torch.stack(losses).mean()

            if opt is not None:
                loss.backward()
                opt.step()

            cnt += 1
            loss_sum += float(loss.item())

            if self.log_batches and batch_cms:
                self._log_from_cms(batch_cms, float(loss.item()), mode=mode, scope="batch", step=self.steps[mode])

            task_desc = " ".join(
                f"{task['task'][:5]}: {task_loss/cnt:2.4f} - {task_acc/cnt:2.2f}"
                for task, task_loss, task_acc in zip(self.task_specs, per_task_losses, per_task_acc)
            )
            desc = f"{prefix:<12} loss {loss_sum/cnt:2.4f} {task_desc}"

            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1)
        epoch_cms_np = {task: cm.detach().cpu().numpy() for task, cm in epoch_cms.items()}
        self._log_from_cms(epoch_cms_np, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)
        self.log_confusion_tables(mode, epoch_cms_np)

        return epoch_loss, epoch_cms_np

    def test(self, model, test_loader):
        test_loader = self._wrap_loader_with_repeats(test_loader, self.n_repeat_test, shuffle_default=False)
        return super().test(model, test_loader)

    def fit(self, model, train_loader, val_loader=None):
        train_loader = self._wrap_loader_with_repeats(train_loader, self.n_repeat_train, shuffle_default=True)
        if val_loader is not None:
            val_loader = self._wrap_loader_with_repeats(val_loader, self.n_repeat_test, shuffle_default=False)
        return super().fit(model, train_loader, val_loader)
