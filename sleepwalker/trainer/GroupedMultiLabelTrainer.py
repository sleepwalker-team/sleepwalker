import random
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.trainer.GroupedChanelMulticlassTrainer import ChannelTensor
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.utils import logger


class GroupedMultiLabelTrainer(MultiLabelTrainer):
    def __init__(
        self,
        groups:list[list[str]],
        n_repeat_train:int = 1,
        n_repeat_test:int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.n_repeat_train = n_repeat_train
        self.n_repeat_test = n_repeat_test
        self.rng = np.random.default_rng()

    @staticmethod
    def get_item(groups:list[list[str]], patient, time, data, target, target_extra = None, percentage:float = 0.5, class_cnts:Optional[List[float]] = None, task_config: Optional[dict[str, dict]] = None):
        if task_config is None:
            raise ValueError("task_config must not be None.")

        target = MultiLabelTrainer.target_to_multiclass(target, task_config, raise_error=False)
        if target is None:
            return None

        if class_cnts and len(class_cnts) == sum(len(cfg["labels"]) for cfg in task_config.values()):
            probas = class_cnts / np.sum(class_cnts)
            m = min(probas)
            idx = []
            offset = 0
            for task_idx, cfg in enumerate(task_config.values()):
                idx.extend([offset + int(i) for i in target[task_idx].tolist() if i >= 0])
                offset += len(cfg["labels"])
            if len(idx) > 0 and random.random() > min([m / probas[i] for i in idx]):
                return None

        x_tensor = torch.from_numpy(data.values.astype(np.float32))
        item = {"patient":patient, "time":time, "target":target, "data": ChannelTensor(x_tensor, columns = list(data.columns), groups=groups)}

        if target_extra is not None:
            extra = MultiLabelTrainer.target_to_multiclass(target_extra, task_config, raise_error=False)
            if extra is not None:
                item["target_extra"] = extra

        return item

    def warmup_preprocessors(self, model: BaseModel, data_loader:DataLoader, device:str = "cuda") -> BaseModel:
        model.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size

        if batch_size is None:
            raise ValueError(f"batch_size should not be None here.")

        for idx in range(len(model.preprocessors)):
            prog_size = total_batches*batch_size*self.n_repeat_train
            logger.progress_start(prog_size, desc=f" {idx}/{len(model.preprocessors) - 1}", leave=True)

            if model.preprocessors[idx].requires_warmup():
                for batch in data_loader:
                    for _ in range(self.n_repeat_train):
                        tmp_x = torch.stack([
                            x.sample(self.rng, self.groups).to(self.device)
                            for x in batch["data"]
                        ])
                        x = model.apply_preprocessors(tmp_x, idx)
                        model.preprocessors[idx].update(x)
                        logger.progress_advance(batch_size)
            else:
                logger.progress_advance(prog_size)
            logger.progress_close()

        return model

    def run_epoch(self, loader, opt, model, prefix=""):
        self.validate_model_and_loader(model, loader)
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)

        loss_sum = 0
        epoch_cms = {
            cfg["task"]: torch.zeros(
                (len(cfg["labels"]), len(cfg["labels"])),
                dtype=torch.int64,
                device=self.device,
            )
            for cfg in self.task_specs
        }
        cnt = 0
        n_samples = 0
        t0 = time.perf_counter()

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = self.n_repeat_train if mode == "train" else self.n_repeat_test

        for batch in loader:
            y = batch["target"].to(self.device, non_blocking=True)

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            logits_sum = None
            for _ in range(n_repeat):
                tmp_x = torch.stack([
                    x.sample(self.rng, self.groups).to(self.device)
                    for x in batch["data"]
                ])
                current_logits = model(tmp_x)
                if logits_sum is None:
                    logits_sum = {task: value.clone() for task, value in current_logits.items()}
                else:
                    for task, value in current_logits.items():
                        logits_sum[task] += value

            if logits_sum is None:
                continue
            logits = {task: value / n_repeat for task, value in logits_sum.items()}

            losses = []
            batch_task_stats = [] if self.log_batches else None
            for task_idx, cfg in enumerate(self.task_specs):
                task = cfg["task"]
                n_classes = len(cfg["labels"])
                n_steps = cfg["n_steps"]
                y_task = y[:, task_idx, :n_steps]
                logits_task = logits[task]

                if logits_task.shape[1] != n_steps or logits_task.shape[2] != n_classes:
                    raise ValueError(
                        f"Task '{task}' logits shape {tuple(logits_task.shape)} does not match expected "
                        f"(B, {n_steps}, {n_classes})."
                    )

                logits_flat = logits_task.reshape(-1, logits_task.shape[-1])
                y_flat = y_task.reshape(-1)
                if (y_flat < 0).any():
                    raise ValueError(f"Task '{task}' contains invalid targets. Unclear labels must be filtered in get_item().")
                if (y_flat >= n_classes).any():
                    raise ValueError(f"Task '{task}' contains out-of-range targets.")

                loss_task = self.loss_function(logits_flat, y_flat)
                losses.append(loss_task)

                pred_flat = logits_task.argmax(dim=-1).reshape(-1)
                if self.log_batches and batch_task_stats is not None:
                    batch_acc = (pred_flat == y_flat).float().mean().item() * 100.0
                    batch_task_stats.append((task, float(loss_task.item()), batch_acc))

                counts = torch.bincount(
                    y_flat.to(dtype=torch.int64) * n_classes + pred_flat.to(dtype=torch.int64),
                    minlength=n_classes * n_classes,
                ).reshape(n_classes, n_classes)
                epoch_cms[task] += counts

            if len(losses) == 0:
                continue
            loss = torch.stack(losses).mean()

            if opt is not None:
                loss.backward()
                opt.step()

            cnt += 1
            n_samples += y.shape[0]
            loss_sum += float(loss.item())
            throughput = n_samples / max(time.perf_counter() - t0, 1e-8)

            if self.log_batches:
                task_desc = " ".join(
                    f"{task[:8]} l {task_loss:1.3f} a {task_acc:2.1f}"
                    for task, task_loss, task_acc in batch_task_stats
                )
                desc = f"{prefix:<12} loss {loss_sum/cnt:2.4f} {task_desc}"
            else:
                desc = f"{prefix:<12} loss {loss_sum/cnt:2.4f}"

            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1)
        epoch_cms_np = {
            task: cm.detach().cpu().numpy()
            for task, cm in epoch_cms.items()
        }
        self._log_from_cms(epoch_cms_np, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)
        self.log_confusion_tables(mode, epoch_cms_np)

        return epoch_loss, epoch_cms_np
