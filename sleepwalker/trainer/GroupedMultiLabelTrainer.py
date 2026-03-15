import random
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.trainer.GroupedChanelMulticlassTrainer import ChannelTensor
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix
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

        try:
            freq = pd.to_timedelta(target.index.freq).total_seconds()
            targets = {str(k): float(v) for k, v in target.sum().to_dict().items()}
            target = MultiLabelTrainer.target_to_multilabel(targets, task_config, len(target)*freq, True)

            if class_cnts and len(class_cnts) == len(target):
                probas = class_cnts / np.sum(class_cnts)
                m = min(probas)
                idx = target.nonzero(as_tuple=False).flatten().tolist()
                if len(idx) > 0 and random.random() > min([m / probas[i] for i in idx]):
                    return None

            x_tensor = torch.from_numpy(data.values.astype(np.float32))
            item = {"patient":patient, "time":time, "target":target, "data": ChannelTensor(x_tensor, columns = list(data.columns), groups=groups)}

            if target_extra is not None:
                freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
                targets = {str(k): float(v) for k, v in target_extra.sum().to_dict().items()}
                item["target_extra"] = MultiLabelTrainer.target_to_multilabel(targets, task_config, len(target_extra)*freq, False)

            return item
        except Exception:
            pass
        return None

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
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)

        loss_sum = 0
        cm_sum = {
            task: np.zeros((len(cfg["labels"]), len(cfg["labels"])), dtype=np.int64)
            for task, cfg in self.task_config.items()
        }
        cnt = 0
        n_samples = 0
        t0 = time.perf_counter()

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = self.n_repeat_train if mode == "train" else self.n_repeat_test

        for batch in loader:
            y = batch["target"].to(self.device)

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            logits_sum = 0
            for _ in range(n_repeat):
                tmp_x = torch.stack([
                    x.sample(self.rng, self.groups).to(self.device)
                    for x in batch["data"]
                ])
                logits_sum += model(tmp_x)

            logits = logits_sum / n_repeat

            losses = []
            cms = {}
            for task, task_slice in self.task_slices.items():
                y_task = y[:, task_slice]
                logits_task = logits[:, task_slice]

                losses.append(self.loss_function(logits_task, y_task))

                target_np = y_task.argmax(axis=1).cpu().numpy()
                pred_np = logits_task.argmax(axis=1).cpu().numpy()
                cms[task] = confusion_matrix(target_np, pred_np, labels=range(logits_task.shape[1]))

            loss = sum(losses)

            if opt is not None:
                loss.backward()
                opt.step()

            for task, cm in cms.items():
                cm_sum[task] += cm

            cnt += 1
            n_samples += y.shape[0]
            loss_sum += float(loss.item()) / len(self.task_config)
            throughput = n_samples / max(time.perf_counter() - t0, 1e-8)

            step = self.steps[mode]
            self._log_from_cms(cms, float(loss.item()) / len(self.task_config), throughput, mode=mode, scope="batch", step=step)

            accs = np.mean([
                cm.trace() / cm.sum() * 100.0
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])
            f1_micro = np.mean([
                f1_score_from_confusion_matrix(cm, macro=False)
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])
            f1_macro = np.mean([
                f1_score_from_confusion_matrix(cm, macro=True)
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])
            coehns_kappa = np.mean([
                cohen_kappa_from_confusion_matrix(cm)
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])

            desc = f"{prefix:<12} {loss_sum/cnt:2.4f} acc {accs:2.3f} " \
                   f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} κ {coehns_kappa:2.3f} thr {throughput:4.1f}/s"

            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1)
        self._log_from_cms(cm_sum, epoch_loss, n_samples / max(time.perf_counter() - t0, 1e-8), mode=mode, scope="epoch", step=self.epoch_step)

        for task, cm in cm_sum.items():
            logger.info(f"{mode.upper()} | task={task} | labels={self.task_config[task]['labels']} | confusion_matrix=\n{cm}")

        return epoch_loss, cm_sum
