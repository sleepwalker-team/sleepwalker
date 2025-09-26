import inspect
import os
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from abc import ABC
from torch.optim.lr_scheduler import OneCycleLR

from sleepwalker.models.Basemodel import BaseModel, warmup_model
from sleepwalker.utils import logger

from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix, store_checkpoint
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer

class GroupedChanelMulticlassTrainer(MulticlassTrainer):
    def __init__(
        self,
        groups:dict[list[str],list[str]],
        n_apply_repeats_train:int = 5,
        n_apply_repeats_test:int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.n_apply_repeats_train = n_apply_repeats_train
        self.n_apply_repeats_test = n_apply_repeats_test
        self.rng = np.random.default_rng()

    @staticmethod
    def get_item(patient, time, data, target, groups, target_extra = None, percentage:float = 0.5):
        try:
            freq = pd.to_timedelta(target.index.freq).total_seconds()
            targets = torch.tensor(target.sum().to_numpy())
            target = MulticlassTrainer.target_to_multiclass(targets, None, len(target)*freq*percentage, True)

            if target_extra is not None:
                freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
                targets = torch.tensor(target_extra.sum().to_numpy())
                target_extra = MulticlassTrainer.target_to_multiclass(targets, None, len(target_extra)*freq*percentage, True)
            
            rng = np.random.default_rng()
            selected = []

            for g in groups:
                available = [col for col in g if col in data.columns]
                if not available:
                    raise ValueError(f"No available columns in group {g}")

                for col in g:
                    if col in data.columns:
                        selected.append(col)
                    else:
                        selected.append(rng.choice(available))

            data = torch.from_numpy(data[selected].values).float()
            return {"patient":patient, "time":time, "data":data, "target":target, "target_extra":target_extra}
        except Exception as e:
            pass
        return None

    def apply_model(self, model, x, is_test:bool=False):
        n_apply = self.n_apply_repeats_train if not is_test else self.n_apply_repeats_test

        all_logits = []
        for _ in range(n_apply):
            offset = 0
            select_idx = []
            for g in self.groups:
                idx = self.rng.integers(low=0,high=len(g))
                offset += len(g)
                select_idx.append(idx)
            # TODO FROM HERE, MODEL IS WRONG? 
            logits = model(x[:,:,select_idx])
            all_logits.append(logits)

        return torch.stack(all_logits).mean(axis=0)
