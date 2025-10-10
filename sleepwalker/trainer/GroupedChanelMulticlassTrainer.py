import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.utils import logger

from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer

class GroupedChanelMulticlassTrainer(MulticlassTrainer):
    def __init__(
        self,
        groups:dict[list[str],list[str]],
        n_apply_repeats_train:int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.n_apply_repeats_train = n_apply_repeats_train
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

    def warmup_preprocessors(self, model: BaseModel, data_loader:DataLoader, device:str = "cuda") -> BaseModel:
        model.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size  

        if batch_size is None:
            raise ValueError(f"batch_size should not be None here.")

        for idx in range(len(model.preprocessors)):
            prog_size = total_batches*batch_size*self.n_apply_repeats_train
            logger.progress_start(prog_size, desc=f" {idx}/{len(model.preprocessors) - 1}", leave=True)

            if model.preprocessors[idx].requires_warmup():
                for batch in data_loader:
                    x = batch["data"].to(device)

                    # new_x = []
                    for _ in range(self.n_apply_repeats_train):
                        # select_idx = [1]
                        offset = 0
                        select_idx = []
                        for g in self.groups:
                            i = self.rng.integers(low=0,high=len(g))
                            offset += len(g)
                            select_idx.append(i)
                        x = x[:,:,select_idx]
                        x = model.apply_preprocessors(x, idx)
                        model.preprocessors[idx].update(x)
                        logger.progress_advance(batch_size)
                        # new_x.append(x)
                    # x = torch.vstack(new_x)
            else:
                # No warmup required -> Set tqdm bar to final value directly                
                logger.progress_advance(prog_size)
            logger.progress_close()

        return model

    def test(self, model: BaseModel, test_loader, n_apply_repeats:int=5):
        self.n_apply_repeats_test = n_apply_repeats

        model.eval()
        with torch.inference_mode():
            test_loss, test_cm = self.run_epoch(test_loader, None, model, f"TEST") 
        return test_loss, test_cm

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
            logits = model(x[:,:,select_idx])
            all_logits.append(logits)

        return torch.stack(all_logits).mean(axis=0)
