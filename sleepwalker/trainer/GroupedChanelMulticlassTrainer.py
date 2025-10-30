import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torch
from dataclasses import dataclass

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.utils import logger

from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix, store_checkpoint

from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer

@dataclass
class ChannelTensor:
    data: torch.Tensor          # shape (T, D)
    columns: list[str]          # original column names
    groups: list[list[str]]     # same structure you used before, optional
    
    def sample(self, n_repeat, rng, groups=None):
        """Fast channel sampling."""
        if groups is None:
            groups = self.groups

        col_arr = np.array(self.columns)
        group_indices = [np.flatnonzero(np.isin(col_arr, g)) for g in groups]
        if any(len(idx) == 0 for idx in group_indices):
            raise ValueError("Empty group for sample()")

        out = []
        for _ in range(n_repeat):
            chosen = [rng.choice(idx) for idx in group_indices]
            out.append(self.data[:, chosen])  # (T, G)

        # -> (n_repeat, T, G)
        return torch.stack(out, dim=0)

# def sample_channels(data, n_repeat, groups, rng):
#     new_data = []
#     for _ in range(n_repeat):
#         selected = []

#         for g in groups:
#             available = [col for col in g if col in data.columns]
#             if not available:
#                 raise ValueError(f"No available columns in group {g}")
            
#             selected.append(rng.choice(available))
#         new_data.append(torch.from_numpy(data[selected].values).float())
#     return new_data
    
class GroupedChanelMulticlassTrainer(MulticlassTrainer):
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
    def get_item(groups:list[list[str]], patient, time, data, target, target_extra = None, percentage:float = 0.5):
        try:
            freq = pd.to_timedelta(target.index.freq).total_seconds()
            targets = torch.tensor(target.sum().to_numpy())
            target = MulticlassTrainer.target_to_multiclass(targets, None, len(target)*freq*percentage, True)

            x_tensor = torch.from_numpy(data.values.astype(np.float32))
            item = {"patient":patient, "time":time, "target":target, "data": ChannelTensor(x_tensor, columns = list(data.columns), groups=groups)}

            if target_extra is not None:
                freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
                targets = torch.tensor(target_extra.sum().to_numpy())
                target_extra = MulticlassTrainer.target_to_multiclass(targets, None, len(target_extra)*freq*percentage, True)
                item["target_extra"] = target_extra
            return item
        except Exception as e:
            pass
        return None
    
    #     try:
    #         freq = pd.to_timedelta(target.index.freq).total_seconds()
    #         targets = torch.tensor(target.sum().to_numpy())
    #         target = MulticlassTrainer.target_to_multiclass(targets, None, len(target)*freq*percentage, True)

    #         item = {"patient":patient, "time":time, "target":target}

    #         if target_extra is not None:
    #             freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
    #             targets = torch.tensor(target_extra.sum().to_numpy())
    #             target_extra = MulticlassTrainer.target_to_multiclass(targets, None, len(target_extra)*freq*percentage, True)
    #             item["target_extra"] = target_extra
            
    #         # TODO Make 1 configurable
    #         item["data"] = sample_channels(data, 1, groups, rng)

    #         return item 
    #     except Exception as e:
    #         pass
    #     return None

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
                    new_x = []
                    for x in batch["data"]:
                        new_x.append(x.sample(self.n_repeat_train, self.rng))

                    x = torch.vstack(new_x).to(device)
                    x = model.apply_preprocessors(x, idx) 
                    model.preprocessors[idx].update(x)
                    logger.progress_advance(batch_size)
            else:
                # No warmup required -> Set tqdm bar to final value directly                
                logger.progress_advance(prog_size)
            logger.progress_close()

        return model

    def run_epoch(self, loader, opt, model, prefix=""):
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        nc = self.num_classes
        
        loss_sum = 0
        cm_sum = np.zeros((nc, nc), dtype=np.int64)
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        if mode == "train":
            n_repeat = self.n_repeat_train
        else:
            n_repeat = self.n_repeat_test

        for batch in loader:
            y = batch["target"].to(self.device)
            
            if opt is not None: opt.zero_grad(set_to_none=True)

            all_logits = []
            for x in batch["data"]:
                tmp_x = x.sample(n_repeat, self.rng, self.groups).to(self.device)
                logits = model(tmp_x).mean(dim=0)
                all_logits.append(logits)

            logits = torch.vstack(all_logits)
            loss = self._loss(logits, y)
            
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
    