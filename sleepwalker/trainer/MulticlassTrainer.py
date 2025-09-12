import inspect
from typing import Callable, Optional, Sequence
import torch
from sleepwalker.models.base import BaseModel
from sleepwalker.trainer import Trainer
from sleepwalker.utils import logger
from torch.nn.parallel import DistributedDataParallel 

def to_multiclass(target, default_idx, min_event_seconds, raise_error=True):
    """
    Optimized version: Converts event durations to one-hot based on threshold.
    """
    num_classes = target.shape[-1]
    flat_target = target.view(-1, num_classes)

    active = flat_target > min_event_seconds
    active_sum = active.sum(dim=1)

    if raise_error and torch.any(active_sum > 1):
        raise ValueError("Multiple active classes found.")

    out = torch.zeros_like(flat_target, dtype=torch.float)
    idx = active.float().argmax(dim=1)
    out[torch.arange(flat_target.size(0)), idx] = 1

    if default_idx is not None:
        no_active = active_sum == 0
        if torch.any(no_active):
            out[no_active] = 0
            out[no_active, default_idx] = 1
    elif torch.any(active_sum == 0):
        if raise_error:
            raise ValueError("Ambiguous class labels found with no active class and no default_idx.")

    return out.view(target.shape)

def ensure_loss_signature(func: Callable, check_weight:bool = False) -> None:
    sig = inspect.signature(func)
    params = sig.parameters
    
    # 1) At least two parameters
    if len(params) < 2:
        raise TypeError(
            f"{func.__name__} must accept at least 2 parameters, "
            f"but only has {len(params)}"
        )
    
    if check_weight:
        # 2) Optional 'weight' argument
        weight_param = params.get("weight")
        if weight_param is None:
            raise TypeError(f"{func.__name__} must accept a 'weight' parameter")
        
        if weight_param.default is inspect._empty:
            raise TypeError(f"'weight' in {func.__name__} must be optional (have a default value)")
    
class MulticlassTrainer(Trainer):

    def __init__(self, classes, loss_function:Callable, class_weights:Optional[dict[str,float]] = None, loss_mode:Optional[str] = "regular", **kwargs):
        super().__init__(**kwargs)
        self.loss_mode = loss_mode
        
        self.classes = classes
        if self.loss_mode == "inverse" or self.loss_mode == "inverse-log":
            self.estimate_class_cnts = True
            self.class_cnts = torch.zeros(len(classes)) 
        else:
            self.estimate_class_cnts = False

        if class_weights is not None:
            class_weights = {k.lower():v for k,v in class_weights.items()}
            weight_list = []
            for c in classes:
                if c not in class_weights:
                    logger.warning(f"Did not find class weights for class {c},assuming weight 1")
                    weight_list.append(1.0)
                else:
                    weight_list.append(class_weights[c])
            
            self.user_class_weights = torch.tensor(weight_list)
            self.class_weights = torch.tensor(weight_list)
        else:
            if self.loss_mode in ["inverse", "inverse-log"]:
                self.user_class_weights = torch.ones(len(classes))
                self.class_weights = torch.ones(len(classes))
            else:
                self.user_class_weights = None
                self.class_weights = None

        ensure_loss_signature(loss_function, self.class_weights is not None)
        self.loss_function = loss_function

    def train_step(self, model: BaseModel|DistributedDataParallel, batch: dict):
        x = batch['data']
        y = batch['target']
        logits = model(x)
        if self.class_weights is None or len(set(self.class_weights)) <= 1:
            loss = self.loss_function(logits, y.argmax(1))
        else:
            w_tensor = self.class_weights.to(logits.device)
            loss = self.loss_function(logits, y.argmax(1), weight=w_tensor)
        return loss
    
    def val_step(self, model: BaseModel|DistributedDataParallel, batch: dict):
        x = batch['data']
        y = batch['target']
        logits = model(x)

        if self.class_weights is None or len(set(self.class_weights)) <= 1:
            loss = torch.nn.functional.cross_entropy(logits, y.argmax(1))
        else:
            w_tensor = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, y.argmax(1), weight=w_tensor)
        return loss
    
    def warmup_step(self, model: BaseModel|DistributedDataParallel, batch: dict):
        if self.estimate_class_cnts:
            y = batch["target"]
            target = y.argmax(dim=1)
            idx, cnt = torch.unique(target, return_counts=True)
            
            self.class_cnts = self.class_cnts.to(y.device)
            self.class_weights = self.class_weights.to(y.device)
            self.user_class_weights = self.user_class_weights.to(y.device)

            self.class_cnts[idx] += cnt
            if self.loss_mode == "inverse":
                # Weight classes by their (inverse) occurrence. This can lead to relatively small losses,
                # hence we will also weight normalize it. This is technically not necessary.
                self.class_weights = self.user_class_weights * torch.clamp(1.0 / self.class_cnts, min = 1e-4)
                self.class_weights /= self.class_weights.sum() 
            else:
                """
                See 
                    - MRASleepNet: a multi-resolution attention network for sleep stage classification using single-channel EEG by Rui Yu, Zhuhuang Zhou, Shuicai Wu, Xiaorong Gao and Guangyu Bin in Journal of Neural Engineering 2022, https://github.com/YuRui8879/MRASleepNet/blob/781aee2d2ff1422c598b099081a4c1d7d4bd05d7/DataAdapter/DataAdapter.py#L110
                    - An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG by Eldele et al. in IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING 2021, https://github.com/emadeldeen24/AttnSleep/blob/6b4d2665884628c8a7bb09f36589a8ec0992f8e2/utils/util.py#L62
                """
                total = self.class_cnts.sum()
                factor = 1.0 / self.class_cnts.shape[0]
                mu = factor * self.user_class_weights
                self.class_weights = mu * torch.clamp(torch.log( (total * mu) / self.class_cnts), min=1.0)