from typing import Optional

import numpy as np
import pandas as pd
import torch


def resolve_multiclass_index(target, default_idx, min_event_seconds, raise_error=True):
    active = target > min_event_seconds
    active_sum = int(active.sum().item())

    if raise_error and active_sum > 1:
        raise ValueError("Multiple active classes found.")
    if active_sum == 1:
        return int(active.nonzero(as_tuple=False).item())
    if active_sum == 0:
        if default_idx is not None:
            return int(default_idx)
        if raise_error:
            raise ValueError("Ambiguous class labels found with no active class and no default_idx.")
        return None
    return None


def build_multiclass_target(target, target_extra=None, percentage: float = 0.5):
    try:
        freq = pd.to_timedelta(target.index.freq).total_seconds()
        targets = torch.tensor(target.sum().to_numpy())
        target_idx = resolve_multiclass_index(targets, None, len(target) * freq * percentage, True)
        target_onehot = torch.zeros(len(targets), dtype=torch.float)
        if target_idx is not None:
            target_onehot[target_idx] = 1.0

        item = {"target": target_onehot}

        if target_extra is not None:
            freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
            targets = torch.tensor(target_extra.sum().to_numpy())
            target_extra_idx = resolve_multiclass_index(targets, None, len(target_extra) * freq * percentage, True)
            target_extra_onehot = torch.zeros(len(targets), dtype=torch.float)
            if target_extra_idx is not None:
                target_extra_onehot[target_extra_idx] = 1.0
            item["target_extra"] = target_extra_onehot

        return item
    except Exception:
        return None
