"""Repeated-view model execution."""

import torch
from torch import nn

def average_repeated_outputs(outputs, batch_size: int, n_repeat: int):
    if isinstance(outputs, torch.Tensor):
        return outputs.reshape(batch_size, n_repeat, *outputs.shape[1:]).mean(dim=1)
    if isinstance(outputs, dict):
        return {key: average_repeated_outputs(value, batch_size, n_repeat) for key, value in outputs.items()}
    raise TypeError(f"Cannot average repeated outputs of type {type(outputs).__name__}.")


class RepeatedViewModel(nn.Module):
    """Execute grouped views and average logits for each logical sample."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        values = list(x.values()) if isinstance(x, dict) else [x]
        if not values or any(not isinstance(value, torch.Tensor) or value.ndim < 3 for value in values):
            raise ValueError("RepeatedViewModel expects tensor inputs shaped [B, R, ...].")
        batch_size, n_repeat = values[0].shape[:2]
        if any(value.shape[:2] != (batch_size, n_repeat) for value in values[1:]):
            raise ValueError("Repeated mapping inputs disagree on batch size or view count.")
        flattened = {key: value.reshape(batch_size * n_repeat, *value.shape[2:]) for key, value in x.items()} if isinstance(x, dict) else x.reshape(batch_size * n_repeat, *x.shape[2:])
        outputs = self.model(flattened)
        return average_repeated_outputs(outputs, batch_size, n_repeat)
