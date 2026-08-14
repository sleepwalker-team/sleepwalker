"""Shared model execution used by training, testing, and packaged inference."""

from collections.abc import Iterator

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

    def forward(self, x: torch.Tensor):
        if x.ndim < 3:
            raise ValueError(f"RepeatedViewModel expects [B, R, ...], got {tuple(x.shape)}.")
        batch_size, n_repeat = x.shape[:2]
        outputs = self.model(x.reshape(batch_size * n_repeat, *x.shape[2:]))
        return average_repeated_outputs(outputs, batch_size, n_repeat)


def execute_batches(model: nn.Module, loader, device: str | torch.device) -> Iterator[tuple[object, dict]]:
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            yield model(batch["data"].to(device, non_blocking=True)), batch
