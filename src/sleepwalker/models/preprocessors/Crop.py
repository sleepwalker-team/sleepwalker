"""Window-cropping preprocessor for `[B, T, C]` tensors."""

from typing import Literal
import pandas as pd
import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class Crop(Preprocessor):
    """Crop a fixed-length segment from each sequence.

    Args:
        total_input: Desired crop duration.
        sampling_rate: Sampling interval as a pandas-compatible timedelta.
        where: Crop anchor, one of `"left"`, `"middle"`, or `"right"`.
    """
    def __init__(self, total_input, sampling_rate, where: Literal["left", "middle", "right"] = "middle", channels=None, **kwargs):
        super().__init__(channels=channels)
        self.total_input = pd.to_timedelta(total_input)
        self.sampling_rate = pd.to_timedelta(sampling_rate)
        self.where = where
        self.len = int(self.total_input / self.sampling_rate)  # ensure integer length

    def _update(self, data:torch.Tensor):
        """No-op warmup hook because cropping is stateless."""
        ...

    def requires_warmup(self) -> bool:
        """Return whether this preprocessor requires warmup."""
        return False 

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Crop one fixed-length segment from each input sequence."""
        # data: (batch_size, num_samples, num_features)
        B, T, D = data.shape
        L = self.len

        if L > T:
            raise ValueError(f"Requested crop length {L} exceeds input length {T}.")

        if self.where == "middle":
            start = (T - L) // 2
        elif self.where == "left":
            start = 0
        elif self.where == "right":
            start = T - L
        else:
            raise ValueError(f"Invalid crop location: {self.where}")

        end = start + L
        return data[:, start:end, :]
    
