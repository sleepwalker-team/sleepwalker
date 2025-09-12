from typing import Literal
import pandas as pd
import torch

from sleepwalker.event.preprocessing.Preprocesser import Preprocessor

class Crop(Preprocessor):
    def __init__(self, total_input, sampling_rate, where: Literal["left", "middle", "right"] = "middle", **kwargs):
        self.total_input = pd.to_timedelta(total_input)
        self.sampling_rate = pd.to_timedelta(sampling_rate)
        self.where = where
        self.len = int(self.total_input / self.sampling_rate)  # ensure integer length

    def update(self, data:torch.Tensor):
        ...

    def requires_warmup(self) -> bool:
        return False 

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
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
    