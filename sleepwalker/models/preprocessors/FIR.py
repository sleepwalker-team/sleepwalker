import itertools
from typing import Literal
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class FIR(Preprocessor):
    def __init__(self, sampling_rate, channels, filter_params, zero_phase=False, **kwargs):
        fs = 1.0/sampling_rate#.total_seconds()

        self.zero_phase = zero_phase
        self.groups = channels
        self.channels = list(itertools.chain.from_iterable(channels))
        self.num_channels = len(self.channels)

        # Validate parameter config
        if set(filter_params.keys()) != set(self.channels):
            missing = set(self.channels) - set(filter_params.keys())
            raise ValueError(f"Filter parameters missing for channels: {missing}")
        self.per_channel_params = filter_params

        # Design filters per channel
        self.coefficients = []
        for ch in self.channels:
            params = self.per_channel_params[ch]
            fir = scipy.signal.firwin(**params, fs=fs)
            self.coefficients.append(fir)

    def requires_warmup(self) -> bool:
        return False

    def update(self, data:torch.Tensor):
        ...

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # data: (batch_size, num_samples, num_features)
        B, T, D = data.shape
        assert D == self.num_channels, "Mismatch between provided channels and data features."

        data_np = data.detach().cpu().numpy()
        filtered_np = np.zeros_like(data_np)

        for b in range(B):
            for d, coeff in enumerate(self.coefficients):
                # TODO Using scipy here can be a performance issue due to two data conversion: torch -> numpy -> torch. Maybe we should update this code if performance becomes an issue
                if self.zero_phase:
                    filtered_np[b, :, d] = scipy.signal.filtfilt(coeff, [1.0], data_np[b, :, d])
                else:
                    filtered_np[b, :, d] = scipy.signal.lfilter(coeff, [1.0], data_np[b, :, d])

        return torch.tensor(filtered_np, dtype=data.dtype, device=data.device)
    