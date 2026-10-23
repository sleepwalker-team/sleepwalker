"""MiniRocket as an encoder: [B, T, C] windows in, [B, n_features] features out."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.utils import logger


# Adapted from tsai's MINIROCKET_Pytorch.py (Malcolm McLean, Ignacio Oguiza; https://github.com/timeseriesAI/tsai,
# Apache License 2.0), with the channel combinations folded into the convolution weights.
class MiniRocket(BaseModel, EmbeddingModel):
    """MiniRocket (Dempster et al., 2021) encoder: proportions of positive values of random convolutions.

    Args:
        ts_len: Number of samples per input window, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        num_features: Number of output features, rounded down to a multiple of 84.
        max_dilations_per_kernel: Upper limit on the number of dilations.
        fit_windows: Training windows the biases are fitted on during warmup; None uses all windows the warmup delivers.
        random_state: Seed for the channel combinations and the fitting windows.
    """

    def __init__(self, *, ts_len: int, n_channels: int, num_features: int = 10_000, max_dilations_per_kernel: int = 32, fit_windows: int | None = 1024, random_state: int = 0):
        super().__init__()
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.fit_windows = fit_windows
        self.generator = np.random.default_rng(random_state)
        self.dilations, self.features_per_dilation = minirocket_dilations(self.ts_len, num_features, max_dilations_per_kernel)

        positions = torch.combinations(torch.arange(9), 3)  # weight 2 at 3 of 9 positions, -1 elsewhere
        self.register_buffer("kernels", (-torch.ones(84, 9)).scatter_(1, positions, 2.0))
        self.register_buffer("channel_masks", torch.stack([self.channel_mask() for _ in self.dilations]))  # [dilations, 84, C]
        self.register_buffer("biases", torch.zeros(84, sum(self.features_per_dilation)))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def warmup_preprocessor(self, data_loader, device: str):
        """Fit the biases on the first fit_windows training windows, then stop. The trainer calls this before training."""
        self.to(device)
        if self.fitted:
            return self
        windows, collected = [], 0
        for batch in data_loader:
            if batch is None:
                continue
            windows.append(batch["data"] if self.fit_windows is None else batch["data"][: self.fit_windows - collected])
            collected += len(windows[-1])
            if self.fit_windows is not None and collected >= self.fit_windows:
                break
        self.fit(torch.cat(windows).to(device))
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """PPV features per dilation; before fitting it warns and uses zero biases."""
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"MiniRocket expects [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        if not self.fitted:
            logger.warning("MiniRocket is not fitted yet; features use zero biases until the trainer's warmup has run.")
        x = x.transpose(1, 2).float()
        biases = torch.split(self.biases, self.features_per_dilation, dim=1)
        features = []
        for i, dilation in enumerate(self.dilations):
            padding, first = 4 * dilation, i % 2
            convolved = self.convolve(x, i, dilation)
            features.append(ppv(convolved[:, first::2], biases[i][first::2]))  # these kernels use the padded output
            features.append(ppv(convolved[:, 1 - first::2, padding:-padding], biases[i][1 - first::2]))
        return torch.cat(features, dim=1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return 84 * sum(self.features_per_dilation)

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}

    def fit(self, data: torch.Tensor):
        """Biases: quantiles of each kernel's convolution of one random training window."""
        x = data.transpose(1, 2).float()
        kernels = torch.arange(84, device=x.device)
        biases = []
        for i, dilation in enumerate(self.dilations):
            windows = torch.as_tensor(self.generator.integers(len(x), size=84), device=x.device)  # one window per kernel
            samples = self.convolve(x[windows], i, dilation)[kernels, kernels]  # [84, T]: kernel k on its own window
            quantiles = golden_ratio_quantiles(self.features_per_dilation[i]).to(x.device)
            biases.append(torch.quantile(samples, quantiles, dim=1).T)
        self.biases = torch.cat(biases, dim=1)
        self.fitted = torch.tensor(True, device=x.device)
        logger.info(f"MiniRocket fitted on {len(x)} windows.")

    def channel_mask(self) -> torch.Tensor:
        """One random channel combination per kernel: [84, C] of 0/1."""
        max_exponent = np.log2(min(self.n_channels, 9) + 1)
        counts = (2 ** self.generator.uniform(0, max_exponent, 84)).astype(np.int32)
        mask = torch.zeros(84, self.n_channels)
        for kernel, count in enumerate(counts):
            mask[kernel, self.generator.choice(self.n_channels, count, replace=False)] = 1
        return mask

    def convolve(self, x: torch.Tensor, i: int, dilation: int) -> torch.Tensor:
        """[B, C, T] -> [B, 84, T]: one convolution with the channel mask folded into the weights."""
        weights = self.kernels[:, None, :] * self.channel_masks[i][:, :, None]  # [84, C, 9]
        return F.conv1d(x, weights, padding=4 * dilation, dilation=dilation)


@torch.compile(dynamic=True)  # fusing the steps keeps the comparison mask out of memory
def ppv(C, bias):
    """Proportion of positive values: the share of time steps where the convolution exceeds each bias."""
    C = C.unsqueeze(-1)
    bias = bias.view(1, bias.shape[0], 1, bias.shape[1])
    return (C > bias).float().mean(2).flatten(1)


def golden_ratio_quantiles(n):
    """MiniRocket's low-discrepancy quantiles: (k * golden ratio) mod 1 for k = 1..n."""
    return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)]).float()


def minirocket_dilations(input_length, num_features, max_dilations_per_kernel):
    """MiniRocket's dilations and features per dilation for an input length."""
    num_features_per_kernel = num_features // 84
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel
    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32), return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)
    remainder = num_features_per_kernel - num_features_per_dilation.sum()
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)
    return [int(d) for d in dilations], [int(n) for n in num_features_per_dilation]
