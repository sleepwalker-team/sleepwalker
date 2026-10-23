"""MultiRocket as an encoder: [B, T, C] windows in, [B, n_features] features out."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.MiniRocket import golden_ratio_quantiles, minirocket_dilations
from sleepwalker.utils import logger


class MultiRocket(BaseModel, EmbeddingModel):
    """MultiRocket (Tan et al., 2022) encoder: PPV, MPV, MIPV and LSPV of random convolutions of each window and its first difference.

    Args:
        ts_len: Number of samples per input window, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        num_features: Number of output features, rounded down to a multiple of 8 * 84 (4 operators x 2 representations x 84 kernels).
        max_dilations_per_kernel: Upper limit on the number of dilations.
        fit_windows: Training windows the biases are fitted on during warmup; None uses all windows the warmup delivers.
        random_state: Seed for the channel combinations and the fitting windows.
    """

    def __init__(self, *, ts_len: int, n_channels: int, num_features: int = 50_000, max_dilations_per_kernel: int = 32, fit_windows: int | None = 1024, random_state: int = 0):
        super().__init__()
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.fit_windows = fit_windows
        self.generator = np.random.default_rng(random_state)
        if self.ts_len < 10:
            raise ValueError(f"MultiRocket needs windows of at least 10 samples, got {self.ts_len}.")
        if num_features < 8 * 84:
            raise ValueError(f"MultiRocket needs num_features >= {8 * 84}, got {num_features}.")
        biases_per_representation = int(num_features) // 8
        self.dilations, self.features_per_dilation = minirocket_dilations(self.ts_len, biases_per_representation, max_dilations_per_kernel)
        self.diff_dilations, self.diff_features_per_dilation = minirocket_dilations(self.ts_len - 1, biases_per_representation, max_dilations_per_kernel)

        positions = torch.combinations(torch.arange(9), 3)  # weight 2 at 3 of 9 positions, -1 elsewhere
        self.register_buffer("kernels", (-torch.ones(84, 9)).scatter_(1, positions, 2.0))
        self.register_buffer("channel_masks", self.draw_channel_masks(len(self.dilations)))  # [dilations, 84, C]
        self.register_buffer("diff_channel_masks", self.draw_channel_masks(len(self.diff_dilations)))
        self.register_buffer("biases", torch.zeros(84 * sum(self.features_per_dilation)))
        self.register_buffer("diff_biases", torch.zeros(84 * sum(self.diff_features_per_dilation)))
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

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """Fit the biases of both representations on training windows [N, T, C]."""
        x = data.transpose(1, 2).float()
        self.biases = self.fit_biases(x, self.dilations, self.features_per_dilation, self.channel_masks)
        self.diff_biases = self.fit_biases(torch.diff(x, dim=-1), self.diff_dilations, self.diff_features_per_dilation, self.diff_channel_masks)
        self.fitted = torch.tensor(True, device=x.device)
        logger.info(f"MultiRocket fitted on {len(x)} windows.")

    def fit_biases(self, x: torch.Tensor, dilations: list[int], features_per_dilation: list[int], channel_masks: torch.Tensor) -> torch.Tensor:
        """Per dilation and kernel: quantiles of the kernel's convolution of one random training window.

        The quantiles run through one low-discrepancy sequence over all dilations, kernels and biases, so every kernel
        gets its own quantile positions.
        """
        quantiles = golden_ratio_quantiles(84 * sum(features_per_dilation)).to(x.device).split([84 * n for n in features_per_dilation])
        kernels = torch.arange(84, device=x.device)
        biases = []
        for dilation, n, masks, dilation_quantiles in zip(dilations, features_per_dilation, channel_masks, quantiles):
            windows = torch.as_tensor(self.generator.integers(len(x), size=84), device=x.device)  # one window per kernel
            samples = convolve(x[windows], self.kernels, masks, dilation)[kernels, kernels]  # [84, T]: kernel k on its own window
            biases += [torch.quantile(sample, q) for sample, q in zip(samples, dilation_quantiles.view(84, n))]
        return torch.cat(biases)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, n_features]: the features of the window, then those of its first difference."""
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"MultiRocket expects [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        if not self.fitted:
            logger.warning("MultiRocket is not fitted yet; features use zero biases until the trainer's warmup has run.")
        x = x.transpose(1, 2).float()
        window = self.representation_features(x, self.dilations, self.features_per_dilation, self.channel_masks, self.biases)
        diff = self.representation_features(torch.diff(x, dim=-1), self.diff_dilations, self.diff_features_per_dilation, self.diff_channel_masks, self.diff_biases)
        return torch.cat([window, diff], dim=1)

    def representation_features(self, x: torch.Tensor, dilations: list[int], features_per_dilation: list[int], channel_masks: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """[B, C, L] -> [B, 4 * 84 * sum(features_per_dilation)]: PPV, MPV, MIPV and LSPV of every dilation, kernel and bias."""
        biases = torch.split(biases, [84 * n for n in features_per_dilation])
        features = []
        for i, (dilation, n) in enumerate(zip(dilations, features_per_dilation)):
            convolved = convolve(x, self.kernels, channel_masks[i], dilation)  # [B, 84, L]
            kernel_biases = biases[i].view(84, n)
            padding, first = 4 * dilation, i % 2
            # half of the kernels keep the zero-padded edges, the other half only count positions inside the window
            halves = [(convolved[:, first::2], kernel_biases[first::2]), (convolved[:, 1 - first::2, padding:-padding], kernel_biases[1 - first::2])]
            for output, half_biases in halves:
                features += pooling(output, half_biases)  # PPV, MPV, MIPV, LSPV, each [B, 42, n]
        return torch.cat([f.flatten(1) for f in features], dim=1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return 4 * 84 * (sum(self.features_per_dilation) + sum(self.diff_features_per_dilation))

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}

    def draw_channel_masks(self, n_dilations: int) -> torch.Tensor:
        """One random channel combination per dilation and kernel: [dilations, 84, C] of 0/1."""
        masks = torch.zeros(n_dilations, 84, self.n_channels)
        max_exponent = np.log2(min(self.n_channels, 9) + 1)
        for i in range(n_dilations):
            counts = (2 ** self.generator.uniform(0, max_exponent, 84)).astype(np.int32)
            for kernel, count in enumerate(counts):
                masks[i, kernel, self.generator.choice(self.n_channels, count, replace=False)] = 1
        return masks


def convolve(x: torch.Tensor, kernels: torch.Tensor, channel_masks: torch.Tensor, dilation: int) -> torch.Tensor:
    """[B, C, L] -> [B, 84, L]: all kernels at one dilation, each on its own channels, with zero padding."""
    weights = kernels[:, None, :] * channel_masks[:, :, None]  # [84, C, 9]
    return F.conv1d(x, weights, padding=4 * dilation, dilation=dilation)


@torch.compile(dynamic=True)  # fusing the steps keeps z and the masks out of memory
def pooling(output: torch.Tensor, biases: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """[B, K, L] convolution outputs and [K, n] biases -> PPV, MPV, MIPV and LSPV of z = output - bias, each [B, K, n]."""
    z = output[:, :, None, :] - biases[None, :, :, None]  # [B, K, n, L]
    return ppv(z), mpv(z), mipv(z), lspv(z)


def ppv(z: torch.Tensor) -> torch.Tensor:
    """Proportion of positive values along the last dimension."""
    return (z > 0).float().mean(-1)


def mpv(z: torch.Tensor) -> torch.Tensor:
    """Mean of the positive values along the last dimension; 0 if there are none."""
    count = (z > 0).sum(-1)
    return torch.where(count > 0, z.clamp(min=0).sum(-1) / count.clamp(min=1), 0.0)  # non-positive values add nothing


def mipv(z: torch.Tensor) -> torch.Tensor:
    """Mean (0-based) index of the positive values along the last dimension; -1 if there are none."""
    positive = z > 0
    count = positive.sum(-1)
    index = torch.arange(z.shape[-1], device=z.device, dtype=z.dtype)
    return torch.where(count > 0, (positive * index).sum(-1) / count.clamp(min=1), -1.0)


def lspv(z: torch.Tensor) -> torch.Tensor:
    """Length of the longest stretch of consecutive positive values along the last dimension."""
    positive = z > 0
    index = torch.arange(z.shape[-1], device=z.device, dtype=torch.int32)  # int32 makes the cummax faster
    last_non_positive = torch.where(positive, -1, index).cummax(-1).values  # latest position so far that is not positive
    return (index - last_non_positive).amax(-1).to(z.dtype)  # length of the positive stretch ending at each position
