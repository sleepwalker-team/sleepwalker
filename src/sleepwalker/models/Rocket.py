"""Rocket as an encoder: [B, T, C] windows in, [B, 2 * num_kernels] features out."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel

TAPS = 11  # longest kernel; shorter kernels are zero-padded to this length


class Rocket(BaseModel, EmbeddingModel):
    """Rocket (Dempster et al., 2020) encoder: PPV and max of random convolutions on z-normalized windows.

    The random biases in (-1, 1) assume z-normalized input.

    Args:
        ts_len: Number of samples per input window, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        num_kernels: Number of random kernels; each gives two features.
        normalise: Z-normalize each window per channel before the convolutions.
        random_state: Seed for the kernels (numpy ``RandomState``, whose stream is fixed across numpy versions).
    """

    def __init__(self, *, ts_len: int, n_channels: int, num_kernels: int = 10_000, normalise: bool = True, random_state: int = 0):
        super().__init__()
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.num_kernels = int(num_kernels)
        self.normalise = normalise
        self.generator = np.random.RandomState(random_state)

        # Drawn in the order and precision of the authors' reference implementation (sktime), so a seed gives its kernels.
        lengths = self.generator.choice([7, 9, 11], self.num_kernels)
        n_used = [int(2 ** self.generator.uniform(0, np.log2(min(self.n_channels, length) + 1))) for length in lengths]  # channels each kernel sees
        weights, channels, biases, dilations, padded = [], [], [], [], []
        for length, count in zip(lengths, n_used):
            kernel = self.generator.normal(0, 1, (count, length)).astype(np.float32)
            weights.append(kernel - kernel.mean(axis=1, keepdims=True))  # mean-centred per channel
            channels.append(self.generator.choice(self.n_channels, count, replace=False))
            biases.append(np.float32(self.generator.uniform(-1, 1)))
            max_exponent = np.log2((self.ts_len - 1) / (length - 1))  # the widest dilation still fits the window
            dilations.append(int(2 ** self.generator.uniform(0, max_exponent)))
            padded.append(self.generator.randint(2) == 1)
        self.set_kernels(lengths, weights, channels, biases, dilations, padded)

    def set_kernels(self, lengths, weights, channels, biases, dilations, padded):
        """Store the kernels sorted by dilation, so that one conv1d per dilation covers a whole group.

        ``valid`` marks, per kernel, the output positions that count: all of them with padding, otherwise only those
        where the kernel lies completely inside the window.
        """
        order = np.argsort(dilations, kind="stable")
        lengths, weights, channels, biases, dilations, padded = ([values[i] for i in order] for values in (lengths, weights, channels, biases, dilations, padded))
        unique, counts = np.unique(dilations, return_counts=True)
        self.dilations, self.kernels_per_dilation = unique.tolist(), counts.tolist()

        all_weights = torch.zeros(len(order), self.n_channels, TAPS)
        valid = torch.zeros(len(order), self.ts_len, dtype=torch.bool)
        for i, (length, kernel, used, dilation, is_padded) in enumerate(zip(lengths, weights, channels, dilations, padded)):
            start = (TAPS - length) // 2  # centred, so padding (TAPS // 2) * dilation gives the kernel's own output
            all_weights[i, used, start : start + length] = torch.as_tensor(kernel, dtype=torch.float32)
            crop = 0 if is_padded else (length - 1) * int(dilation) // 2
            valid[i, crop : self.ts_len - crop] = True
        self.register_buffer("weights", all_weights)
        self.register_buffer("biases", torch.tensor(biases, dtype=torch.float32))
        # valid and the way back to the drawing order follow from the kernels, so they are not saved
        self.register_buffer("valid", valid, persistent=False)
        self.register_buffer("drawing_order", torch.as_tensor(np.argsort(order)), persistent=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, 2 * num_kernels]: PPV and max per kernel, in the order the kernels were drawn."""
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"Rocket expects [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        x = x.transpose(1, 2).float()
        if self.normalise:
            x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, correction=0, keepdim=True) + 1e-8)
        weights = torch.split(self.weights, self.kernels_per_dilation)
        biases = torch.split(self.biases, self.kernels_per_dilation)
        valid = torch.split(self.valid, self.kernels_per_dilation)
        features = []
        for i, dilation in enumerate(self.dilations):
            convolved = F.conv1d(x, weights[i], biases[i], padding=(TAPS // 2) * dilation, dilation=dilation)  # [B, kernels, T]
            features.append(pooling(convolved, valid[i]))  # [B, kernels, 2]
        return torch.cat(features, dim=1)[:, self.drawing_order].flatten(1)  # (ppv, max) per kernel

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return 2 * self.num_kernels

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}


@torch.compile(dynamic=True)  # fusing the steps keeps the masks out of memory
def pooling(convolved: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B, K, T] convolution outputs and [K, T] counted positions -> [B, K, 2]: PPV and max of each kernel."""
    ppv = ((convolved > 0) & valid).sum(-1) / valid.sum(-1)  # share of counted positions above zero
    maximum = convolved.masked_fill(~valid, float("-inf")).amax(-1)
    return torch.stack([ppv, maximum], dim=-1)
