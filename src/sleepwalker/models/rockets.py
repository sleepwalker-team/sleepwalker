"""Rocket, MiniRocket, MultiRocket and Hydra as encoders: [B, T, C] windows in, [B, n_features] features out."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.utils import logger


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

    TAPS = 11  # longest kernel; shorter kernels are zero-padded to this length

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

        all_weights = torch.zeros(len(order), self.n_channels, self.TAPS)
        valid = torch.zeros(len(order), self.ts_len, dtype=torch.bool)
        for i, (length, kernel, used, dilation, is_padded) in enumerate(zip(lengths, weights, channels, dilations, padded)):
            start = (self.TAPS - length) // 2  # centred, so padding (TAPS // 2) * dilation gives the kernel's own output
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
            convolved = F.conv1d(x, weights[i], biases[i], padding=(self.TAPS // 2) * dilation, dilation=dilation)  # [B, kernels, T]
            features.append(rocket_pooling(convolved, valid[i]))  # [B, kernels, 2]
        return torch.cat(features, dim=1)[:, self.drawing_order].flatten(1)  # (ppv, max) per kernel

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return 2 * self.num_kernels

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}


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

        self.register_buffer("kernels", minirocket_kernels())
        self.register_buffer("channel_masks", draw_channel_masks(self.generator, len(self.dilations), self.n_channels))  # [dilations, 84, C]
        self.register_buffer("biases", torch.zeros(84 * sum(self.features_per_dilation)))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def warmup_preprocessor(self, data_loader, device: str):
        """Fit the biases on the first fit_windows training windows, then stop. The trainer calls this before training."""
        self.to(device)
        if not self.fitted:
            self.fit(collect_windows(data_loader, self.fit_windows).to(device))
        return self

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """Fit the biases on training windows [N, T, C]."""
        x = data.transpose(1, 2).float()
        self.biases = fit_biases(x, self.kernels, self.dilations, self.features_per_dilation, self.channel_masks, self.generator)
        self.fitted = torch.tensor(True, device=x.device)
        logger.info(f"MiniRocket fitted on {len(x)} windows.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, n_features]: PPV of every dilation, kernel and bias; before fitting it warns and uses zero biases."""
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"MiniRocket expects [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        if not self.fitted:
            logger.warning("MiniRocket is not fitted yet; features use zero biases until the trainer's warmup has run.")
        x = x.transpose(1, 2).float()
        biases = torch.split(self.biases, [84 * n for n in self.features_per_dilation])
        features = []
        for i, (dilation, n) in enumerate(zip(self.dilations, self.features_per_dilation)):
            convolved = convolve(x, self.kernels, self.channel_masks[i], dilation)  # [B, 84, T]
            kernel_biases = biases[i].view(84, n)
            padding, first = 4 * dilation, i % 2
            # half of the kernels keep the zero-padded edges, the other half only count positions inside the window
            features.append(minirocket_pooling(convolved[:, first::2], kernel_biases[first::2]))  # PPV, [B, 42, n]
            features.append(minirocket_pooling(convolved[:, 1 - first::2, padding:-padding], kernel_biases[1 - first::2]))
        return torch.cat([f.flatten(1) for f in features], dim=1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return 84 * sum(self.features_per_dilation)

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}


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

        self.register_buffer("kernels", minirocket_kernels())
        self.register_buffer("channel_masks", draw_channel_masks(self.generator, len(self.dilations), self.n_channels))  # [dilations, 84, C]
        self.register_buffer("diff_channel_masks", draw_channel_masks(self.generator, len(self.diff_dilations), self.n_channels))
        self.register_buffer("biases", torch.zeros(84 * sum(self.features_per_dilation)))
        self.register_buffer("diff_biases", torch.zeros(84 * sum(self.diff_features_per_dilation)))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def warmup_preprocessor(self, data_loader, device: str):
        """Fit the biases on the first fit_windows training windows, then stop. The trainer calls this before training."""
        self.to(device)
        if not self.fitted:
            self.fit(collect_windows(data_loader, self.fit_windows).to(device))
        return self

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """Fit the biases of both representations on training windows [N, T, C]."""
        x = data.transpose(1, 2).float()
        self.biases = fit_biases(x, self.kernels, self.dilations, self.features_per_dilation, self.channel_masks, self.generator)
        self.diff_biases = fit_biases(torch.diff(x, dim=-1), self.kernels, self.diff_dilations, self.diff_features_per_dilation, self.diff_channel_masks, self.generator)
        self.fitted = torch.tensor(True, device=x.device)
        logger.info(f"MultiRocket fitted on {len(x)} windows.")

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
            features += multirocket_pooling(convolved[:, first::2], kernel_biases[first::2])  # PPV, MPV, MIPV, LSPV, each [B, 42, n]
            features += multirocket_pooling(convolved[:, 1 - first::2, padding:-padding], kernel_biases[1 - first::2])
        return torch.cat([f.flatten(1) for f in features], dim=1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return 4 * 84 * (sum(self.features_per_dilation) + sum(self.diff_features_per_dilation))

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}


class Hydra(BaseModel, EmbeddingModel):
    """Hydra (Dempster et al., 2023) encoder: counts of the winning kernels in groups of competing random convolutions of each window and its first difference.

    With several channels, each group convolves the sum of its own random channels.

    Args:
        ts_len: Number of samples per input window, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        num_groups: Groups of competing kernels per dilation; half of them see the first difference.
        kernels_per_group: Number of kernels competing in each group.
        max_channels_per_group: Upper limit on the channels a group sums when there are several.
        random_state: Seed for the kernels and the channels of each group.
    """

    def __init__(self, *, ts_len: int, n_channels: int, num_groups: int = 64, kernels_per_group: int = 8, max_channels_per_group: int = 8, random_state: int = 0):
        super().__init__()
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.kernels_per_group = int(kernels_per_group)
        if self.ts_len < 9:
            raise ValueError(f"Hydra needs windows of at least 9 samples, got {self.ts_len}.")
        self.representations = min(2, int(num_groups))  # the window and, with two or more groups, its first difference
        self.groups_per_representation = int(num_groups) // self.representations
        max_exponent = np.log2((self.ts_len - 1) / (9 - 1))  # the widest dilation whose kernel of length 9 still fits the window
        self.dilations = [2 ** i for i in range(int(max_exponent) + 1)]

        # Drawn in the order of the authors' implementation, so a seed gives its kernels and channels.
        generator = torch.Generator().manual_seed(int(random_state))
        kernels = torch.randn(len(self.dilations), self.representations, self.groups_per_representation * self.kernels_per_group, 9, generator=generator)
        kernels = kernels - kernels.mean(-1, keepdim=True)
        kernels = kernels / kernels.abs().sum(-1, keepdim=True)  # mean 0, absolute values summing to 1
        if self.n_channels == 1:
            counts = torch.ones(len(self.dilations), self.representations, self.groups_per_representation, 1)
        else:
            # each group sums C // 2 random channels (2 to max_channels_per_group), picked with replacement
            per_group = int(np.clip(self.n_channels // 2, 2, max_channels_per_group))
            picks = torch.stack([torch.randint(0, self.n_channels, (self.representations, self.groups_per_representation, per_group), generator=generator) for _ in self.dilations])
            counts = F.one_hot(picks, self.n_channels).sum(-2).float()  # [dilations, 2, groups, C]: how often a group picked each channel
        self.register_buffer("kernels", kernels)  # [dilations, 2, groups * kernels_per_group, 9]
        self.register_buffer("channel_masks", counts.repeat_interleave(self.kernels_per_group, dim=2))  # a group's counts for each of its kernels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, n_features]: soft and hard counts of every dilation, representation, group and kernel."""
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"Hydra expects [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        x = x.transpose(1, 2).float()
        representations = [x, torch.diff(x, dim=-1)] if self.representations == 2 else [x]
        features = []
        for i, dilation in enumerate(self.dilations):
            for j, series in enumerate(representations):
                convolved = convolve(series, self.kernels[i, j], self.channel_masks[i, j], dilation)  # [B, groups * kernels_per_group, L]
                features += hydra_pooling(convolved.view(len(x), self.groups_per_representation, self.kernels_per_group, -1))  # soft and hard counts, each [B, groups, kernels_per_group]
        return torch.cat([f.flatten(1) for f in features], dim=1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return len(self.dilations) * self.representations * 2 * self.groups_per_representation * self.kernels_per_group

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}


# Rocket's pooling


@torch.compile(dynamic=True)  # fusing the steps keeps the masks out of memory
def rocket_pooling(convolved: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B, K, T] convolution outputs and [K, T] counted positions -> [B, K, 2]: PPV and max of each kernel."""
    proportion = ((convolved > 0) & valid).sum(-1) / valid.sum(-1)  # share of counted positions above zero
    maximum = convolved.masked_fill(~valid, float("-inf")).amax(-1)
    return torch.stack([proportion, maximum], dim=-1)


# MiniRocket's kernels, dilations, channel combinations and biases; MultiRocket uses them for both its representations,
# Hydra uses convolve


def minirocket_kernels() -> torch.Tensor:
    """MiniRocket's 84 kernels of length 9: weight 2 at 3 of the 9 positions, -1 elsewhere."""
    positions = torch.combinations(torch.arange(9), 3)
    return (-torch.ones(84, 9)).scatter_(1, positions, 2.0)


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


def draw_channel_masks(generator: np.random.Generator, n_dilations: int, n_channels: int) -> torch.Tensor:
    """One random channel combination per dilation and kernel: [dilations, 84, C] of 0/1."""
    masks = torch.zeros(n_dilations, 84, n_channels)
    max_exponent = np.log2(min(n_channels, 9) + 1)
    for i in range(n_dilations):
        counts = (2 ** generator.uniform(0, max_exponent, 84)).astype(np.int32)
        for kernel, count in enumerate(counts):
            masks[i, kernel, generator.choice(n_channels, count, replace=False)] = 1
    return masks


def convolve(x: torch.Tensor, kernels: torch.Tensor, channel_masks: torch.Tensor, dilation: int) -> torch.Tensor:
    """[B, C, L] -> [B, K, L]: K kernels of length 9 at one dilation, each on its channels weighted by its mask, with zero padding."""
    weights = kernels[:, None, :] * channel_masks[:, :, None]  # [K, C, 9]
    return F.conv1d(x, weights, padding=4 * dilation, dilation=dilation)


def fit_biases(x: torch.Tensor, kernels: torch.Tensor, dilations: list[int], features_per_dilation: list[int], channel_masks: torch.Tensor, generator: np.random.Generator) -> torch.Tensor:
    """Per dilation and kernel: quantiles of the kernel's convolution of one random training window of x [N, C, L].

    The quantiles run through one low-discrepancy sequence over all dilations, kernels and biases, so every kernel
    gets its own quantile positions.
    """
    quantiles = golden_ratio_quantiles(84 * sum(features_per_dilation)).to(x.device).split([84 * n for n in features_per_dilation])
    kernel_index = torch.arange(84, device=x.device)
    biases = []
    for dilation, n, masks, dilation_quantiles in zip(dilations, features_per_dilation, channel_masks, quantiles):
        windows = torch.as_tensor(generator.integers(len(x), size=84), device=x.device)  # one window per kernel
        samples = convolve(x[windows], kernels, masks, dilation)[kernel_index, kernel_index]  # [84, L]: kernel k on its own window
        biases += [torch.quantile(sample, q) for sample, q in zip(samples, dilation_quantiles.view(84, n))]
    return torch.cat(biases)


def golden_ratio_quantiles(n):
    """MiniRocket's low-discrepancy quantiles: (k * golden ratio) mod 1 for k = 1..n."""
    return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)]).float()


def collect_windows(data_loader, fit_windows: int | None) -> torch.Tensor:
    """The first fit_windows windows [N, T, C] the loader delivers; None takes all of them."""
    windows, collected = [], 0
    for batch in data_loader:
        if batch is None:
            continue
        windows.append(batch["data"] if fit_windows is None else batch["data"][: fit_windows - collected])
        collected += len(windows[-1])
        if fit_windows is not None and collected >= fit_windows:
            break
    return torch.cat(windows)


# MiniRocket's and MultiRocket's pooling of z = convolution output - bias


@torch.compile(dynamic=True)  # fusing the steps keeps z out of memory
def minirocket_pooling(output: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """[B, K, L] convolution outputs and [K, n] biases -> PPV of z = output - bias, [B, K, n]."""
    return ppv(output[:, :, None, :] - biases[None, :, :, None])


@torch.compile(dynamic=True)  # fusing the steps keeps z and the masks out of memory
def multirocket_pooling(output: torch.Tensor, biases: torch.Tensor) -> tuple[torch.Tensor, ...]:
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


# Hydra's pooling


@torch.compile(dynamic=True)  # fusing the steps keeps the comparison masks out of memory
def hydra_pooling(convolved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, G, K, L] responses of G groups of K competing kernels -> soft and hard counts, each [B, G, K].

    At every time step, the kernel with the largest response in each group adds that response to its soft count, and
    the kernel with the smallest response adds 1 to its hard count.
    """
    max_values, max_indices = convolved.max(2)  # [B, G, L]
    min_indices = convolved.argmin(2)
    kernel_index = torch.arange(convolved.shape[2], device=convolved.device)[:, None]  # [K, 1]
    soft = ((max_indices[:, :, None, :] == kernel_index) * max_values[:, :, None, :]).sum(-1)
    hard = (min_indices[:, :, None, :] == kernel_index).sum(-1).to(convolved.dtype)
    return soft, hard
