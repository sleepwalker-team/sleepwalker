"""Select fixed-duration, forward-delayed slices from arbitrary input channels."""

from collections.abc import Sequence
import math

import torch
from torch import nn


def integral_samples(seconds: float, sampling_frequency: float, *, name: str) -> int:
    """Convert a duration without silently rounding a fractional sample."""
    samples = seconds * sampling_frequency
    if not math.isfinite(samples) or not math.isclose(samples, round(samples), rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{name} must correspond to an integral number of samples.")
    return round(samples)


class DelayedChannelSlices(nn.Module):
    """Build [B,window_samples,len(slices)] from floating-point [B,T,C] input.

    Each (input_channel, delay_seconds) pair selects a window starting at the
    given forward offset. Pair order defines output channel order. Channels
    and delays may repeat; no input channel count, physiology or units are
    assumed. Missing future context is rejected and extra context is allowed.
    Scaling and clipping can be configured separately with ScaleChannels.

    Motivated by DRIVEN's 30-second ABD slice and five forward-delayed SpO2
    slices. Original preparation source:
    https://github.com/LCSB-SCG/DRIVEN/blob/dce8bfadd05d249ca7b55fa5d2d1974b52ba35ec/create_hdf5_from_psg/gen_img_complete.m
    See docs/reference/signal-preprocessing.md for its example configuration.
    Forward delays imply offline latency
    of at least the largest configured delay.
    """

    def __init__(self, sampling_frequency: float, window_duration: float, slices: Sequence[Sequence[float]]):
        super().__init__()
        self.sampling_frequency = float(sampling_frequency)
        self.window_duration = float(window_duration)
        if not math.isfinite(self.sampling_frequency) or self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be finite and positive.")
        if not math.isfinite(self.window_duration) or self.window_duration <= 0:
            raise ValueError("window_duration must be finite and positive.")
        if isinstance(slices, (str, bytes)) or not isinstance(slices, Sequence) or not slices:
            raise ValueError("slices must be a non-empty sequence of (channel, delay_seconds) pairs.")
        configured_slices = []
        for pair in slices:
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("Each slice must be a (channel, delay_seconds) pair.")
            channel, delay = pair
            if not isinstance(channel, int) or isinstance(channel, bool) or channel < 0:
                raise ValueError("Slice channels must be non-negative integer indices.")
            delay = float(delay)
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("Slice delays must be finite and non-negative.")
            configured_slices.append((channel, delay))
        self.slices = tuple(configured_slices)
        self.slice_samples = integral_samples(self.window_duration, self.sampling_frequency, name="window_duration")
        if self.slice_samples < 1:
            raise ValueError("window_duration must contain at least one sample.")
        self.delay_samples = tuple(integral_samples(delay, self.sampling_frequency, name="delay") for _, delay in self.slices)
        self.required_samples = self.slice_samples + max(self.delay_samples)

    def requires_warmup(self) -> bool:
        return False

    def update(self, data: torch.Tensor) -> None:
        """No learned statistics; safe for trainer warmup calls."""

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim != 3:
            raise ValueError(f"DelayedChannelSlices expects [B,T,C], got {tuple(data.shape)}.")
        if not data.is_floating_point():
            raise TypeError("DelayedChannelSlices expects floating-point input.")
        invalid = sorted({channel for channel, _ in self.slices if channel >= data.shape[-1]})
        if invalid:
            raise ValueError(f"Slice channels {invalid} are out of range for input with {data.shape[-1]} channels.")
        if data.shape[1] < self.required_samples:
            raise ValueError(f"DelayedChannelSlices requires at least {self.required_samples} context samples, got {data.shape[1]}.")
        if not torch.isfinite(data).all():
            raise ValueError("DelayedChannelSlices input must contain only finite values.")
        return torch.stack([data[:, delay:delay + self.slice_samples, channel] for (channel, _), delay in zip(self.slices, self.delay_samples)], dim=-1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)
