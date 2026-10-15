"""Fixed, explicitly selected channel scaling and optional clipping."""

from collections.abc import Mapping, Sequence
import math

import torch
from torch import nn


class ScaleChannels(nn.Module):
    """Multiply selected input channels, then optionally clip selected channels.

    scales maps zero-based input channel indices to finite multipliers; clip
    maps input channel indices to finite [minimum, maximum] bounds, applied
    after scaling. Unspecified channels retain their values. Both maps use
    indices in this step's input, so placement before/after channel slicing is
    explicit. Input shape, dtype and device are preserved without mutation.
    No physiological meaning or unit conversion is inferred.
    """

    def __init__(self, scales: Mapping[int, float], clip: Mapping[int, Sequence[float]] | None = None):
        super().__init__()
        if not isinstance(scales, Mapping) or (clip is not None and not isinstance(clip, Mapping)):
            raise TypeError("scales and clip must be channel-index mappings.")
        self.scales = {}
        self.clip = {}
        for channel in [*scales, *(clip or {})]:
            if not isinstance(channel, int) or isinstance(channel, bool) or channel < 0:
                raise ValueError("Scale/clip channels must be non-negative integer indices.")
        for channel, scale in scales.items():
            scale = float(scale)
            if not math.isfinite(scale):
                raise ValueError("Channel scale multipliers must be finite.")
            self.scales[channel] = scale
        for channel, bounds in (clip or {}).items():
            if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence) or len(bounds) != 2:
                raise ValueError("Channel clip bounds must be [minimum, maximum].")
            lower, upper = (float(bound) for bound in bounds)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("Channel clip bounds must be finite and increasing.")
            self.clip[channel] = (lower, upper)
        if not self.scales and not self.clip:
            raise ValueError("Configure at least one channel scale or clip bound.")

    def requires_warmup(self) -> bool:
        return False

    def update(self, data: torch.Tensor) -> None:
        """No learned statistics; all multipliers and bounds are configured."""

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim == 0:
            raise ValueError("ScaleChannels requires a channel dimension on the last axis.")
        if not data.is_floating_point():
            raise TypeError("ScaleChannels expects floating-point input.")
        invalid = sorted(channel for channel in self.scales.keys() | self.clip.keys() if channel >= data.shape[-1])
        if invalid:
            raise ValueError(f"Scale/clip channels {invalid} are out of range for input with {data.shape[-1]} channels.")
        if not torch.isfinite(data).all():
            raise ValueError("ScaleChannels input must contain only finite values.")
        result = data.clone()
        for channel in self.scales.keys() | self.clip.keys():
            values = data[..., channel] * self.scales.get(channel, 1.0)
            if channel in self.clip:
                lower, upper = self.clip[channel]
                values = values.clamp(lower, upper)
            result[..., channel] = values
        return result

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)
