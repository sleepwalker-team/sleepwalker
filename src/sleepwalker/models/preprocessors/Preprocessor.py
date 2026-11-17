"""Abstract base class for tensor preprocessors.

Preprocessors are small modules attached to models and warmed externally by
trainer code when needed. Current tests and model code assume tensors shaped
`[B, T, C]`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence
import torch
from torch import nn 

class Preprocessor(nn.Module, ABC):
    """Common interface for repository preprocessors.

    The base class optionally restricts preprocessing to a subset of channel
    indices on the last tensor dimension. Subclasses only implement the actual
    warmup and transformation logic for the selected tensor slice.

    Args:
        channels: Optional list of channel indices to preprocess. ``None``
            means all channels. Invalid configurations such as duplicates,
            negative indices, or an empty list are rejected eagerly.
    """

    def __init__(self, channels: Optional[Sequence[int]] = None):
        super().__init__()
        if channels is None:
            self.register_buffer("channels", torch.empty(0, dtype=torch.long), persistent=False)
            self._has_channel_selection = False
            return

        channel_list = list(channels)
        if len(channel_list) == 0:
            raise ValueError("channels must not be empty when channel selection is enabled.")
        if any(not isinstance(idx, int) for idx in channel_list):
            raise ValueError("channels must contain integers only.")
        if any(idx < 0 for idx in channel_list):
            raise ValueError("channels must not contain negative indices.")
        if len(set(channel_list)) != len(channel_list):
            raise ValueError("channels must not contain duplicate indices.")

        self.register_buffer("channels", torch.tensor(channel_list, dtype=torch.long), persistent=False)
        self._has_channel_selection = True

    @torch.no_grad()
    def update(self, data: torch.Tensor):
        """Update internal running statistics from a warmup batch."""
        self._validate_channels(data)
        self._update(self._select_channels(data))
    
    @abstractmethod
    def requires_warmup(self) -> bool:
        """Return whether this preprocessor needs warmup before use."""
        ...

    @torch.no_grad()
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform one batch tensor.

        When channel selection is active, only the configured channels are
        passed to the subclass implementation. Shape-changing preprocessors can
        only be applied selectively if they preserve every dimension except the
        last channel dimension.
        """
        self._validate_channels(data)
        selected = self._select_channels(data)
        transformed = self._transform(selected)

        if not self._has_channel_selection:
            return transformed

        if transformed.shape[:-1] != data.shape[:-1] or transformed.shape[-1] != len(self.channels):
            raise ValueError(
                "Selective preprocessing requires the transformed tensor to preserve all "
                "non-channel dimensions and return exactly one output channel per selected input channel."
            )

        out = data.clone()
        out.index_copy_(-1, self.channels, transformed)
        return out

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Alias :meth:`transform` for ``nn.Module`` compatibility."""
        return self.transform(data)

    def _validate_channels(self, data: torch.Tensor) -> None:
        """Validate channel selection against the current input tensor."""
        if not self._has_channel_selection:
            return
        n_channels = data.shape[-1]
        if torch.any(self.channels >= n_channels):
            invalid = self.channels[self.channels >= n_channels].tolist()
            raise ValueError(
                f"Selected channel indices {invalid} are out of range for input with {n_channels} channels."
            )

    def _select_channels(self, data: torch.Tensor) -> torch.Tensor:
        """Return either the full tensor or the selected channel subset."""
        if not self._has_channel_selection:
            return data
        return data.index_select(-1, self.channels)

    @abstractmethod
    def _update(self, data: torch.Tensor):
        """Update internal state from the selected tensor slice."""
        ...

    @abstractmethod
    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform the selected tensor slice."""
        ...
    
    
