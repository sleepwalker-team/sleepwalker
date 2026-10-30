"""Classification head on the concatenated features of one or more encoders."""

from collections.abc import Sequence
from typing import Any

import torch

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class EncoderClassifier(BaseModel, ClassifierModel):
    """Head on the concatenated ``features()`` of its encoders, e.g. MiniRocket: the given layers, then a linear output layer.

    Args:
        encoders: ``EmbeddingModel``s; each gets the same [B, T, C] input.
        ts_len: Number of samples per input window, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        classes: Output classes, filled in from the data config.
        sequence_len: Number of predicted epochs per window, filled in from the trainer config.
        layers: Layers before the output layer, e.g. a BatchNorm, hidden layers and activations; empty gives a plain
            linear head. The encoders' features come on different scales, so the layers usually start with a
            normalization. Lazy layers, such as ``torch.nn.LazyLinear``, are sized from the encoders' features.
    """

    def __init__(self, *, encoders: list[EmbeddingModel], ts_len: int, n_channels: int, classes: list[str], sequence_len: int = 1, layers: Sequence[torch.nn.Module] = ()):
        super().__init__()
        self.encoders = torch.nn.ModuleList(encoders)
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.classes = list(classes)
        self.sequence_len = int(sequence_len)
        n_features = sum(encoder.feature_dim() for encoder in self.encoders)
        if not all(isinstance(layer, torch.nn.Module) for layer in layers):
            raise ValueError(f"layers must be torch layers, got {layers!r}.")
        body = torch.nn.Sequential(*layers)
        # one batch gives lazy layers their sizes before the optimizer sees them, and the output layer its input width;
        # in eval mode it leaves BatchNorm's statistics alone
        body.eval()
        with torch.no_grad():
            width = body(torch.zeros(1, n_features)).shape[1]
        body.train()
        self.head = torch.nn.Sequential(*body, torch.nn.Linear(width, self.sequence_len * len(self.classes)))

    def warmup_preprocessor(self, data_loader, device: str):
        """Let every encoder fit itself (e.g. MiniRocket's biases) before training. The trainer calls this."""
        self.to(device)
        for encoder in self.encoders:
            if hasattr(encoder, "warmup_preprocessor"):
                encoder.warmup_preprocessor(data_loader, device)
        return self

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Logits shaped [B, sequence_len, classes] from the concatenated encoder features."""
        with torch.no_grad():  # the encoders stay frozen; only the head is trained
            features = torch.cat([encoder.features(x) for encoder in self.encoders], dim=1)
        return self.head(features).view(x.shape[0], self.sequence_len, len(self.classes))

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}
