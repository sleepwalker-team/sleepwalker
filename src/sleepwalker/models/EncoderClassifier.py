"""Classification head on the concatenated features of one or more encoders."""

from collections.abc import Sequence
from typing import Any

import torch

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class EncoderClassifier(BaseModel, ClassifierModel):
    """Head on the concatenated ``features()`` of its encoders, e.g. MiniRocket: the given layers, then a linear output layer.

    Args:
        encoders: ``EmbeddingModel``s; each gets the same [B, T, C] input, the windows or, with ``context``, their epochs.
        ts_len: Number of samples per epoch, the data's ``total_input``, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        classes: Output classes, filled in from the data config.
        sequence_len: Number of predicted epochs per window, filled in from the trainer config.
        layers: Layers before the output layer, e.g. a BatchNorm, hidden layers and activations; empty gives a plain
            linear head. The encoders' features come on different scales, so the layers usually start with a
            normalization. Lazy layers, such as ``torch.nn.LazyLinear``, are sized from the encoders' features.
        context: Number of epochs on each side of the predicted one. The model lengthens the data's windows to
            2 * context + 1 epochs, labelled by the middle one, encodes each epoch on its own and gives the head
            their features side by side.
    """

    def __init__(self, *, encoders: list[EmbeddingModel], ts_len: int, n_channels: int, classes: list[str], sequence_len: int = 1, layers: Sequence[torch.nn.Module] = (), context: int = 0):
        super().__init__()
        self.encoders = torch.nn.ModuleList(encoders)
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.classes = list(classes)
        self.sequence_len = int(sequence_len)
        self.context = int(context)
        if self.context < 0:
            raise ValueError(f"context must not be negative, got {self.context}.")
        self.epochs_per_window = 2 * self.context + 1
        for encoder in self.encoders:
            if encoder.ts_len != self.ts_len:
                raise ValueError(f"The encoders take one epoch of {self.ts_len} samples; {type(encoder).__name__} has ts_len {encoder.ts_len}.")
        n_features = self.epochs_per_window * sum(encoder.feature_dim() for encoder in self.encoders)
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

    def pair_dataset(self, dataset):
        """Lengthen the dataset's windows to the epoch and its context on each side. The training CLI calls this."""
        dataset.total_input = dataset.total_input * self.epochs_per_window
        return dataset

    def warmup_preprocessor(self, data_loader, device: str):
        """Let every encoder fit itself (e.g. MiniRocket's biases) on the windows' epochs before training. The trainer calls this."""
        self.to(device)
        for encoder in self.encoders:
            if hasattr(encoder, "warmup_preprocessor"):
                epochs = ({"data": self.split_epochs(batch["data"])} for batch in data_loader if batch is not None)
                encoder.warmup_preprocessor(epochs, device)
        return self

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Logits shaped [B, sequence_len, classes] from the concatenated encoder features of the windows' epochs."""
        with torch.no_grad():  # the encoders stay frozen; only the head is trained
            epochs = self.split_epochs(x)
            features = torch.cat([encoder.features(epochs) for encoder in self.encoders], dim=1)
        # each window's epochs side by side
        return self.head(features.reshape(x.shape[0], -1)).view(x.shape[0], self.sequence_len, len(self.classes))

    def split_epochs(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] windows -> [B * (2 * context + 1), T / (2 * context + 1), C]: each window's epochs, in order."""
        return x.reshape(x.shape[0] * self.epochs_per_window, self.ts_len, x.shape[2])

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        window = self.epochs_per_window * self.ts_len
        return (1, window, self.n_channels), {"layout": "BTC", "ts_len": window, "n_channels": self.n_channels}
