"""Classification head on the concatenated features of one or more encoders."""

from typing import Any

import torch

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class EncoderClassifier(BaseModel, ClassifierModel):
    """Linear or MLP head on the batch-normalized, concatenated ``features()`` of its encoders, e.g. MiniRocket.

    Args:
        encoders: ``EmbeddingModel``s; each gets the same [B, T, C] input.
        ts_len: Number of samples per input window, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        classes: Output classes, filled in from the data config.
        sequence_len: Number of predicted epochs per window, filled in from the trainer config.
        head: ``"linear"`` or ``"mlp"``.
        hidden_dim: Hidden units of the MLP head.
        dropout: Dropout before the MLP head's output layer.
    """

    def __init__(self, *, encoders: list[EmbeddingModel], ts_len: int, n_channels: int, classes: list[str], sequence_len: int = 1, head: str = "linear", hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.encoders = torch.nn.ModuleList(encoders)
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.classes = list(classes)
        self.sequence_len = int(sequence_len)
        n_features = sum(encoder.feature_dim() for encoder in self.encoders)
        output_dim = self.sequence_len * len(self.classes)
        # encoders' features come on different scales (e.g. proportions in [0, 1] next to maxima around 10)
        if head == "linear":
            self.head = torch.nn.Sequential(torch.nn.BatchNorm1d(n_features), torch.nn.Linear(n_features, output_dim))
        elif head == "mlp":
            self.head = torch.nn.Sequential(
                torch.nn.BatchNorm1d(n_features),
                torch.nn.Linear(n_features, int(hidden_dim)),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_dim), output_dim),
            )
        else:
            raise ValueError("head must be 'linear' or 'mlp'.")

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
