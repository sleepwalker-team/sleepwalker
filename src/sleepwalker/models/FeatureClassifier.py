"""Classification head on feature vectors from the model's preprocessors."""

from typing import Any

import torch

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel


class FeatureClassifier(BaseModel, ClassifierModel):
    """Linear or MLP head on the [B, n_features] output of the last preprocessor, e.g. MiniRocket.

    Args:
        ts_len: Number of samples per input window, filled in from the data config.
        n_channels: Number of input channels, filled in from the data config.
        classes: Output classes, filled in from the data config.
        preprocessors: Preprocessors run on the [B, T, C] input; the last one must expose ``n_features``.
        sequence_len: Number of predicted epochs per window, filled in from the trainer config.
        head: ``"linear"`` or ``"mlp"``.
        hidden_dim: Hidden units of the MLP head.
        dropout: Dropout before the MLP head's output layer.
    """

    def __init__(self, *, ts_len: int, n_channels: int, classes: list[str], preprocessors: list[torch.nn.Module], sequence_len: int = 1, head: str = "linear", hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__(preprocessors=preprocessors)
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.classes = list(classes)
        self.sequence_len = int(sequence_len)
        self.n_features = self.preprocessors[-1].n_features
        output_dim = self.sequence_len * len(self.classes)
        if head == "linear":
            self.head = torch.nn.Linear(self.n_features, output_dim)
        elif head == "mlp":
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.n_features, int(hidden_dim)),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_dim), output_dim),
            )
        else:
            raise ValueError("head must be 'linear' or 'mlp'.")

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Logits shaped [B, sequence_len, classes]."""
        return self.head(x).view(x.shape[0], self.sequence_len, len(self.classes))

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}
