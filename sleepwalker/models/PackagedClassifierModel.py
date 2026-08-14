"""Trainable classification head on embeddings from a packaged model."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class PackagedClassifierModel(BaseModel, EmbeddingModel, ClassifierModel):
    """Stack native-window embeddings and train a linear or MLP head."""

    def __init__(self, *, package: str | Path | PackagedModel, classes: list[str], ts_len: int, sampling_frequency: float, input_channels: list[str], sequence_len: int = 1, embedding_stride: str | None = None, freeze_encoder: bool = True, head: str = "linear", hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        package = load_packaged_model(package) if isinstance(package, (str, Path)) else package
        if not isinstance(package, PackagedModel):
            raise TypeError(f"package must be a path or PackagedModel, got {type(package).__name__}.")
        if not isinstance(package.model, EmbeddingModel):
            raise TypeError(f"Package '{package.name}' does not expose embeddings.")

        expected_channels = list(package.dataset.get_input_channels())
        if list(input_channels) != expected_channels:
            raise ValueError(f"Package '{package.name}' expects channels {expected_channels}, got {list(input_channels)}.")
        if float(sampling_frequency) != float(package.dataset.sample_frequency):
            raise ValueError(f"Package '{package.name}' expects sample_frequency={package.dataset.sample_frequency}, got {sampling_frequency}.")

        self.encoder = package.model
        self.encoder_name = package.name
        self.freeze_encoder = bool(freeze_encoder)
        self.classes = list(classes)
        self.sequence_len = int(sequence_len)
        self.ts_len = int(ts_len)
        self.n_channels = len(input_channels)
        self.native_window_samples = int(package.dataset.get_timeseries_len())
        stride = package.dataset.stride if embedding_stride is None else embedding_stride
        self.embedding_stride_samples = int(round(pd.to_timedelta(stride).total_seconds() * float(sampling_frequency)))
        if self.native_window_samples < 1 or self.embedding_stride_samples < 1:
            raise ValueError("Native embedding window and stride must be positive.")
        if self.ts_len < self.native_window_samples:
            raise ValueError(f"Outer input has {self.ts_len} samples but package '{package.name}' requires {self.native_window_samples}.")
        remainder = (self.ts_len - self.native_window_samples) % self.embedding_stride_samples
        if remainder != 0:
            raise ValueError("Outer input must contain an exact number of native embedding windows at embedding_stride.")
        if self.sequence_len < 1 or len(self.classes) == 0:
            raise ValueError("sequence_len and classes must be non-empty.")

        self.n_embedding_windows = 1 + (self.ts_len - self.native_window_samples) // self.embedding_stride_samples
        self.encoder_feature_dim = int(self.encoder.feature_dim())
        self.stacked_feature_dim = self.n_embedding_windows * self.encoder_feature_dim
        output_dim = self.sequence_len * len(self.classes)
        if head == "linear":
            self.head = torch.nn.Linear(self.stacked_feature_dim, output_dim)
        elif head == "mlp":
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.stacked_feature_dim, int(hidden_dim)),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_dim), output_dim),
            )
        else:
            raise ValueError("head must be 'linear' or 'mlp'.")

        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def native_windows(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"Expected input shaped [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        windows = x.unfold(1, self.native_window_samples, self.embedding_stride_samples)
        return windows.permute(0, 1, 3, 2).contiguous()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        windows = self.native_windows(x)
        batch_size, n_windows = windows.shape[:2]
        encoder_input = windows.reshape(batch_size * n_windows, self.native_window_samples, self.n_channels)
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder.features(encoder_input)
        else:
            embeddings = self.encoder.features(encoder_input)
        if tuple(embeddings.shape) != (batch_size * n_windows, self.encoder_feature_dim):
            raise ValueError(f"Package '{self.encoder_name}' returned embeddings shaped {tuple(embeddings.shape)}, expected {(batch_size * n_windows, self.encoder_feature_dim)}.")
        return embeddings.reshape(batch_size, self.stacked_feature_dim)

    def feature_dim(self) -> int:
        return self.stacked_feature_dim

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.head(features).view(features.shape[0], self.sequence_len, len(self.classes))

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}
