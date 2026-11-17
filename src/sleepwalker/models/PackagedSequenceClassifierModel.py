"""Position-wise classifier for temporal tokens from a packaged encoder."""

from pathlib import Path
from typing import Any

import torch

from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class PackagedSequenceClassifierModel(BaseModel, EmbeddingModel, ClassifierModel):
    """Pool contiguous encoder tokens and classify every resulting time step."""

    def __init__(self, *, package: str | Path | PackagedModel, classes: list[str], ts_len: int, sequence_len: int, token_count: int, token_dim: int, modality_count: int = 1, freeze_encoder: bool = True, head: str = "linear", hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        package = load_packaged_model(package) if isinstance(package, (str, Path)) else package
        if not isinstance(package, PackagedModel):
            raise TypeError(f"package must be a path or PackagedModel, got {type(package).__name__}.")
        if not isinstance(package.model, EmbeddingModel):
            raise TypeError(f"Package '{package.name}' does not expose embeddings.")

        self.encoder = package.model
        self.encoder_name = package.name
        self.classes = list(classes)
        self.sequence_len = int(sequence_len)
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.modality_count = int(modality_count)
        self.freeze_encoder = bool(freeze_encoder)
        self.ts_len = int(ts_len)
        self.input_channels = list(package.dataset.get_input_channels())
        self.sampling_frequency = float(package.dataset.sample_frequency)
        self.n_channels = len(self.input_channels)

        if self.ts_len != package.dataset.get_timeseries_len():
            raise ValueError(f"Sequence classifier input must equal package '{package.name}' native window of {package.dataset.get_timeseries_len()} samples, got {self.ts_len}.")
        if min(self.sequence_len, self.token_count, self.token_dim, self.modality_count) < 1 or len(self.classes) < 2:
            raise ValueError("sequence_len, token_count, token_dim, modality_count, and classes must be non-empty.")
        if self.token_count % self.sequence_len != 0:
            raise ValueError("token_count must be divisible by sequence_len.")
        expected_feature_dim = self.modality_count * self.token_count * self.token_dim
        if self.encoder.feature_dim() != expected_feature_dim:
            raise ValueError(f"Package '{package.name}' exposes {self.encoder.feature_dim()} features, expected modality_count * token_count * token_dim = {expected_feature_dim}.")

        self.tokens_per_step = self.token_count // self.sequence_len
        self.step_feature_dim = self.modality_count * self.token_dim
        self.stacked_feature_dim = self.sequence_len * self.step_feature_dim
        if head == "linear":
            self.head = torch.nn.Linear(self.step_feature_dim, len(self.classes))
        elif head == "mlp":
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.step_feature_dim, int(hidden_dim)),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_dim), len(self.classes)),
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

    def encode_steps(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"Expected input shaped [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder.features(x)
        else:
            embeddings = self.encoder.features(x)
        expected = (x.shape[0], self.modality_count * self.token_count * self.token_dim)
        if tuple(embeddings.shape) != expected:
            raise ValueError(f"Package '{self.encoder_name}' returned embeddings shaped {tuple(embeddings.shape)}, expected {expected}.")
        embeddings = embeddings.view(x.shape[0], self.modality_count, self.token_count, self.token_dim)
        embeddings = embeddings.view(x.shape[0], self.modality_count, self.sequence_len, self.tokens_per_step, self.token_dim).mean(dim=3)
        return embeddings.permute(0, 2, 1, 3).reshape(x.shape[0], self.sequence_len, self.step_feature_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_steps(x).reshape(x.shape[0], self.stacked_feature_dim)

    def feature_dim(self) -> int:
        return self.stacked_feature_dim

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_steps(x))

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {
            "layout": "BTC",
            "ts_len": self.ts_len,
            "n_channels": self.n_channels,
            "input_channels": list(self.input_channels),
            "sampling_frequency": self.sampling_frequency,
        }
