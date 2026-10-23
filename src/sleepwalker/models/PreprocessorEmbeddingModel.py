"""An explicit preprocessing pipeline that produces one feature vector per row."""

import torch

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel


class PreprocessorEmbeddingModel(BaseModel, EmbeddingModel):
    def __init__(self, ts_len: int, n_channels: int, embedding_dim: int, preprocessors=None, input_channels=None, sampling_frequency=None):
        super().__init__(preprocessors)
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.embedding_dim = int(embedding_dim)
        self.input_channels = input_channels
        self.sampling_frequency = sampling_frequency
        if min(self.ts_len, self.n_channels, self.embedding_dim) < 1:
            raise ValueError("Input and feature dimensions must be positive.")

    def encode(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.embedding_dim or not torch.isfinite(x).all():
            raise ValueError(f"The configured pipeline must return [B, {self.embedding_dim}]; configure pooling/flattening explicitly.")
        return x

    def compute(self, x):
        return self.encode(x)

    def feature_dim(self):
        return self.embedding_dim

    def input_spec(self):
        meta = {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}
        if self.input_channels is not None:
            meta["input_channels"] = list(self.input_channels)
        if self.sampling_frequency is not None:
            meta["sampling_frequency"] = float(self.sampling_frequency)
        return (1, self.ts_len, self.n_channels), meta
