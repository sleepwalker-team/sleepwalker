"""Embedding capability for a converted, self-contained PyTorch encoder."""

import io
from typing import Any

import torch

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel


class PackagedEmbeddingModel(BaseModel, EmbeddingModel):
    def __init__(self, *, encoder: torch.nn.Module, embedding_dim: int, ts_len: int, n_channels: int, channel_first: bool = True, preprocessors: list[torch.nn.Module] | None = None):
        super().__init__(preprocessors=preprocessors)
        self.encoder = encoder
        self.embedding_dim = int(embedding_dim)
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.channel_first = bool(channel_first)

    def __getstate__(self):
        state = super().__getstate__()
        encoder = state["_modules"].get("encoder")
        if isinstance(encoder, torch.jit.ScriptModule):
            stream = io.BytesIO()
            torch.jit.save(encoder, stream)
            state["_modules"] = dict(state["_modules"])
            state["_modules"].pop("encoder")
            state["serialized_encoder"] = stream.getvalue()
        return state

    def __setstate__(self, state):
        serialized_encoder = state.pop("serialized_encoder", None)
        super().__setstate__(state)
        if serialized_encoder is not None:
            self.encoder = torch.jit.load(io.BytesIO(serialized_encoder), map_location="cpu")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"Expected input shaped [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        encoder_input = x.transpose(1, 2).contiguous() if self.channel_first else x
        embeddings = self.encoder(encoder_input)
        if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != (x.shape[0], self.embedding_dim):
            shape = tuple(embeddings.shape) if isinstance(embeddings, torch.Tensor) else type(embeddings).__name__
            raise ValueError(f"Converted encoder returned {shape}, expected {(x.shape[0], self.embedding_dim)}.")
        return embeddings

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def feature_dim(self) -> int:
        return self.embedding_dim

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}
