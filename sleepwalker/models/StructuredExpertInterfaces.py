"""Structured low-bandwidth interfaces between heterogeneous expert embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.MultiModel import MetaModelEntry


@dataclass(frozen=True)
class ExpertInterfaceEdge:
    """Directed communication edge between two expert embeddings."""

    source: int
    target: int
    bottleneck_dim: int = 8
    gated: bool = True


class _InterfaceModule(torch.nn.Module):
    def __init__(self, sender_dim: int, bottleneck_dim: int, receiver_dim: int, gated: bool):
        super().__init__()
        self.down = torch.nn.Linear(sender_dim, bottleneck_dim)
        self.up = torch.nn.Linear(bottleneck_dim, receiver_dim)
        self.gated = bool(gated)
        self.gate = torch.nn.Parameter(torch.tensor(0.0)) if self.gated else None

    def forward(self, sender_embedding: torch.Tensor) -> torch.Tensor:
        message = self.up(self.down(sender_embedding))
        if self.gated:
            message = torch.sigmoid(self.gate) * message
        return message


class StructuredExpertInterfaceMetaModel(BaseModel):
    """Fuse expert embeddings through sparse directional bottleneck messages.

    Unlike :class:`MetaModel`, this module does not concatenate raw sender
    embeddings directly into every receiver. Each configured edge projects the
    sender embedding into a low-dimensional message and decodes it into the
    receiver's feature space before task heads see the final representation.
    """

    def __init__(
        self,
        *,
        task_config: dict[str, dict],
        input_channels: list[str],
        models: list[MetaModelEntry],
        edges: list[ExpertInterfaceEdge],
        preprocessors=None,
    ):
        super().__init__(preprocessors=preprocessors)
        self.task_config = dict(task_config)
        self.input_channels = list(input_channels)
        self.edges = list(edges)

        if len(models) == 0:
            raise ValueError("StructuredExpertInterfaceMetaModel requires at least one expert.")

        self.model_entries = []
        self.feature_dims = []
        for idx, entry in enumerate(models):
            missing = [c for c in entry.input_channels if c not in self.input_channels]
            if len(missing) > 0:
                raise ValueError(f"Unknown input channels for expert #{idx}: {missing}")
            if not hasattr(entry.model, "features"):
                raise ValueError(f"Expert #{idx} must implement features(x).")
            if not hasattr(entry.model, "feature_dim"):
                raise ValueError(f"Expert #{idx} must implement feature_dim().")

            self.add_module(f"expert_{idx}", entry.model)
            self.model_entries.append({
                "model": entry.model,
                "indices": [self.input_channels.index(c) for c in entry.input_channels],
            })
            self.feature_dims.append(int(entry.model.feature_dim()))

        self.edge_modules = torch.nn.ModuleList()
        for edge in self.edges:
            self._validate_edge(edge, len(models))
            sender_dim = self.feature_dims[edge.source]
            receiver_dim = self.feature_dims[edge.target]
            self.edge_modules.append(_InterfaceModule(sender_dim, edge.bottleneck_dim, receiver_dim, edge.gated))

        self._feature_dim = sum(self.feature_dims)
        self.heads = torch.nn.ModuleDict()
        for task, cfg in self.task_config.items():
            if "labels" not in cfg or "n_steps" not in cfg:
                raise ValueError(f"Task '{task}' must provide normalized config with 'labels' and 'n_steps'.")
            self.heads[task] = torch.nn.Linear(self._feature_dim, int(cfg["n_steps"]) * len(cfg["labels"]))

    @staticmethod
    def _validate_edge(edge: ExpertInterfaceEdge, n_models: int) -> None:
        if edge.source < 0 or edge.source >= n_models:
            raise ValueError(f"Edge source index out of range: {edge.source}")
        if edge.target < 0 or edge.target >= n_models:
            raise ValueError(f"Edge target index out of range: {edge.target}")
        if edge.source == edge.target:
            raise ValueError("Self-edges are not supported.")
        if edge.bottleneck_dim <= 0:
            raise ValueError(f"Edge bottleneck_dim must be positive, got {edge.bottleneck_dim}.")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for entry in self.model_entries:
            emb = entry["model"].features(x[:, :, entry["indices"]])
            if emb.ndim != 2:
                raise ValueError("StructuredExpertInterfaceMetaModel expects expert features shaped [B, E].")
            embeddings.append(emb)

        receiver_states = [emb.clone() for emb in embeddings]
        for edge, module in zip(self.edges, self.edge_modules):
            receiver_states[edge.target] = receiver_states[edge.target] + module(embeddings[edge.source])

        return torch.cat(receiver_states, dim=1)

    def feature_dim(self) -> int:
        return self._feature_dim

    def _classifier(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for task, cfg in self.task_config.items():
            logits = self.heads[task](x)
            out[task] = logits.view(x.shape[0], int(cfg["n_steps"]), len(cfg["labels"]))
        return out

    def input_spec(self):
        first_shape, _ = self.model_entries[0]["model"].input_spec()
        return (
            (1, first_shape[1], len(self.input_channels)),
            {"layout": "BTC", "ts_len": first_shape[1], "n_channels": len(self.input_channels)},
        )
