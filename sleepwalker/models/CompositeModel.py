"""Compose embedding models over a shared raw input."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel
from sleepwalker.utils import logger


@dataclass
class CompositeModelEntry:
    """An embedding model and the shared-input channels routed to it."""

    model: BaseModel
    input_channels: list[str]
    trainable: bool = True

    def __post_init__(self):
        if not isinstance(self.model, BaseModel) or not isinstance(self.model, EmbeddingModel):
            raise TypeError("CompositeModelEntry.model must be a BaseModel implementing EmbeddingModel.")
        if len(self.input_channels) == 0:
            raise ValueError("CompositeModelEntry.input_channels must not be empty.")
        if not self.trainable:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)


@dataclass(frozen=True)
class CompositeModelEdge:
    """A directed low-dimensional message between two embedding models."""

    source: int
    target: int
    bottleneck_dim: int = 8
    gated: bool = True


class InterfaceModule(torch.nn.Module):
    def __init__(self, sender_dim: int, bottleneck_dim: int, receiver_dim: int, gated: bool):
        super().__init__()
        self.down = torch.nn.Linear(sender_dim, bottleneck_dim)
        self.up = torch.nn.Linear(bottleneck_dim, receiver_dim)
        self.gate = torch.nn.Parameter(torch.tensor(0.0)) if gated else None

    def forward(self, sender_embedding: torch.Tensor) -> torch.Tensor:
        message = self.up(self.down(sender_embedding))
        if self.gate is not None:
            message = torch.sigmoid(self.gate) * message
        return message


class CompositeModel(BaseModel, EmbeddingModel, ClassifierModel):
    """Route channels to embedding models, optionally exchange messages, and classify.

    Exactly one output contract must be supplied: ``classes`` for a single
    tensor output or ``task_config`` for a dictionary of task outputs. A head
    consumes all concatenated embeddings unless its receiver is specified.
    """

    def __init__(
        self,
        *,
        models: list[CompositeModelEntry],
        input_channels: list[str] | None = None,
        classes: list[str] | None = None,
        task_config: dict[str, dict] | None = None,
        sequence_len: int = 1,
        receiver: int | None = None,
        task_receivers: dict[str, int | None] | None = None,
        edges: list[CompositeModelEdge] | None = None,
        preprocessors: list[torch.nn.Module] | None = None,
    ):
        super().__init__(preprocessors=preprocessors)
        if (classes is None) == (task_config is None):
            raise ValueError("CompositeModel requires exactly one of classes or task_config.")
        if len(models) == 0:
            raise ValueError("CompositeModel requires at least one embedding model.")

        self.input_channels = self.resolve_input_channels(models, input_channels)
        self.embedding_models = torch.nn.ModuleList([entry.model for entry in models])
        self.channel_indices = []
        self.feature_dims = []
        for index, entry in enumerate(models):
            missing = [channel for channel in entry.input_channels if channel not in self.input_channels]
            if missing:
                raise ValueError(f"Unknown input channels for model #{index}: {missing}.")
            self.channel_indices.append([self.input_channels.index(channel) for channel in entry.input_channels])
            self.feature_dims.append(int(entry.model.feature_dim()))

        self.edges = list(edges or [])
        self.interface_modules = torch.nn.ModuleList()
        for edge in self.edges:
            self.validate_edge(edge)
            self.interface_modules.append(InterfaceModule(self.feature_dims[edge.source], edge.bottleneck_dim, self.feature_dims[edge.target], edge.gated))

        self.classes = None if classes is None else list(classes)
        self.task_config = None if task_config is None else {task: dict(config) for task, config in task_config.items()}
        self.sequence_len = int(sequence_len)
        self.receiver = receiver
        self.task_receivers = None if task_receivers is None else dict(task_receivers)

        if self.classes is not None:
            self.build_single_head()
        else:
            self.build_task_heads()

    @staticmethod
    def resolve_input_channels(models: list[CompositeModelEntry], input_channels: list[str] | None) -> list[str]:
        if input_channels is not None:
            return list(input_channels)
        channels = []
        for entry in models:
            for channel in entry.input_channels:
                if channel not in channels:
                    channels.append(channel)
        return channels

    def validate_receiver(self, receiver: int | None, label: str) -> None:
        if receiver is not None and (receiver < 0 or receiver >= len(self.embedding_models)):
            raise ValueError(f"{label} index is out of range: {receiver}.")

    def validate_edge(self, edge: CompositeModelEdge) -> None:
        self.validate_receiver(edge.source, "Edge source")
        self.validate_receiver(edge.target, "Edge target")
        if edge.source == edge.target:
            raise ValueError("CompositeModel does not support self-edges.")
        if edge.bottleneck_dim <= 0:
            raise ValueError(f"Edge bottleneck_dim must be positive, got {edge.bottleneck_dim}.")

    def build_single_head(self) -> None:
        if len(self.classes) == 0:
            raise ValueError("CompositeModel classes must not be empty.")
        if self.sequence_len < 1:
            raise ValueError("sequence_len must be at least 1.")
        self.validate_receiver(self.receiver, "Single-head receiver")
        input_dim = sum(self.feature_dims) if self.receiver is None else self.feature_dims[self.receiver]
        self.head = torch.nn.Linear(input_dim, self.sequence_len * len(self.classes))

    def build_task_heads(self) -> None:
        if len(self.task_config) == 0:
            raise ValueError("CompositeModel task_config must not be empty.")
        if self.task_receivers is None:
            self.task_receivers = {task: None for task in self.task_config}
        if set(self.task_receivers) != set(self.task_config):
            raise ValueError(f"task_receivers must define exactly {sorted(self.task_config)}, got {sorted(self.task_receivers)}.")

        self.heads = torch.nn.ModuleDict()
        for task, config in self.task_config.items():
            if "labels" not in config or "sequence_len" not in config:
                raise ValueError(f"Task '{task}' must provide labels and sequence_len.")
            if len(config["labels"]) == 0 or int(config["sequence_len"]) < 1:
                raise ValueError(f"Task '{task}' labels must be non-empty and sequence_len must be positive.")
            receiver = self.task_receivers[task]
            self.validate_receiver(receiver, f"Task '{task}' receiver")
            input_dim = sum(self.feature_dims) if receiver is None else self.feature_dims[receiver]
            self.heads[task] = torch.nn.Linear(input_dim, int(config["sequence_len"]) * len(config["labels"]))

    def expert_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        embeddings = []
        for model, indices in zip(self.embedding_models, self.channel_indices):
            embedding = model.features(x[:, :, indices])
            if embedding.ndim != 2:
                raise ValueError(f"CompositeModel expects 2D embeddings [B, E], got {tuple(embedding.shape)}.")
            embeddings.append(embedding)

        states = [embedding.clone() for embedding in embeddings]
        for edge, interface in zip(self.edges, self.interface_modules):
            states[edge.target] = states[edge.target] + interface(embeddings[edge.source])
        return states

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.expert_states(x), dim=1)

    def feature_dim(self) -> int:
        return sum(self.feature_dims)

    def compute(self, x: torch.Tensor):
        states = self.expert_states(x)
        if self.classes is not None:
            features = torch.cat(states, dim=1) if self.receiver is None else states[self.receiver]
            return self.head(features).view(features.shape[0], self.sequence_len, len(self.classes))

        outputs = {}
        for task, config in self.task_config.items():
            receiver = self.task_receivers[task]
            features = torch.cat(states, dim=1) if receiver is None else states[receiver]
            logits = self.heads[task](features)
            outputs[task] = logits.view(features.shape[0], int(config["sequence_len"]), len(config["labels"]))
        return outputs

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        first_shape, _ = self.embedding_models[0].input_spec()
        return (1, first_shape[1], len(self.input_channels)), {"layout": "BTC", "ts_len": first_shape[1], "n_channels": len(self.input_channels)}

    def warmup_preprocessor(self, data_loader: DataLoader, device: str = "cuda"):
        self.to(device)
        batch_size = data_loader.batch_size
        if batch_size is None:
            raise ValueError("batch_size must not be None during preprocessor warmup.")

        for index, step in enumerate(self.preprocessors):
            logger.progress_start(len(data_loader) * batch_size, desc=f"composite {index}/{len(self.preprocessors) - 1}", leave=True)
            if step.requires_warmup():
                for batch in data_loader:
                    step.update(self.apply_preprocessors(batch["data"].to(device), index))
                    logger.progress_advance(batch_size)
            else:
                logger.progress_advance(len(data_loader) * batch_size)
            logger.progress_close()

        for model, indices in zip(self.embedding_models, self.channel_indices):
            for index, step in enumerate(model.preprocessors):
                logger.progress_start(len(data_loader) * batch_size, desc=f"nested {index}/{len(model.preprocessors) - 1}", leave=True)
                if step.requires_warmup():
                    for batch in data_loader:
                        x = self.apply_preprocessors(batch["data"].to(device))[:, :, indices]
                        step.update(model.apply_preprocessors(x, index))
                        logger.progress_advance(batch_size)
                else:
                    logger.progress_advance(len(data_loader) * batch_size)
                logger.progress_close()
        return self
