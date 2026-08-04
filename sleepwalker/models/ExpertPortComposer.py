"""Composition heads for aligned outputs from independently trained experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class ExpertPortSpec:
    name: str
    feature_dim: int
    probability_dim: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ExpertPortSpec.name must not be empty.")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive for expert '{self.name}'.")
        if self.probability_dim <= 0:
            raise ValueError(f"probability_dim must be positive for expert '{self.name}'.")


@dataclass(frozen=True)
class PortInterfaceEdge:
    source: str
    target: str
    bottleneck_dim: int
    gated: bool = True

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("Port interface self-edges are not supported.")
        if self.bottleneck_dim <= 0:
            raise ValueError("Port interface bottleneck_dim must be positive.")


class _MessageEncoder(nn.Module):
    def __init__(self, source_dim: int, message_dim: int, gated: bool):
        super().__init__()
        self.norm = nn.LayerNorm(source_dim)
        self.projection = nn.Linear(source_dim, message_dim)
        self.gate_logit = nn.Parameter(torch.zeros(())) if gated else None

    def forward(self, source: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        message = torch.tanh(self.projection(self.norm(source)))
        if self.gate_logit is not None:
            message = torch.sigmoid(self.gate_logit) * message
        return message * available


class ExpertPortComposer(nn.Module):
    """Predict one receiver task from time-aligned expert ports.

    The composer never owns or calls the experts. They are executed independently
    under their own input/preprocessing contracts and their aligned features,
    probabilities, and availability masks are supplied to :meth:`forward`.
    """

    METHODS = {"independent", "stacking", "concat", "structured"}

    def __init__(
        self,
        *,
        receiver: str,
        port_specs: list[ExpertPortSpec],
        num_classes: int,
        method: str,
        edges: list[PortInterfaceEdge] | None = None,
        hidden_dim: int = 64,
        receiver_adapter_dim: int = 0,
    ):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"Unknown composition method '{method}'. Expected one of {sorted(self.METHODS)}.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if len({spec.name for spec in port_specs}) != len(port_specs):
            raise ValueError("Expert port names must be unique.")

        self.receiver = receiver
        self.method = method
        self.specs = {spec.name: spec for spec in port_specs}
        if receiver not in self.specs:
            raise ValueError(f"Receiver '{receiver}' is not present in port_specs.")

        self.edges = list(edges or [])
        for edge in self.edges:
            if edge.source not in self.specs or edge.target not in self.specs:
                raise ValueError(f"Unknown expert in edge {edge.source}->{edge.target}.")
        self.incoming_edges = [edge for edge in self.edges if edge.target == receiver]
        if method == "structured" and not self.incoming_edges:
            raise ValueError(f"Structured composer for '{receiver}' requires at least one incoming edge.")

        receiver_dim = self.specs[receiver].feature_dim
        if receiver_adapter_dim < 0:
            raise ValueError("receiver_adapter_dim must be non-negative.")
        self.receiver_adapter = None
        if receiver_adapter_dim > 0:
            self.receiver_adapter = nn.Sequential(
                nn.LayerNorm(receiver_dim),
                nn.Linear(receiver_dim, receiver_adapter_dim),
                nn.GELU(),
                nn.Linear(receiver_adapter_dim, receiver_dim),
            )

        self.message_encoders = nn.ModuleDict()
        for edge in self.incoming_edges:
            key = self.edge_key(edge.source, edge.target)
            if key in self.message_encoders:
                raise ValueError(f"Duplicate interface edge {edge.source}->{edge.target}.")
            self.message_encoders[key] = _MessageEncoder(
                self.specs[edge.source].feature_dim,
                edge.bottleneck_dim,
                edge.gated,
            )

        if method == "independent":
            head_input_dim = receiver_dim
        elif method == "stacking":
            head_input_dim = sum(spec.probability_dim + 1 for spec in port_specs)
        elif method == "concat":
            head_input_dim = sum(spec.feature_dim + 1 for spec in port_specs)
        else:
            head_input_dim = receiver_dim + sum(edge.bottleneck_dim + 1 for edge in self.incoming_edges)

        if hidden_dim <= 0:
            self.head = nn.Linear(head_input_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(head_input_dim),
                nn.Linear(head_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )

    @staticmethod
    def edge_key(source: str, target: str) -> str:
        return f"{source}__to__{target}"

    def communication_scalars_per_sample(self) -> int:
        """Count structured message scalars, excluding availability bits."""
        if self.method != "structured":
            return 0
        return sum(edge.bottleneck_dim for edge in self.incoming_edges)

    def gate_values(self) -> dict[str, float]:
        values = {}
        for key, encoder in self.message_encoders.items():
            value = 1.0 if encoder.gate_logit is None else float(torch.sigmoid(encoder.gate_logit).detach().cpu())
            values[key] = value
        return values

    def communication_l1(self) -> torch.Tensor:
        penalties = [
            torch.sigmoid(encoder.gate_logit)
            for encoder in self.message_encoders.values()
            if encoder.gate_logit is not None
        ]
        if penalties:
            return torch.stack(penalties).sum()
        return next(self.parameters()).new_zeros(())

    def _validate_ports(
        self,
        features: Mapping[str, torch.Tensor],
        probabilities: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
    ) -> int:
        expected = set(self.specs)
        for label, ports in [
            ("features", features),
            ("probabilities", probabilities),
            ("available", available),
        ]:
            if set(ports) != expected:
                raise ValueError(f"{label} must contain exactly {sorted(expected)}, got {sorted(ports)}.")

        batch_size = None
        for name, spec in self.specs.items():
            feature = features[name]
            probability = probabilities[name]
            mask = available[name]
            if feature.ndim != 2 or feature.shape[1] != spec.feature_dim:
                raise ValueError(
                    f"Feature port '{name}' must have shape [B, {spec.feature_dim}], got {tuple(feature.shape)}."
                )
            if probability.ndim != 2 or probability.shape[1] != spec.probability_dim:
                raise ValueError(
                    f"Probability port '{name}' must have shape [B, {spec.probability_dim}], "
                    f"got {tuple(probability.shape)}."
                )
            if mask.ndim != 2 or mask.shape[1] != 1:
                raise ValueError(f"Availability port '{name}' must have shape [B, 1], got {tuple(mask.shape)}.")
            if batch_size is None:
                batch_size = feature.shape[0]
            if feature.shape[0] != batch_size or probability.shape[0] != batch_size or mask.shape[0] != batch_size:
                raise ValueError("All expert ports must have the same batch size.")
        return int(batch_size or 0)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        probabilities: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        self._validate_ports(features, probabilities, available)
        receiver_state = features[self.receiver]
        if self.receiver_adapter is not None:
            receiver_state = receiver_state + self.receiver_adapter(receiver_state)

        if self.method == "independent":
            head_input = receiver_state
        elif self.method == "stacking":
            head_input = torch.cat(
                [
                    tensor
                    for name in self.specs
                    for tensor in (probabilities[name] * available[name], available[name])
                ],
                dim=1,
            )
        elif self.method == "concat":
            head_input = torch.cat(
                [
                    tensor
                    for name in self.specs
                    for tensor in (features[name] * available[name], available[name])
                ],
                dim=1,
            )
        else:
            parts = [receiver_state]
            for edge in self.incoming_edges:
                mask = available[edge.source]
                key = self.edge_key(edge.source, edge.target)
                parts.extend([self.message_encoders[key](features[edge.source], mask), mask])
            head_input = torch.cat(parts, dim=1)
        return self.head(head_input)
