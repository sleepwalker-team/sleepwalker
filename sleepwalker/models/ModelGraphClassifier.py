"""Compose task models over a directed acyclic graph."""

import bisect
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sleepwalker.datasets.Basedataset import BaseDataset, ChannelConfig, stack_repeated_views
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment.package import PackagedModel, load_packaged_model
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


@dataclass
class GraphNode:
    """A model and the tasks it predicts inside a graph."""

    model: BaseModel
    outputs: dict[str, dict[str, Any]]
    trainable: bool = False

    def __post_init__(self):
        if not isinstance(self.model, BaseModel):
            raise TypeError(f"GraphNode.model must be a BaseModel, got {type(self.model).__name__}.")
        self.outputs = normalize_outputs(self.outputs)
        if not self.trainable:
            self.model.requires_grad_(False)
            self.model.eval()


def normalize_outputs(outputs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized = {}
    for task, config in outputs.items():
        if set(config) != {"classes", "sequence_len"}:
            raise ValueError(f"Output '{task}' must contain exactly classes and sequence_len, got {sorted(config)}.")
        classes = list(config["classes"])
        sequence_len = int(config["sequence_len"])
        if len(classes) < 2:
            raise ValueError(f"Output '{task}' must contain at least two classes.")
        if len(classes) != len(set(classes)):
            raise ValueError(f"Output '{task}' contains duplicate classes: {classes}.")
        if sequence_len < 1:
            raise ValueError(f"Output '{task}' sequence_len must be positive.")
        normalized[str(task)] = {"classes": classes, "sequence_len": sequence_len}
    return normalized


def outputs_from_package(package: PackagedModel) -> dict[str, dict[str, Any]]:
    contract = package.classification_contract
    if contract is None:
        raise ValueError(f"Package '{package.name}' has no classification contract. Pass outputs explicitly.")
    if contract.get("type") == "single-head-multiclass":
        if package.task is None:
            raise ValueError(f"Package '{package.name}' does not declare its task.")
        return {
            package.task: {
                "classes": list(contract["classes"]),
                "sequence_len": int(contract["sequence_len"]),
            }
        }
    if contract.get("type") == "multitask":
        return {
            task: {
                "classes": list(config["classes"]),
                "sequence_len": int(config["n_steps"]),
            }
            for task, config in contract["tasks"].items()
        }
    raise ValueError(f"Package '{package.name}' has unsupported classification contract {contract.get('type')!r}.")


def load_graph_node(package: str | Path | PackagedModel, *, outputs: dict[str, dict[str, Any]] | None = None, trainable: bool = False) -> GraphNode:
    """Load one packaged model as a graph node."""

    package = load_packaged_model(package) if isinstance(package, (str, Path)) else package
    if not isinstance(package, PackagedModel):
        raise TypeError(f"package must be a path or PackagedModel, got {type(package).__name__}.")
    package_outputs = outputs_from_package(package) if outputs is None else outputs
    return GraphNode(model=package.model, outputs=package_outputs, trainable=trainable)


def load_graph_nodes(packages: Mapping[str, str | Path | PackagedModel], *, trainable: bool = False) -> dict[str, GraphNode]:
    """Load a named package mapping as graph nodes."""
    if not packages:
        raise ValueError("load_graph_nodes requires at least one package.")
    return {name: load_graph_node(package, trainable=trainable) for name, package in packages.items()}


def reference_dataset_name(datasets: Mapping[str, BaseDataset]) -> str:
    """Choose the longest input window as the shared paired-data timeline."""
    if not datasets:
        raise ValueError("At least one dataset is required.")
    return max(datasets, key=lambda name: datasets[name].total_input)


class PairedDataset(Dataset):
    """Load synchronized package-native model inputs around one target center."""

    def __init__(self, datasets: Mapping[str, BaseDataset]):
        if not datasets:
            raise ValueError("PairedDataset requires at least one component dataset.")
        if any(not isinstance(dataset, BaseDataset) for dataset in datasets.values()):
            raise TypeError("Every paired component must be a BaseDataset.")
        self.datasets = dict(datasets)
        self.reference_name = reference_dataset_name(self.datasets)
        self.reference = self.datasets[self.reference_name]
        self.target_resolution = self.reference.target_resolution
        self.stride = self.reference.stride
        self.total_input = self.reference.total_input
        self.sample_frequency = None
        self.label_classes = list(self.reference.label_classes)
        self.classes = list(self.reference.classes)
        self.online_max_tries = self.reference.online_max_tries
        self.rejection_strategy = getattr(self.reference, "rejection_strategy", "patient_then_global")
        self.n_views = 1
        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.component_files = {}
        self.initialized = False
        self.all_patients = []

    def dataset_kwargs(self) -> dict:
        return self.reference.dataset_kwargs()

    def get_input_channels(self) -> dict[str, list[str]]:
        return {name: dataset.get_input_channels() for name, dataset in self.datasets.items()}

    def input_spec(self) -> dict[str, tuple[int, ...]]:
        return {
            name: (1, dataset.get_timeseries_len(), len(dataset.get_input_channels()))
            for name, dataset in self.datasets.items()
        }

    def get_n_patients(self) -> int:
        return len(self.edf_files)

    def get_patient_ranges(self) -> list[tuple[int, int]]:
        if not self.initialized:
            raise ValueError("PairedDataset is not initialized.")
        return list(zip(self.lower_bounds, self.upper_bounds))

    def get_classes(self) -> list[str]:
        return list(self.classes)

    def has_extra_target(self) -> bool:
        return self.reference.has_extra_target()

    def set_rejection_strategy(self, strategy: str) -> None:
        self.reference.set_rejection_strategy(strategy)
        self.rejection_strategy = self.reference.rejection_strategy

    def set_n_views(self, n_views: int) -> None:
        n_views = int(n_views)
        if n_views < 1:
            raise ValueError("n_views must be positive.")
        self.n_views = n_views

    def clone(self):
        datasets = {}
        for name, dataset in self.datasets.items():
            if isinstance(dataset, UnlabelledDataset):
                datasets[name] = dataset.clone()
            elif dataset.initialized:
                raise ValueError("A labelled PairedDataset can only be cloned before initialization.")
            else:
                datasets[name] = copy.deepcopy(dataset)
        return PairedDataset(datasets)

    def with_labels(self, labelled_dataset: BaseDataset):
        """Replace the reference with a compatible labelled dataset for evaluation."""
        if not isinstance(labelled_dataset, BaseDataset):
            raise TypeError(f"labelled_dataset must be a BaseDataset, got {type(labelled_dataset).__name__}.")
        datasets = {
            name: labelled_dataset if name == self.reference_name else dataset.clone()
            for name, dataset in self.datasets.items()
        }
        paired = PairedDataset(datasets)
        if paired.reference_name != self.reference_name:
            raise ValueError(f"The labelled dataset for '{self.reference_name}' must remain the longest paired input.")
        return paired

    def to_unlabelled(self):
        datasets = {
            name: dataset.clone() if isinstance(dataset, UnlabelledDataset) else UnlabelledDataset.from_dataset(dataset)
            for name, dataset in self.datasets.items()
        }
        return PairedDataset(datasets)

    def initialize(self, patients: Sequence[str | Path], num_workers: int = 4, *, strict: bool = False) -> None:
        self.all_patients = list(patients)
        self.initialized = False
        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.component_files = {}
        for dataset in self.datasets.values():
            dataset.initialize(patients, num_workers=num_workers, strict=strict)

        component_files = {
            name: {str(file.path): file for file in dataset.edf_files}
            for name, dataset in self.datasets.items()
        }
        common_paths = set.intersection(*(set(files) for files in component_files.values()))
        reference_files = [file for file in self.reference.edf_files if str(file.path) in common_paths]
        if not reference_files:
            raise ValueError("Paired datasets have no initialized recordings in common.")

        lower_bounds = []
        upper_bounds = []
        lower = 0
        for file in reference_files:
            lower_bounds.append(lower)
            lower += int(file.length)
            upper_bounds.append(lower)
        self.component_files = component_files
        self.edf_files = reference_files
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.label_classes = list(self.reference.label_classes)
        self.classes = list(self.reference.classes)
        self.initialized = True

    def __len__(self) -> int:
        return self.upper_bounds[-1] if self.upper_bounds else 0

    def get_item(self, reference_file, reference_start: pd.Timestamp):
        reference_item = self.reference.get_item(reference_file, reference_start)
        if reference_item is None:
            return None

        target_center = pd.Timestamp(reference_item["time"]) + self.target_resolution / 2
        inputs = {}
        patient_path = str(reference_file.path)
        for name, dataset in self.datasets.items():
            if name == self.reference_name:
                item = reference_item
            else:
                start = target_center - dataset.total_input / 2
                item = dataset.get_item(self.component_files[name][patient_path], start)
            if item is None:
                return None
            inputs[name] = item["data"]

        result = dict(reference_item)
        result["data"] = inputs
        return result

    def candidate_indices(self, original_index: int):
        original_patient_index = bisect.bisect_right(self.upper_bounds, original_index)
        yield original_index
        if self.rejection_strategy == "none":
            return
        if self.rejection_strategy in {"patient", "patient_then_global"}:
            for _ in range(max(0, self.online_max_tries - 1)):
                yield int(np.random.randint(self.lower_bounds[original_patient_index], self.upper_bounds[original_patient_index]))
        if self.rejection_strategy in {"global", "patient_then_global"}:
            for _ in range(self.online_max_tries):
                yield int(np.random.choice(len(self)))

    def item_from_index(self, index: int):
        patient_index = bisect.bisect_right(self.upper_bounds, index)
        reference_file = self.edf_files[patient_index]
        local_index = index - self.lower_bounds[patient_index]
        if reference_file.start_offsets is None:
            reference_start = reference_file.start_date + self.stride * local_index
        else:
            reference_start = reference_file.start_date + self.stride * int(reference_file.start_offsets[local_index])
        views = []
        for _ in range(self.n_views):
            item = self.get_item(reference_file, reference_start)
            if item is None:
                return None
            views.append(item)
        return views[0] if self.n_views == 1 else stack_repeated_views(views)

    def __getitem__(self, index: int):
        if not self.initialized:
            raise ValueError("PairedDataset is not initialized.")
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for candidate_index in self.candidate_indices(index):
            item = self.item_from_index(candidate_index)
            if item is not None:
                return item
        return None


def pair_datasets(datasets: Mapping[str, BaseDataset], *, target_resolution: str | pd.Timedelta | None = None, stride: str | pd.Timedelta | None = None) -> PairedDataset:
    """Combine package-native unlabelled datasets on the longest input timeline."""
    if not datasets:
        raise ValueError("pair_datasets requires at least one dataset.")
    if any(not isinstance(dataset, UnlabelledDataset) for dataset in datasets.values()):
        raise TypeError("pair_datasets expects package-native UnlabelledDataset instances.")
    components = {name: dataset.clone() for name, dataset in datasets.items()}
    reference_name = reference_dataset_name(components)
    if target_resolution is not None or stride is not None:
        reference_arguments = components[reference_name].dataset_kwargs()
        if target_resolution is not None:
            reference_arguments["target_resolution"] = target_resolution
        if stride is not None:
            reference_arguments["stride"] = stride
        components[reference_name] = UnlabelledDataset(**reference_arguments)
    return PairedDataset(components)


def channel_configs(channels: Sequence[ChannelConfig | Mapping[str, Any]]) -> list[ChannelConfig]:
    """Normalize optional YAML channel overrides to dataset channel objects."""
    return [channel if isinstance(channel, ChannelConfig) else ChannelConfig(**channel) for channel in channels]


def paired_dataset_from_packages(
    *,
    packages: Mapping[str, str | Path | PackagedModel],
    dataset_class: type[BaseDataset],
    input_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    **labelled_dataset_arguments,
) -> PairedDataset:
    """Build one labelled paired dataset from packaged signal contracts."""
    if not packages:
        raise ValueError("paired_dataset_from_packages requires at least one package.")
    if not isinstance(dataset_class, type) or not issubclass(dataset_class, BaseDataset):
        raise TypeError("dataset_class must be a BaseDataset class.")
    unknown_overrides = sorted(set(input_overrides or {}) - set(packages))
    if unknown_overrides:
        raise ValueError(f"input_overrides contains unknown package names: {unknown_overrides}.")

    loaded_packages = {
        name: load_packaged_model(package) if isinstance(package, (str, Path)) else package
        for name, package in packages.items()
    }
    if any(not isinstance(package, PackagedModel) for package in loaded_packages.values()):
        raise TypeError("Every package must be a path or PackagedModel.")
    if any(not isinstance(package.dataset, UnlabelledDataset) for package in loaded_packages.values()):
        raise TypeError("Every expert package must contain one UnlabelledDataset.")

    components = {}
    for name, package in loaded_packages.items():
        arguments = package.dataset.dataset_kwargs()
        arguments.update(dict((input_overrides or {}).get(name, {})))
        if "channels" in arguments:
            arguments["channels"] = channel_configs(arguments["channels"])
        components[name] = UnlabelledDataset(**arguments)

    reference_name = reference_dataset_name(components)
    reference_arguments = components[reference_name].dataset_kwargs()
    reference_arguments.update(labelled_dataset_arguments)
    if "channels" in reference_arguments:
        reference_arguments["channels"] = channel_configs(reference_arguments["channels"])
    components[reference_name] = dataset_class(**reference_arguments)
    return PairedDataset(components)


class ModelGraphClassifier(BaseModel, ClassifierModel):
    """Compose node models from either one shared tensor or paired native inputs."""

    def __init__(
        self,
        *,
        nodes: dict[str, GraphNode],
        method: str,
        edges: list[tuple[str, str]] | None = None,
        message_dim: int = 8,
        preprocessors: list[torch.nn.Module] | None = None,
    ):
        super().__init__(preprocessors=preprocessors)
        if method not in {"probability", "latent", "sei"}:
            raise ValueError("method must be 'probability', 'latent', or 'sei'.")
        if len(nodes) == 0:
            raise ValueError("ModelGraphClassifier requires at least one node.")
        if int(message_dim) < 1:
            raise ValueError("message_dim must be positive.")

        self.method = method
        self.message_dim = int(message_dim)
        self.node_names = list(nodes)
        self.models = torch.nn.ModuleDict({name: node.model for name, node in nodes.items()})
        self.node_outputs = {name: node.outputs for name, node in nodes.items()}
        self.trainable_nodes = {name for name, node in nodes.items() if node.trainable}
        self.feature_dims = {}
        self.node_input_specs = {}

        task_owners = {}
        for name, node in nodes.items():
            shape, meta = node.model.input_spec()
            if not isinstance(shape, tuple) or meta.get("layout") != "BTC" or len(shape) != 3:
                raise ValueError(f"Graph node '{name}' must expose a BTC input_spec, got {shape} and {meta}.")
            self.node_input_specs[name] = tuple(shape)
            if isinstance(node.model, EmbeddingModel):
                self.feature_dims[name] = int(node.model.feature_dim())
            for task in node.outputs:
                if task in task_owners:
                    raise ValueError(f"Output task '{task}' is owned by both '{task_owners[task]}' and '{name}'.")
                task_owners[task] = name

        if len(task_owners) == 0:
            raise ValueError("At least one graph node must declare an output task.")
        self.task_owners = task_owners
        self.edges = normalize_edges(edges or [], self.node_names)
        self.parents = {name: [] for name in self.node_names}
        for source, target in self.edges:
            self.parents[target].append(source)
        self.node_order = topological_order(self.node_names, self.edges)

        self.validate_capabilities(nodes)
        self.state_dims, head_dims = self.build_dimensions()
        self.interfaces = torch.nn.ModuleDict()
        if self.method == "sei":
            for source, target in self.edges:
                self.interfaces[edge_name(source, target)] = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.state_dims[source]),
                    torch.nn.Linear(self.state_dims[source], self.message_dim),
                    torch.nn.Tanh(),
                )

        self.heads = torch.nn.ModuleDict()
        for task, owner in self.task_owners.items():
            config = self.node_outputs[owner][task]
            self.heads[task] = torch.nn.Linear(head_dims[owner], config["sequence_len"] * len(config["classes"]))

        self.train(self.training)

    def validate_capabilities(self, nodes: dict[str, GraphNode]) -> None:
        for name, node in nodes.items():
            if self.method == "probability":
                if not isinstance(node.model, ClassifierModel):
                    raise TypeError(f"Probability graph node '{name}' must implement ClassifierModel.")
                if len(node.outputs) == 0:
                    raise ValueError(f"Probability graph node '{name}' must declare its classifier outputs.")
            elif not isinstance(node.model, EmbeddingModel):
                raise TypeError(f"{self.method.capitalize()} graph node '{name}' must implement EmbeddingModel.")

    def build_dimensions(self) -> tuple[dict[str, int], dict[str, int]]:
        output_dims = {
            name: sum(config["sequence_len"] * len(config["classes"]) for config in outputs.values())
            for name, outputs in self.node_outputs.items()
        }
        state_dims = {}
        head_dims = {}
        for name in self.node_order:
            if self.method == "probability":
                head_dims[name] = output_dims[name] + sum(output_dims[parent] for parent in self.parents[name])
                state_dims[name] = output_dims[name]
            elif self.method == "latent":
                state_dims[name] = self.feature_dims[name] + sum(state_dims[parent] for parent in self.parents[name])
                head_dims[name] = state_dims[name]
            else:
                state_dims[name] = self.feature_dims[name] + self.message_dim * len(self.parents[name])
                head_dims[name] = state_dims[name]
        return state_dims, head_dims

    def communication_scalars(self) -> int:
        """Return the number of scalar values transmitted across all graph edges per window."""
        if self.method == "sei":
            return len(self.edges) * self.message_dim
        return sum(self.state_dims[source] for source, target in self.edges)

    def train(self, mode: bool = True):
        super().train(mode)
        for name, model in self.models.items():
            if name not in self.trainable_nodes:
                model.eval()
        return self

    def node_input(self, inputs: torch.Tensor | Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        value = inputs[name] if isinstance(inputs, Mapping) else inputs
        expected = self.node_input_specs[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or tuple(value.shape[1:]) != expected[1:]:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"Graph node '{name}' expects input [B, {expected[1]}, {expected[2]}], got {shape}.")
        return value

    def validate_node_logits(self, name: str, logits) -> dict[str, torch.Tensor]:
        outputs = self.node_outputs[name]
        if isinstance(logits, torch.Tensor):
            if len(outputs) != 1:
                raise ValueError(f"Graph node '{name}' returned one tensor but declares outputs {sorted(outputs)}.")
            logits = {next(iter(outputs)): logits}
        if not isinstance(logits, dict) or set(logits) != set(outputs):
            keys = sorted(logits) if isinstance(logits, dict) else type(logits).__name__
            raise ValueError(f"Graph node '{name}' must return outputs {sorted(outputs)}, got {keys}.")

        for task, config in outputs.items():
            expected = (config["sequence_len"], len(config["classes"]))
            value = logits[task]
            if value.ndim != 3 or tuple(value.shape[1:]) != expected:
                raise ValueError(f"Graph node '{name}' output '{task}' must have shape [B, {expected[0]}, {expected[1]}], got {tuple(value.shape)}.")
        return logits

    def probabilities(self, name: str, logits) -> torch.Tensor:
        logits = self.validate_node_logits(name, logits)
        parts = []
        for task in self.node_outputs[name]:
            value = logits[task]
            parts.append(torch.softmax(value, dim=-1).flatten(start_dim=1))
        return torch.cat(parts, dim=1)

    def task_logits(self, name: str, state: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        for task, config in self.node_outputs[name].items():
            logits = self.heads[task](state)
            outputs[task] = logits.view(state.shape[0], config["sequence_len"], len(config["classes"]))
        return outputs

    def compute(self, x: torch.Tensor | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(x, Mapping) and set(x) != set(self.node_names):
            raise ValueError(f"Expected paired inputs for nodes {self.node_names}, got {sorted(x)}.")
        if not isinstance(x, (torch.Tensor, Mapping)):
            raise TypeError(f"ModelGraphClassifier expects a tensor or tensor mapping, got {type(x).__name__}.")

        base_states = {}
        for name in self.node_names:
            value = self.node_input(x, name)
            if self.method == "probability":
                base_states[name] = self.probabilities(name, self.models[name](value))
            else:
                features = self.models[name].features(value)
                expected = self.feature_dims[name]
                if features.ndim != 2 or features.shape[1] != expected:
                    raise ValueError(f"Graph node '{name}' must return embeddings shaped [B, {expected}], got {tuple(features.shape)}.")
                base_states[name] = features

        states = {}
        outputs = {}
        for name in self.node_order:
            if self.method == "probability":
                head_state = torch.cat([base_states[name], *(states[parent] for parent in self.parents[name])], dim=1)
                node_logits = self.task_logits(name, head_state)
                states[name] = self.probabilities(name, node_logits)
            elif self.method == "latent":
                states[name] = torch.cat([base_states[name], *(states[parent] for parent in self.parents[name])], dim=1)
                node_logits = self.task_logits(name, states[name])
            else:
                messages = [self.interfaces[edge_name(parent, name)](states[parent]) for parent in self.parents[name]]
                states[name] = torch.cat([base_states[name], *messages], dim=1)
                node_logits = self.task_logits(name, states[name])
            outputs.update(node_logits)
        return outputs

    def input_spec(self) -> tuple[Any, dict[str, Any]]:
        if len(self.node_names) == 1:
            return self.models[self.node_names[0]].input_spec()
        return (
            dict(self.node_input_specs),
            {"layout": "mapping", "inputs": list(self.node_names)},
        )


def normalize_edges(edges: list[tuple[str, str]], node_names: list[str]) -> list[tuple[str, str]]:
    normalized = []
    known = set(node_names)
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Graph edges must be (source, target) pairs, got {edge!r}.")
        source, target = str(edge[0]), str(edge[1])
        if source not in known or target not in known:
            raise ValueError(f"Graph edge {source!r}->{target!r} refers to an unknown node.")
        if source == target:
            raise ValueError(f"Graph edge {source!r}->{target!r} is a self-edge.")
        normalized.append((source, target))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Graph contains duplicate edges.")
    return normalized


def topological_order(node_names: list[str], edges: list[tuple[str, str]]) -> list[str]:
    children = {name: [] for name in node_names}
    indegree = {name: 0 for name in node_names}
    for source, target in edges:
        children[source].append(target)
        indegree[target] += 1

    ready = [name for name in node_names if indegree[name] == 0]
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(node_names):
        raise ValueError("Model graph must be acyclic.")
    return order


def edge_name(source: str, target: str) -> str:
    return f"{source}__to__{target}"
