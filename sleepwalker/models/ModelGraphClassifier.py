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

from sleepwalker.datasets.Basedataset import BaseDataset, stack_repeated_views
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment.package import PackagedModel, load_packaged_model
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


@dataclass
class GraphNode:
    """A model and the tasks it predicts inside a graph."""

    model: BaseModel
    outputs: dict[str, dict[str, Any]]
    native_outputs: dict[str, dict[str, Any]] | None = None
    input_offsets: list[int] | None = None
    prediction_indices: dict[str, list[int]] | None = None
    dataset: BaseDataset | None = None
    trainable: bool = False

    def __post_init__(self):
        self.native_outputs = self.outputs if self.native_outputs is None else self.native_outputs
        if not self.trainable:
            self.model.requires_grad_(False)
            self.model.eval()


def package_outputs(package: PackagedModel) -> dict[str, dict[str, Any]]:
    contract = package.classification_contract
    if contract["type"] == "single-head-multiclass":
        return {package.task: {"classes": list(contract["classes"]), "sequence_len": int(contract["sequence_len"])}}
    return {task: {"classes": list(config["classes"]), "sequence_len": int(config["n_steps"])} for task, config in contract["tasks"].items()}


def schedule_package(package: PackagedModel, graph_outputs: Mapping[str, Mapping[str, Any]]) -> tuple[list[int], dict[str, list[int]]]:
    """Return the package input offsets and closest native prediction for each graph step."""
    contract = package.classification_contract
    native_tasks = {package.task: contract} if contract["type"] == "single-head-multiclass" else contract["tasks"]
    stride_ns = int(package.dataset.stride.value)
    chosen_by_task = {}

    for task, native in native_tasks.items():
        graph = graph_outputs[task]
        native_classes = list(native["classes"])
        graph_classes = list(graph.get("classes", graph.get("labels")))
        if native_classes != graph_classes:
            raise ValueError(f"Class order differs between package and graph for task '{task}'.")
        native_steps = int(native.get("sequence_len", native.get("n_steps")))
        native_resolution = pd.to_timedelta(native["target_resolution"])
        native_step_ns = int((native_resolution / native_steps if contract["type"] == "single-head-multiclass" else native_resolution).value)
        native_offset_ns = int(pd.to_timedelta(native.get("target_offset", "0s")).value)
        graph_steps = int(graph.get("sequence_len", graph.get("n_steps")))
        graph_step_ns = int(pd.to_timedelta(graph["target_resolution"]).value)
        graph_offset_ns = int(pd.to_timedelta(graph.get("target_offset", "0s")).value)
        graph_centers = [graph_offset_ns + step * graph_step_ns + graph_step_ns / 2 for step in range(graph_steps)]
        native_span_ns = native_steps * native_step_ns
        graph_span_ns = graph_steps * graph_step_ns
        centered_input_ns = graph_offset_ns + graph_span_ns // 2 - native_offset_ns - native_span_ns // 2
        radius = int(np.ceil((graph_span_ns + native_span_ns) / stride_ns)) + 1
        possible_offsets = [centered_input_ns + call * stride_ns for call in range(-radius, radius + 1)]
        candidates = [(input_offset + native_offset_ns + step * native_step_ns + native_step_ns / 2, input_offset, step) for input_offset in possible_offsets for step in range(native_steps)]
        chosen_by_task[task] = [min(candidates, key=lambda candidate: (abs(candidate[0] - graph_time), candidate[0])) for graph_time in graph_centers]

    input_offsets = sorted({input_offset for chosen in chosen_by_task.values() for _, input_offset, _ in chosen})
    call_indices = {offset: index for index, offset in enumerate(input_offsets)}
    prediction_indices = {task: [call_indices[input_offset] * int(native_tasks[task].get("sequence_len", native_tasks[task].get("n_steps"))) + native_step for _, input_offset, native_step in chosen] for task, chosen in chosen_by_task.items()}
    return input_offsets, prediction_indices


def load_graph_node(package: str | Path | PackagedModel, *, graph_outputs: Mapping[str, Mapping[str, Any]] | None = None, trainable: bool = False) -> GraphNode:
    """Load one packaged model as a graph node."""

    package = load_packaged_model(package) if isinstance(package, (str, Path)) else package
    native_outputs = package_outputs(package)
    if graph_outputs is None:
        return GraphNode(model=package.model, outputs=native_outputs, dataset=package.dataset, trainable=trainable)
    outputs = {task: {"classes": list(graph_outputs[task].get("classes", graph_outputs[task].get("labels"))), "sequence_len": int(graph_outputs[task].get("sequence_len", graph_outputs[task].get("n_steps")))} for task in native_outputs}
    input_offsets, prediction_indices = schedule_package(package, {task: graph_outputs[task] for task in native_outputs})
    return GraphNode(model=package.model, outputs=outputs, native_outputs=native_outputs, input_offsets=input_offsets, prediction_indices=prediction_indices, dataset=package.dataset, trainable=trainable)


def load_graph_nodes(packages: Mapping[str, str | Path | PackagedModel], *, graph_outputs: Mapping[str, Mapping[str, Any]] | None = None, trainable: bool = False) -> dict[str, GraphNode]:
    """Load a named package mapping as graph nodes."""
    return {name: load_graph_node(package, graph_outputs=graph_outputs, trainable=trainable) for name, package in packages.items()}


class PairedDataset(Dataset):
    """Load package-native node windows at the timestamps supplied by a BaseDataset."""

    def __init__(self, datasets: Mapping[str, BaseDataset], *, base: BaseDataset, input_offsets: Mapping[str, Sequence[int]] | None = None):
        self.datasets = dict(datasets)
        self.base = base
        self.input_offsets = {name: list(offsets) for name, offsets in (input_offsets or {}).items() if offsets is not None}
        self.stride = self.base.stride
        self.total_input = self.base.total_input
        self.sample_frequency = None
        self.label_classes = list(self.base.label_classes)
        self.classes = list(self.base.classes)
        self.online_max_tries = self.base.online_max_tries
        self.rejection_strategy = self.base.rejection_strategy
        self.n_views = 1
        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.component_files = {}
        self.initialized = False
        self.all_patients = []

    def dataset_kwargs(self) -> dict:
        return self.base.dataset_kwargs()

    def get_input_channels(self) -> dict[str, list[str]]:
        return {name: dataset.get_input_channels() for name, dataset in self.datasets.items()}

    def input_spec(self) -> dict[str, tuple[int, ...]]:
        result = {}
        for name, dataset in self.datasets.items():
            native_shape = (dataset.get_timeseries_len(), len(dataset.get_input_channels()))
            result[name] = (1, len(self.input_offsets[name]), *native_shape) if name in self.input_offsets else (1, *native_shape)
        return result

    def get_n_patients(self) -> int:
        return len(self.edf_files)

    def get_patient_ranges(self) -> list[tuple[int, int]]:
        if not self.initialized:
            raise ValueError("PairedDataset is not initialized.")
        return list(zip(self.lower_bounds, self.upper_bounds))

    def get_classes(self) -> list[str]:
        return list(self.classes)

    def has_extra_target(self) -> bool:
        return self.base.has_extra_target()

    def set_rejection_strategy(self, strategy: str) -> None:
        self.base.set_rejection_strategy(strategy)
        self.rejection_strategy = self.base.rejection_strategy

    def set_n_views(self, n_views: int) -> None:
        n_views = int(n_views)
        if n_views < 1:
            raise ValueError("n_views must be positive.")
        self.n_views = n_views

    def clone(self):
        datasets = {name: dataset.clone() if isinstance(dataset, UnlabelledDataset) else copy.deepcopy(dataset) for name, dataset in self.datasets.items()}
        base = self.base.clone() if isinstance(self.base, UnlabelledDataset) else copy.deepcopy(self.base)
        return PairedDataset(datasets, base=base, input_offsets=self.input_offsets)

    def with_labels(self, labelled_dataset: BaseDataset):
        """Use a compatible labelled BaseDataset for evaluation."""
        channels = labelled_dataset.channels
        datasets = {}
        for name, dataset in self.datasets.items():
            source = dataset if isinstance(dataset, UnlabelledDataset) else UnlabelledDataset.from_dataset(dataset)
            if channels:
                expected = source.get_input_channels()
                actual = [channel.logical_name for channel in channels]
                if actual != expected:
                    raise ValueError(f"Evaluation channel override for node '{name}' must provide logical channels {expected}, got {actual}.")
                datasets[name] = source.clone(channels=copy.deepcopy(channels), assume_units_if_missing=labelled_dataset.assume_units_if_missing)
            else:
                datasets[name] = source.clone()
        return PairedDataset(datasets, base=labelled_dataset, input_offsets=self.input_offsets)

    def to_unlabelled(self):
        datasets = {
            name: dataset.clone() if isinstance(dataset, UnlabelledDataset) else UnlabelledDataset.from_dataset(dataset)
            for name, dataset in self.datasets.items()
        }
        base = self.base.clone() if isinstance(self.base, UnlabelledDataset) else UnlabelledDataset.from_dataset(self.base)
        return PairedDataset(datasets, base=base, input_offsets=self.input_offsets)

    def set_input_offsets(self, input_offsets: Mapping[str, Sequence[int]]) -> None:
        self.input_offsets = {name: list(offsets) for name, offsets in input_offsets.items() if offsets is not None}

    def initialize(self, patients: Sequence[str | Path], num_workers: int = 4, *, strict: bool = False) -> None:
        self.all_patients = list(patients)
        self.initialized = False
        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.component_files = {}
        self.base.initialize(patients, num_workers=num_workers, strict=strict)
        for dataset in self.datasets.values():
            if dataset is self.base:
                continue
            dataset.initialize(patients, num_workers=num_workers, strict=strict)

        component_files = {
            name: {str(file.path): file for file in dataset.edf_files}
            for name, dataset in self.datasets.items()
        }
        base_paths = {str(file.path) for file in self.base.edf_files}
        common_paths = set.intersection(base_paths, *(set(files) for files in component_files.values()))
        base_files = [file for file in self.base.edf_files if str(file.path) in common_paths]
        if not base_files:
            raise ValueError("Paired datasets have no initialized recordings in common.")

        lower_bounds = []
        upper_bounds = []
        lower = 0
        for file in base_files:
            lower_bounds.append(lower)
            lower += int(file.length)
            upper_bounds.append(lower)
        self.component_files = component_files
        self.edf_files = base_files
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.label_classes = list(self.base.label_classes)
        self.classes = list(self.base.classes)
        self.initialized = True

    def __len__(self) -> int:
        return self.upper_bounds[-1] if self.upper_bounds else 0

    def get_item(self, base_file, start: pd.Timestamp):
        base_item = self.base.get_target_item(base_file, start)
        if base_item is None:
            return None

        inputs = {}
        patient_path = str(base_file.path)
        for name, dataset in self.datasets.items():
            if name in self.input_offsets:
                items = [dataset.get_item(self.component_files[name][patient_path], pd.Timestamp(base_item["time"]) + pd.Timedelta(int(offset), unit="ns")) for offset in self.input_offsets[name]]
                if any(item is None for item in items):
                    return None
                inputs[name] = torch.stack([item["data"] for item in items])
                continue
            target_center = pd.Timestamp(base_item["time"]) + self.base.total_input / 2
            start = target_center - dataset.total_input / 2
            item = dataset.get_item(self.component_files[name][patient_path], start)
            if item is None:
                return None
            inputs[name] = item["data"]

        result = dict(base_item)
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
        base_file = self.edf_files[patient_index]
        local_index = index - self.lower_bounds[patient_index]
        if base_file.start_offsets is None:
            base_start = base_file.start_date + self.stride * local_index
        else:
            base_start = base_file.start_date + self.stride * int(base_file.start_offsets[local_index])
        views = []
        for _ in range(self.n_views):
            item = self.get_item(base_file, base_start)
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


def first_linear(module: torch.nn.Module) -> torch.nn.Linear:
    if isinstance(module, torch.nn.Linear):
        return module
    for child in module.children():
        try:
            return first_linear(child)
        except TypeError:
            continue
    raise TypeError(f"Classification head {type(module).__name__} does not contain a Linear layer to expand.")


def replace_module(module: torch.nn.Module, target: torch.nn.Module, replacement: torch.nn.Module) -> bool:
    for name, child in module.named_children():
        if child is target:
            setattr(module, name, replacement)
            return True
        if replace_module(child, target, replacement):
            return True
    return False


def model_classification_head(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "fc") and model.fc is not None:
        return model.fc
    if hasattr(model, "head") and model.head is not None:
        return model.head
    raise TypeError(f"Graph node model {type(model).__name__} must expose its native classification head as 'fc' or 'head'.")


class ExpandedHead(torch.nn.Module):
    """Copy a native classifier and extend its first linear layer with graph context."""

    def __init__(self, native_head: torch.nn.Module, *, feature_dim: int, context_dim: int, native_output: Mapping[str, Any], graph_output: Mapping[str, Any], prediction_indices: list[int] | None, context_init_std: float):
        super().__init__()
        self.classifier = copy.deepcopy(native_head)
        self.feature_dim = int(feature_dim)
        self.context_dim = int(context_dim)
        self.native_sequence_len = int(native_output["sequence_len"])
        self.graph_sequence_len = int(graph_output["sequence_len"])
        self.n_classes = len(graph_output["classes"])
        self.prediction_indices = None if prediction_indices is None else list(prediction_indices)

        native_first = first_linear(self.classifier)
        if self.feature_dim == native_first.in_features:
            self.feature_layout = "flat"
        elif self.feature_dim == self.native_sequence_len * native_first.in_features:
            self.feature_layout = "sequence"
        else:
            raise ValueError(f"Native head expects {native_first.in_features} features, but the model exposes {self.feature_dim} features over {self.native_sequence_len} output steps.")

        expanded = torch.nn.Linear(native_first.in_features + self.context_dim, native_first.out_features, bias=native_first.bias is not None, device=native_first.weight.device, dtype=native_first.weight.dtype)
        with torch.no_grad():
            expanded.weight[:, :native_first.in_features].copy_(native_first.weight)
            if self.context_dim:
                torch.nn.init.normal_(expanded.weight[:, native_first.in_features:], mean=0.0, std=float(context_init_std))
            if native_first.bias is not None:
                expanded.bias.copy_(native_first.bias)
        if self.classifier is native_first:
            self.classifier = expanded
        elif not replace_module(self.classifier, native_first, expanded):
            raise RuntimeError("Could not replace the copied classification head's first Linear layer.")
        self.classifier.requires_grad_(True)

    @property
    def in_features(self) -> int:
        return first_linear(self.classifier).in_features

    @property
    def weight(self) -> torch.nn.Parameter:
        return first_linear(self.classifier).weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return first_linear(self.classifier).bias

    def forward(self, features: torch.Tensor, context: torch.Tensor, batch_size: int, calls: int) -> torch.Tensor:
        if tuple(features.shape) != (batch_size * calls, self.feature_dim):
            raise ValueError(f"Expected native features shaped {(batch_size * calls, self.feature_dim)}, got {tuple(features.shape)}.")
        if tuple(context.shape) != (batch_size, self.context_dim):
            raise ValueError(f"Expected graph context shaped {(batch_size, self.context_dim)}, got {tuple(context.shape)}.")
        repeated_context = context.repeat_interleave(calls, dim=0)
        if self.feature_layout == "sequence":
            step_features = features.view(batch_size * calls, self.native_sequence_len, -1)
            head_input = torch.cat([step_features, repeated_context.unsqueeze(1).expand(-1, self.native_sequence_len, -1)], dim=-1)
            logits = self.classifier(head_input)
        else:
            logits = self.classifier(torch.cat([features, repeated_context], dim=-1))
        logits = logits.view(batch_size, calls * self.native_sequence_len, self.n_classes)
        if self.prediction_indices is not None:
            indices = torch.as_tensor(self.prediction_indices, device=logits.device)
            logits = logits.index_select(1, indices)
        expected = (batch_size, self.graph_sequence_len, self.n_classes)
        if tuple(logits.shape) != expected:
            raise ValueError(f"Expanded native head produced logits shaped {tuple(logits.shape)}, expected {expected}.")
        return logits


class ModelGraphClassifier(BaseModel, ClassifierModel):
    """Compose node models from either one shared tensor or paired native inputs."""

    def __init__(
        self,
        *,
        nodes: dict[str, GraphNode],
        method: str,
        edges: list[tuple[str, str]] | None = None,
        message_dim: int = 8,
        context_init_std: float = 0.005,
        preprocessors: list[torch.nn.Module] | None = None,
    ):
        super().__init__(preprocessors=preprocessors)
        if method not in {"probability", "latent", "sei"}:
            raise ValueError("method must be 'probability', 'latent', or 'sei'.")
        self.method = method
        self.message_dim = int(message_dim)
        self.context_init_std = float(context_init_std)
        if self.context_init_std < 0:
            raise ValueError("context_init_std must be non-negative.")
        self.node_names = list(nodes)
        self.models = torch.nn.ModuleDict({name: node.model for name, node in nodes.items()})
        self.node_outputs = {name: node.outputs for name, node in nodes.items()}
        self.native_outputs = {name: node.native_outputs for name, node in nodes.items()}
        self.input_offsets = {name: node.input_offsets for name, node in nodes.items()}
        self.prediction_indices = {name: node.prediction_indices for name, node in nodes.items()}
        self.input_datasets = {name: node.dataset for name, node in nodes.items()}
        self.trainable_nodes = {name for name, node in nodes.items() if node.trainable}
        self.feature_dims = {}
        self.node_input_specs = {}

        task_owners = {}
        for name, node in nodes.items():
            shape, _ = node.model.input_spec()
            offsets = self.input_offsets[name]
            self.node_input_specs[name] = (shape[0], len(offsets), *shape[1:]) if offsets is not None else tuple(shape)
            if isinstance(node.model, EmbeddingModel):
                calls = len(offsets) if offsets is not None else 1
                self.feature_dims[name] = int(node.model.feature_dim()) * calls
            else:
                raise TypeError(f"Graph node '{name}' must implement EmbeddingModel so its native head can be reused.")
            if len(node.outputs) != 1 or set(node.outputs) != set(node.native_outputs):
                raise ValueError(f"Graph node '{name}' must own exactly one matching native and graph task.")
            for task in node.outputs:
                if task in task_owners:
                    raise ValueError(f"Output task '{task}' is owned by both '{task_owners[task]}' and '{name}'.")
                task_owners[task] = name

        self.task_owners = task_owners
        self.edges = [tuple(edge) for edge in (edges or [])]
        if len(self.edges) != len(set(self.edges)):
            raise ValueError("Graph contains duplicate edges.")
        self.parents = {name: [] for name in self.node_names}
        for source, target in self.edges:
            self.parents[target].append(source)
        self.node_order = topological_order(self.node_names, self.edges)

        self.state_dims, context_dims = self.build_dimensions()
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
            native = self.native_outputs[owner][task]
            graph = self.node_outputs[owner][task]
            if native["classes"] != graph["classes"]:
                raise ValueError(f"Native and graph class order differs for node '{owner}'.")
            indices = None if self.prediction_indices[owner] is None else self.prediction_indices[owner][task]
            self.heads[task] = ExpandedHead(model_classification_head(self.models[owner]), feature_dim=int(self.models[owner].feature_dim()), context_dim=context_dims[owner], native_output=native, graph_output=graph, prediction_indices=indices, context_init_std=self.context_init_std)

        self.train(self.training)

    def pair_dataset(self, base: BaseDataset) -> PairedDataset:
        """Bind the graph's packaged input datasets to a BaseDataset timeline."""
        datasets = {name: dataset.clone() if isinstance(dataset, UnlabelledDataset) else copy.deepcopy(dataset) for name, dataset in self.input_datasets.items()}
        return PairedDataset(datasets, base=base, input_offsets=self.input_offsets)

    def build_dimensions(self) -> tuple[dict[str, int], dict[str, int]]:
        graph_output_dims = {
            name: sum(config["sequence_len"] * len(config["classes"]) for config in outputs.values())
            for name, outputs in self.node_outputs.items()
        }
        state_dims = {}
        context_dims = {}
        for name in self.node_order:
            if self.method == "probability":
                context_dims[name] = sum(graph_output_dims[parent] for parent in self.parents[name])
                state_dims[name] = graph_output_dims[name]
            elif self.method == "latent":
                context_dims[name] = sum(state_dims[parent] for parent in self.parents[name])
                state_dims[name] = self.feature_dims[name] + sum(state_dims[parent] for parent in self.parents[name])
            else:
                context_dims[name] = self.message_dim * len(self.parents[name])
                state_dims[name] = self.feature_dims[name] + self.message_dim * len(self.parents[name])
        return state_dims, context_dims

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

    def aligned_probabilities(self, name: str, value: torch.Tensor) -> torch.Tensor:
        batch_size, calls = value.shape[:2]
        logits = self.models[name](value.flatten(0, 1))
        if isinstance(logits, torch.Tensor):
            logits = {next(iter(self.native_outputs[name])): logits}
        aligned = []
        for task, output in self.native_outputs[name].items():
            classes = len(output["classes"])
            probabilities = torch.softmax(logits[task], dim=-1).view(batch_size, calls * output["sequence_len"], classes)
            indices = torch.as_tensor(self.prediction_indices[name][task], device=probabilities.device)
            aligned.append(probabilities.index_select(1, indices).flatten(start_dim=1))
        return torch.cat(aligned, dim=1)

    def probabilities(self, logits, outputs: dict[str, dict[str, Any]]) -> torch.Tensor:
        if isinstance(logits, torch.Tensor):
            logits = {next(iter(outputs)): logits}
        parts = []
        for task in outputs:
            value = logits[task]
            parts.append(torch.softmax(value, dim=-1).flatten(start_dim=1))
        return torch.cat(parts, dim=1)

    def task_logits(self, name: str, features: torch.Tensor, context: torch.Tensor, batch_size: int, calls: int) -> dict[str, torch.Tensor]:
        outputs = {}
        for task in self.node_outputs[name]:
            outputs[task] = self.heads[task](features, context, batch_size, calls)
        return outputs

    def compute(self, x: torch.Tensor | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base_states = {}
        features_by_node = {}
        shapes_by_node = {}
        for name in self.node_names:
            value = x[name] if isinstance(x, Mapping) else x
            if self.input_offsets[name] is None:
                batch_size, calls = value.shape[0], 1
                features = self.models[name].features(value)
            else:
                batch_size, calls = value.shape[:2]
                features = self.models[name].features(value.flatten(0, 1))
            features_by_node[name] = features
            shapes_by_node[name] = (batch_size, calls)
            base_states[name] = features.reshape(batch_size, calls, -1).flatten(start_dim=1)

        states = {}
        outputs = {}
        for name in self.node_order:
            if self.method == "probability":
                context = torch.cat([states[parent] for parent in self.parents[name]], dim=1) if self.parents[name] else base_states[name].new_empty((base_states[name].shape[0], 0))
                batch_size, calls = shapes_by_node[name]
                node_logits = self.task_logits(name, features_by_node[name], context, batch_size, calls)
                states[name] = self.probabilities(node_logits, self.node_outputs[name])
            elif self.method == "latent":
                context = torch.cat([states[parent] for parent in self.parents[name]], dim=1) if self.parents[name] else base_states[name].new_empty((base_states[name].shape[0], 0))
                states[name] = torch.cat([base_states[name], context], dim=1)
                batch_size, calls = shapes_by_node[name]
                node_logits = self.task_logits(name, features_by_node[name], context, batch_size, calls)
            else:
                messages = [self.interfaces[edge_name(parent, name)](states[parent]) for parent in self.parents[name]]
                context = torch.cat(messages, dim=1) if messages else base_states[name].new_empty((base_states[name].shape[0], 0))
                states[name] = torch.cat([base_states[name], context], dim=1)
                batch_size, calls = shapes_by_node[name]
                node_logits = self.task_logits(name, features_by_node[name], context, batch_size, calls)
            outputs.update(node_logits)
        return outputs

    def input_spec(self) -> tuple[Any, dict[str, Any]]:
        if len(self.node_names) == 1 and self.input_offsets[self.node_names[0]] is None:
            return self.models[self.node_names[0]].input_spec()
        return (
            dict(self.node_input_specs),
            {"layout": "mapping", "inputs": list(self.node_names), "input_offsets": self.input_offsets},
        )


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
