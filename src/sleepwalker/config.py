"""Shared YAML-to-Python construction helpers for repository command-line tools."""

from functools import partial
import importlib
import inspect
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from sleepwalker.datasets.Basedataset import ChannelConfig


CONTEXT_FIELDS = {"classes", "input_channels", "n_channels", "ts_len", "sampling_frequency", "sequence_len"}


class ScientificNotationLoader(yaml.SafeLoader):
    """Safe YAML loader that recognizes exponent notation without a decimal point."""


ScientificNotationLoader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"), list("-+0123456789."))


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=ScientificNotationLoader) or {}


def import_name(name: str) -> Any:
    """Import a fully qualified module, class, method, or function name."""
    parts = name.split(".")
    for boundary in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:boundary]))
        except ModuleNotFoundError:
            continue
        for part in parts[boundary:]:
            value = getattr(value, part)
        return value
    raise ImportError(name)


def build_value(value: Any, context: Mapping[str, Any] | None = None) -> Any:
    """Build nested named components and import qualified callable values."""
    context = {} if context is None else context
    if isinstance(value, list):
        return [build_value(item, context) for item in value]
    if isinstance(value, dict):
        if isinstance(value.get("name"), str) and "." in value["name"]:
            return build_component(value, context)
        return {key: build_value(item, context) for key, item in value.items()}
    if isinstance(value, str) and value.startswith(("sleepwalker.", "torch.")):
        return import_name(value)
    return value


def build_component(spec: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> Any:
    """Instantiate one named component, injecting standard dataset context."""
    context = {} if context is None else dict(context)
    symbol = import_name(str(spec["name"]))
    raw_arguments = {key: value for key, value in spec.items() if key != "name"}
    local_context = dict(context)
    if isinstance(raw_arguments.get("input_channels"), list):
        local_context["input_channels"] = raw_arguments["input_channels"]
        local_context["n_channels"] = len(raw_arguments["input_channels"])
    arguments = {key: build_value(value, local_context) for key, value in raw_arguments.items()}
    parameters = inspect.signature(symbol).parameters
    for key in CONTEXT_FIELDS:
        if key in {"classes", "sequence_len"} and "task_config" in arguments:
            continue
        if key in parameters and key in local_context and key not in arguments:
            arguments[key] = local_context[key]
    return symbol(**arguments)


def build_callback(spec: str | Mapping[str, Any] | None) -> Any:
    """Bind a configured callback without calling it."""
    if spec is None or callable(spec):
        return spec
    if isinstance(spec, str):
        return import_name(spec)
    symbol = import_name(str(spec["name"]))
    arguments = {key: build_value(value) for key, value in spec.items() if key != "name"}
    return partial(symbol, **arguments) if arguments else symbol


def build_factory(spec: Mapping[str, Any], dependency: str) -> Any:
    """Delay optimizer or scheduler construction until its dependency exists."""
    symbol = import_name(str(spec["name"]))
    arguments = {key: build_value(value) for key, value in spec.items() if key != "name"}
    if dependency == "model":
        return lambda model: symbol(model.parameters(), **arguments)
    return partial(symbol, **arguments)


def build_channel(spec: Mapping[str, Any]) -> ChannelConfig:
    arguments = dict(spec)
    if "preprocessors" in arguments:
        arguments["preprocessors"] = build_value(arguments["preprocessors"])
    return ChannelConfig(**arguments)


def apply_patient_filter(spec: str | Mapping[str, Any] | list, patients: list[str], dataset, num_workers: int = 1) -> list[str]:
    """Apply configured filters in order to paths already assigned to a split."""
    filters = spec if isinstance(spec, list) else [spec]
    filtered = list(patients)
    for filter_spec in filters:
        if isinstance(filter_spec, str):
            patient_filter = import_name(filter_spec)
            arguments = {}
        else:
            patient_filter = import_name(str(filter_spec["name"]))
            arguments = {key: build_value(value) for key, value in filter_spec.items() if key != "name"}
        parameters = inspect.signature(patient_filter).parameters
        if "dataset" in parameters:
            arguments["dataset"] = dataset
        if "num_workers" in parameters and "num_workers" not in arguments:
            arguments["num_workers"] = num_workers
        current = [str(path) for path in patient_filter(patients=filtered, **arguments)]
        unexpected = sorted(set(current) - set(filtered))
        if unexpected:
            raise ValueError(f"A patient filter may only remove paths from its existing partition; it added {unexpected[:5]}.")
        if len(current) != len(set(current)):
            raise ValueError("A patient filter returned duplicate paths.")
        filtered = current
    return filtered
