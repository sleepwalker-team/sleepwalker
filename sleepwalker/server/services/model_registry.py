import json
import logging
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..stores.prediction_store import PredictionKind

logger = logging.getLogger(__name__)
PACKAGE_FORMAT = "sleepwalker-packaged-model-v1"


class InvalidModelPackageError(Exception):
    pass


class UnsupportedModelPackageError(InvalidModelPackageError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    task: str
    classes: tuple[str, ...]
    input_channels: tuple[str, ...]
    capabilities: tuple[str, ...]
    output_resolution: float
    package_path: Path
    output_kind: PredictionKind = PredictionKind.EPOCH
    channel_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    payload_sha256: str | None = None


def read_string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty string.")
    return value


def read_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a non-empty list of strings.")
    values = tuple(read_string(item) for item in value)
    if len(set(values)) != len(values):
        raise ValueError("Duplicate values are not allowed.")
    return values


def read_model_manifest(package_path: Path) -> ModelInfo:
    """Read administrator-supplied JSON only; never deserialize weights in the API."""
    try:
        metadata = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("format_version") != PACKAGE_FORMAT:
            raise InvalidModelPackageError(f"Unsupported model package format: {package_path.name}")

        contract = metadata["classification"]
        if not isinstance(contract, dict):
            raise ValueError("Missing classification contract.")
        if contract.get("type") == "multitask":
            raise UnsupportedModelPackageError(f"Multi-task package is not enabled yet: {package_path.name}")
        if contract.get("type") != "single-head-multiclass":
            raise ValueError("Unsupported classification contract.")

        name = read_string(metadata["name"])
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise ValueError("Package name must also be a URL-safe model ID.")
        sequence_len = contract["sequence_len"]
        if isinstance(sequence_len, bool) or not isinstance(sequence_len, int) or sequence_len <= 0:
            raise ValueError("Classification sequence_len must be a positive integer.")
        resolution = pd.to_timedelta(read_string(contract["target_resolution"])).total_seconds() / sequence_len
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("Classification resolution must be positive.")

        channels = read_strings(metadata["input_channels"])
        channel_configs = metadata["config"]["data"]["channels"]
        if not isinstance(channel_configs, list) or len(channel_configs) != len(channels):
            raise ValueError("Invalid channel configuration.")
        aliases = {
            read_string(channel["logical_name"]): read_strings(channel["physical_names"])
            for channel in channel_configs
        }
        if tuple(aliases) != channels:
            raise ValueError("Channel configuration must match logical input channels in order.")
        capabilities = read_strings(metadata["capabilities"])
        if "classification" not in capabilities:
            raise ValueError("Package does not declare classification capability.")

        payload = read_string(metadata["payload"])
        payload_path = package_path / payload
        if Path(payload).name != payload or "\\" in payload or not payload_path.resolve().is_relative_to(package_path.resolve()):
            raise ValueError("Payload must be a file inside its model package.")
        if not payload_path.is_file():
            raise ValueError("Model payload is missing.")
        checksum = read_string(metadata["sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ValueError("Invalid payload SHA256 digest.")

        return ModelInfo(
            id=name, name=name, task=read_string(metadata["task"]),
            classes=read_strings(contract["classes"]), input_channels=channels, capabilities=capabilities,
            output_resolution=resolution, package_path=package_path,
            channel_aliases=aliases, payload_sha256=checksum,
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidModelPackageError(f"Invalid model metadata: {package_path.name}") from exc


class ModelRegistry:
    def __init__(self, models: Iterable[ModelInfo] = ()) -> None:
        self._models: dict[str, ModelInfo] = {}
        for model in models:
            self.add(model)

    def add(self, model: ModelInfo) -> None:
        if model.id in self._models:
            raise ValueError(f"Model ID is already registered: {model.id}")
        self._models[model.id] = model

    def get(self, model_id: str) -> ModelInfo | None:
        return self._models.get(model_id)

    def list(self) -> list[ModelInfo]:
        return sorted(self._models.values(), key=lambda model: model.id)


def load_model_registry(model_directory: Path) -> ModelRegistry:
    if not model_directory.is_dir():
        raise ValueError(f"Model directory does not exist: {model_directory}")
    registry = ModelRegistry()
    for manifest in sorted(model_directory.glob("*/manifest.json")):
        try:
            registry.add(read_model_manifest(manifest.parent.resolve()))
        except UnsupportedModelPackageError as exc:
            logger.warning("%s", exc)
    return registry
