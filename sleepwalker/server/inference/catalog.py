"""Available models: whatever valid packages currently lie in the model directory.

Models are retrained daily, so nothing is registered. Each lookup reads the
directory again, which makes a new package usable without a restart. Only
manifest JSON is read here; weights are never deserialized in the API process.
"""

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ..results.predictions import PredictionKind, background_class

logger = logging.getLogger(__name__)
PACKAGE_FORMAT = "sleepwalker-packaged-model-v1"

Text = Annotated[str, Field(strict=True, pattern=r"\S")]


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

    def provenance(self) -> dict:
        """Describe exactly this package; the ID alone stays the same across retrainings."""
        metadata = {
            "id": self.id,
            "name": self.name,
            "task": self.task,
            "classes": list(self.classes),
            "output_kind": self.output_kind.value,
            "output_resolution": self.output_resolution,
            "input_channels": list(self.input_channels),
            "capabilities": list(self.capabilities),
        }
        if self.payload_sha256 is not None:
            metadata["version"] = f"sha256:{self.payload_sha256}"
        return metadata


def _require_unique(values: list[str]) -> list[str]:
    if len(set(values)) != len(values):
        raise ValueError("Duplicate values are not allowed.")
    return values


class _Channel(BaseModel):
    logical_name: Text
    physical_names: list[Text] = Field(min_length=1)

    _unique = field_validator("physical_names")(_require_unique)


class _Contract(BaseModel):
    classes: list[Text] = Field(min_length=1)
    sequence_len: Annotated[int, Field(strict=True, gt=0)]
    target_resolution: Text

    _unique = field_validator("classes")(_require_unique)


class _Manifest(BaseModel):
    name: Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    task: Text
    input_channels: list[Text] = Field(min_length=1)
    capabilities: list[Text] = Field(min_length=1)
    payload: Text
    sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    classification: _Contract

    _unique = field_validator("input_channels", "capabilities")(_require_unique)


def read_model_manifest(package_path: Path) -> ModelInfo:
    """Read administrator-supplied JSON only; never deserialize weights in the API."""
    try:
        metadata = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("format_version") != PACKAGE_FORMAT:
            raise InvalidModelPackageError(f"Unsupported model package format: {package_path.name}")

        contract_type = metadata["classification"].get("type")
        if contract_type == "multitask":
            raise UnsupportedModelPackageError(f"Multi-task package is not enabled yet: {package_path.name}")
        if contract_type != "single-head-multiclass":
            raise ValueError("Unsupported classification contract.")

        manifest = _Manifest.model_validate(metadata)
        channels = [_Channel.model_validate(channel) for channel in metadata["config"]["data"]["channels"]]
        aliases = {channel.logical_name: tuple(channel.physical_names) for channel in channels}
        if list(aliases) != manifest.input_channels:
            raise ValueError("Channel configuration must match logical input channels in order.")
        if "classification" not in manifest.capabilities:
            raise ValueError("Package does not declare classification capability.")

        contract = manifest.classification
        resolution = pd.to_timedelta(contract.target_resolution).total_seconds() / contract.sequence_len
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("Classification resolution must be positive.")

        payload_path = package_path / manifest.payload
        inside_package = payload_path.resolve().is_relative_to(package_path.resolve())
        if Path(manifest.payload).name != manifest.payload or "\\" in manifest.payload or not inside_package or not payload_path.is_file():
            raise ValueError("Payload must be a file inside its model package.")

        return ModelInfo(
            id=manifest.name, name=manifest.name, task=manifest.task,
            classes=tuple(contract.classes), input_channels=tuple(manifest.input_channels),
            capabilities=tuple(manifest.capabilities), output_resolution=resolution,
            package_path=package_path, channel_aliases=aliases, payload_sha256=manifest.sha256,
            output_kind=PredictionKind.EVENT if background_class(contract.classes) else PredictionKind.EPOCH,
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise InvalidModelPackageError(f"Invalid model metadata: {package_path.name}") from exc


class ModelCatalog:
    """Finds models in `directory`'s subfolders; `models` adds fixed entries for tests."""

    def __init__(self, directory: Path | None = None, models: Iterable[ModelInfo] = ()) -> None:
        if directory is not None and not directory.is_dir():
            raise ValueError(f"Model directory does not exist: {directory}")
        self.directory = directory
        self._fixed = {model.id: model for model in models}
        self._reported: set[Path] = set()

    def get(self, model_id: str) -> ModelInfo | None:
        return self._models().get(model_id)

    def list(self) -> list[ModelInfo]:
        return sorted(self._models().values(), key=lambda model: model.id)

    def _models(self) -> dict[str, ModelInfo]:
        models = dict(self._fixed)
        newest: dict[str, int] = {}
        for manifest in sorted(self.directory.glob("*/manifest.json")) if self.directory else ():
            package = manifest.parent.resolve()
            try:
                model = read_model_manifest(package)
                modified = manifest.stat().st_mtime_ns
            except (InvalidModelPackageError, OSError) as exc:
                self._report_once(package, exc)
                continue
            self._reported.discard(package)
            # A retrained package may reuse the name of an older one: the newest wins.
            if modified >= newest.get(model.id, -1):
                newest[model.id] = modified
                models[model.id] = model
        return models

    def _report_once(self, package: Path, exc: Exception) -> None:
        # Lookups are frequent, and a package may just be in the middle of being copied.
        if package not in self._reported:
            self._reported.add(package)
            logger.warning("Skipping model package %s: %s", package, exc)
