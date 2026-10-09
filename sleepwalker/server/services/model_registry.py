import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from zipfile import BadZipFile, ZipFile

class InvalidModelPackageError(Exception):
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

def _read_string_list(
    model: dict,
    field: str,
) -> tuple[str, ...]:
    value = model[field]

    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item.strip()
            for item in value
        )
    ):
        raise TypeError(
            f"Model field '{field}' must be a non-empty list of strings."
        )

    return tuple(value)

def _read_non_empty_string(
    model: dict,
    field: str,
) -> str:
    value = model[field]

    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"Model field '{field}' must be a non-empty string."
        )

    return value


def _read_positive_number(
    model: dict,
    field: str,
) -> float:
    value = model[field]

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            f"Model field '{field}' must be a number."
        )

    result = float(value)

    if not math.isfinite(result) or result <= 0:
        raise ValueError(
            f"Model field '{field}' must be positive."
        )

    return result
    
def read_model_manifest(package_path: Path) -> ModelInfo:
    try:
        with ZipFile(package_path, "r") as package:
            metadata = json.loads(
                package.read("meta.json").decode("utf-8")
            )
    except (
        OSError,
        BadZipFile,
        KeyError,
        UnicodeDecodeError,
        JSONDecodeError,
    ) as exc:
        raise InvalidModelPackageError(
            f"Invalid model package: {package_path.name}"
        ) from exc

    if (not isinstance(metadata, dict) or metadata.get("format") != "sleepwalker-swmodel-v1"):
        raise InvalidModelPackageError(f"Unsupported model package format: {package_path.name}")

    try:
        model = metadata["model"]

        if not isinstance(model, dict):
            raise TypeError("The model field must be an object.")

        return ModelInfo(
            id=_read_non_empty_string(model, "id"),
            name=_read_non_empty_string(model, "name"),
            task=_read_non_empty_string(model, "task"),
            classes=_read_string_list(model, "classes"),
            input_channels=_read_string_list(model, "input_channels"),
            capabilities=_read_string_list(model, "capabilities"),
            output_resolution=_read_positive_number(model, "output_resolution"),
            package_path=package_path,
            )

    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidModelPackageError(
            f"Invalid model metadata: {package_path.name}"
        ) from exc
    
class ModelRegistry:
    def __init__(
        self,
        models: Iterable[ModelInfo] = (),
    ) -> None:
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
        return sorted(
            self._models.values(),
            key=lambda model: model.id,
        )

def load_model_registry(
    model_directory: Path,
) -> ModelRegistry:
    if not model_directory.is_dir():
        raise ValueError(
            f"Model directory does not exist: {model_directory}"
        )

    package_paths = sorted(
        path
        for path in model_directory.glob("*.swmodel")
        if path.is_file()
    )

    return ModelRegistry(
        read_model_manifest(path)
        for path in package_paths
    )
