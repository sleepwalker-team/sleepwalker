from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import json
from json import JSONDecodeError
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

    if (
    not isinstance(metadata, dict)
    or metadata.get("format") != "sleepwalker-swmodel-v1"
    ):
        raise InvalidModelPackageError(
        f"Unsupported model package format: {package_path.name}"
    )

    model = metadata["model"]

    return ModelInfo(
        id=model["id"],
        name=model["name"],
        task=model["task"],
        classes=tuple(model["classes"]),
        input_channels=tuple(model["input_channels"]),
        capabilities=tuple(model["capabilities"]),
        output_resolution=float(model["output_resolution"]),
        package_path=package_path,
    )
    
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
