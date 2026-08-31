"""Reloadable model packages for a compatible Sleepwalker checkout."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any, Optional

import cloudpickle
import pandas as pd
import torch

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment.predictions import format_prediction_batch
from sleepwalker.models.BaseModel import ClassifierModel, EmbeddingModel
from sleepwalker.trainer.utils.disk import NumpyEncoder, json_ready
from sleepwalker.training.execution import RepeatedViewModel
from sleepwalker.training.loader import build_loader


FORMAT_VERSION = "sleepwalker-packaged-model-v1"


class CloudpickleAdapter:
    __name__ = "cloudpickle"
    Pickler = cloudpickle.Pickler
    Unpickler = pickle.Unpickler
    dump = staticmethod(cloudpickle.dump)
    dumps = staticmethod(cloudpickle.dumps)
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def class_name(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def git_value(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalizer_details(normalizer: Any) -> Any:
    if normalizer is None:
        return None
    return {"class": class_name(normalizer), "config": json_ready(vars(normalizer))}


def group_contract(dataset: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for config in dataset.channels:
        entry = groups.setdefault(config.logical_name, {"units": set(), "normalizers": set()})
        entry["units"].add(config.unit)
        for physical_name in config.physical_names:
            normalizer = normalizer_details(config.normalizer_for(physical_name))
            entry["normalizers"].add(None if normalizer is None else json.dumps(normalizer, sort_keys=True))
    return groups


def assert_single_dataset_compatible(expected_dataset: Any, actual_dataset: Any, *, allow_preprocessing_override: bool) -> None:
    expected_inputs = list(expected_dataset.get_input_channels())
    actual_inputs = list(actual_dataset.get_input_channels())
    if actual_inputs != expected_inputs:
        raise ValueError(f"Expected logical input channels {expected_inputs}, got {actual_inputs}.")
    for attribute in ["sample_frequency", "resample_type", "total_input", "target_resolution", "stride", "z_normalize"]:
        expected = str(getattr(expected_dataset, attribute))
        actual = str(getattr(actual_dataset, attribute))
        if actual != expected:
            raise ValueError(f"Expected dataset {attribute}={expected}, got {actual}.")
    if allow_preprocessing_override:
        return
    expected_groups = group_contract(expected_dataset)
    actual_groups = group_contract(actual_dataset)
    for group in expected_inputs:
        expected = expected_groups[group]
        actual = actual_groups[group]
        if not actual["units"].issubset(expected["units"]):
            raise ValueError(f"Expected units {expected['units']} for '{group}', got {actual['units']}.")
        if not actual["normalizers"].issubset(expected["normalizers"]):
            raise ValueError(f"Normalizer configuration for '{group}' does not match the package's stored preprocessing.")


@dataclass
class PackagedModel:
    """A model together with its executable input and output contracts."""

    name: str
    model: torch.nn.Module
    dataset: Any
    classification_contract: dict[str, Any] | None = None
    task: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.model, torch.nn.Module) or not isinstance(self.model, (EmbeddingModel, ClassifierModel)):
            raise TypeError("Packaged models must implement EmbeddingModel, ClassifierModel, or both.")
        if self.classification_contract is not None and not isinstance(self.model, ClassifierModel):
            raise TypeError("classification_contract requires a ClassifierModel.")
        if isinstance(self.model, ClassifierModel) and self.classification_contract is None:
            raise ValueError("A packaged ClassifierModel requires classification_contract.")
        if self.classification_contract is not None and self.classification_contract.get("type") == "single-head-multiclass" and self.task is None:
            raise ValueError("A single-head classifier package requires task.")

        shape, meta = self.model.input_spec()
        if isinstance(shape, dict):
            if not hasattr(self.dataset, "input_spec"):
                raise TypeError("A mapping-input model requires a paired dataset.")
            expected_shape = self.dataset.input_spec()
            if shape != expected_shape:
                raise ValueError(f"Model inputs {shape} do not match packaged dataset inputs {expected_shape}.")
            if meta.get("layout") != "mapping":
                raise ValueError(f"Mapping-input models must declare layout='mapping', got {meta.get('layout')!r}.")
        else:
            expected_shape = (1, self.dataset.get_timeseries_len(), len(self.dataset.get_input_channels()))
            if tuple(shape) != expected_shape:
                raise ValueError(f"Model input {tuple(shape)} does not match packaged dataset input {expected_shape}.")
            if meta.get("layout") != "BTC":
                raise ValueError(f"Packaged models must use BTC inputs, got {meta.get('layout')!r}.")
            if "input_channels" in meta and list(meta["input_channels"]) != list(self.dataset.get_input_channels()):
                raise ValueError(f"Model expects input channels {list(meta['input_channels'])}, but the packaged dataset provides {list(self.dataset.get_input_channels())}.")
            if "sampling_frequency" in meta and float(meta["sampling_frequency"]) != float(self.dataset.sample_frequency):
                raise ValueError(f"Model expects sampling_frequency={meta['sampling_frequency']}, but the packaged dataset provides {self.dataset.sample_frequency}.")

    @property
    def capabilities(self) -> list[str]:
        capabilities = []
        if isinstance(self.model, EmbeddingModel):
            capabilities.append("embeddings")
        if isinstance(self.model, ClassifierModel):
            capabilities.append("classification")
        return capabilities

    def assert_compatible(self, dataset: Any, *, allow_preprocessing_override: bool = False) -> None:
        expected_inputs = self.dataset.get_input_channels()
        actual_inputs = dataset.get_input_channels()
        if isinstance(expected_inputs, dict):
            if not isinstance(actual_inputs, dict) or set(actual_inputs) != set(expected_inputs):
                raise ValueError(f"Expected paired dataset inputs {sorted(expected_inputs)}, got {actual_inputs}.")
            if self.dataset.target_resolution != dataset.target_resolution or self.dataset.stride != dataset.stride:
                raise ValueError("Paired datasets must use the same target_resolution and stride.")
            for name in expected_inputs:
                assert_single_dataset_compatible(self.dataset.datasets[name], dataset.datasets[name], allow_preprocessing_override=allow_preprocessing_override)
            return
        assert_single_dataset_compatible(self.dataset, dataset, allow_preprocessing_override=allow_preprocessing_override)

    def predict_dataset(self, dataset: Any, *, batch_size: int = 64, num_workers: int = 0, n_repeat: int = 1, device: str | torch.device = "cpu") -> pd.DataFrame:
        return predict_dataset(self, dataset, batch_size=batch_size, num_workers=num_workers, n_repeat=n_repeat, device=device)

    def predict_edf(self, edf_path: str | os.PathLike, *, batch_size: int = 64, num_workers_dataset: int = 0, num_workers_loader: int = 0, n_repeat: int = 1, device: str | torch.device = "cpu") -> pd.DataFrame:
        return predict_edf(self, edf_path, batch_size=batch_size, num_workers_dataset=num_workers_dataset, num_workers_loader=num_workers_loader, n_repeat=n_repeat, device=device)

    def save(self, path: str | os.PathLike) -> Path:
        root = Path(path)
        if root.exists() and not root.is_dir():
            raise ValueError(f"Package path must be a directory, got {root}.")
        root.mkdir(parents=True, exist_ok=True)
        payload_path = root / "model.pt"
        torch.save(self, payload_path, pickle_module=CloudpickleAdapter)
        manifest = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "task": self.task,
            "payload": payload_path.name,
            "sha256": sha256(payload_path),
            "model_class": class_name(self.model),
            "dataset_class": class_name(self.dataset),
            "capabilities": self.capabilities,
            "input_channels": self.dataset.get_input_channels(),
            "classification": self.classification_contract,
            "config": self.config,
            "git_commit": self.git_commit,
        }
        with (root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, cls=NumpyEncoder)
            handle.write("\n")
        return root

    @classmethod
    def load(cls, path: str | os.PathLike, *, map_location: str | torch.device = "cpu") -> "PackagedModel":
        root = Path(path)
        with (root / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported package format {manifest.get('format_version')!r}.")
        payload_path = root / manifest["payload"]
        actual_hash = sha256(payload_path)
        if actual_hash != manifest["sha256"]:
            raise ValueError(f"Package payload hash mismatch: expected {manifest['sha256']}, got {actual_hash}.")
        package = torch.load(payload_path, map_location=map_location, pickle_module=CloudpickleAdapter, weights_only=False)
        if not isinstance(package, cls):
            raise TypeError(f"Expected a PackagedModel payload, got {type(package).__name__}.")
        package.model = package.model.to(map_location)
        return package


def save_packaged_model(path: str | os.PathLike, *, name: str, model: torch.nn.Module, dataset: Any, classification_contract: dict[str, Any] | None = None, task: str | None = None, config: dict[str, Any] | None = None, git_commit: Optional[str] = None) -> PackagedModel:
    deployment_source = dataset.datasets[0] if isinstance(dataset, MultiDataset) else dataset
    if hasattr(deployment_source, "to_unlabelled"):
        deployment_dataset = deployment_source.to_unlabelled()
    else:
        deployment_dataset = deployment_source if isinstance(deployment_source, UnlabelledDataset) else UnlabelledDataset.from_dataset(deployment_source)
    package = PackagedModel(
        name=name,
        task=task,
        model=model,
        dataset=deployment_dataset,
        classification_contract=json_ready(classification_contract) if classification_contract is not None else None,
        config=json_ready(config or {}),
        git_commit=git_value("rev-parse", "HEAD") if git_commit is None else git_commit,
    )
    package.save(path)
    return package


def load_packaged_model(path: str | os.PathLike, *, map_location: str | torch.device = "cpu") -> PackagedModel:
    return PackagedModel.load(path, map_location=map_location)


def predict_dataset(package: PackagedModel, dataset: Any, *, batch_size: int = 64, num_workers: int = 0, n_repeat: int = 1, device: str | torch.device = "cpu") -> pd.DataFrame:
    """Run a packaged classifier on an initialized compatible dataset."""

    if package.classification_contract is None:
        raise TypeError(f"Package '{package.name}' has no classification contract.")
    package.assert_compatible(dataset)
    loader = build_loader(dataset, batch_size=batch_size, num_workers=num_workers, n_samples=None, collate_fn=batch_collate, sampling="sequential", seed=0, rejection_strategy="none", n_views=n_repeat)
    execution_model = (RepeatedViewModel(package.model) if n_repeat > 1 else package.model).to(device)
    execution_model.eval()
    frames = []
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            data = {key: value.to(device, non_blocking=True) for key, value in batch["data"].items()} if isinstance(batch["data"], dict) else batch["data"].to(device, non_blocking=True)
            outputs = execution_model(data)
            frames.append(format_prediction_batch(package.classification_contract, batch, outputs, dataset.target_resolution))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_edf_dataset(package: PackagedModel, edf_path: str | os.PathLike, *, num_workers: int = 0):
    """Build the package's exact stored dataset contract for one EDF file."""

    dataset = package.dataset.clone()
    package.assert_compatible(dataset)
    dataset.initialize([str(edf_path)], num_workers=num_workers, strict=True)
    if dataset.get_n_patients() != 1:
        raise ValueError(f"Could not prepare EDF file {edf_path}.")
    return dataset


def predict_edf(package: PackagedModel, edf_path: str | os.PathLike, *, batch_size: int = 64, num_workers_dataset: int = 0, num_workers_loader: int = 0, n_repeat: int = 1, device: str | torch.device = "cpu") -> pd.DataFrame:
    """Run a packaged classifier on one complete EDF file."""

    dataset = build_edf_dataset(package, edf_path, num_workers=num_workers_dataset)
    return predict_dataset(package, dataset, batch_size=batch_size, num_workers=num_workers_loader, n_repeat=n_repeat, device=device)
