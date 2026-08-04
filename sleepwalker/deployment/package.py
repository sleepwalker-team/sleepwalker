"""Reloadable expert artifacts for a compatible Sleepwalker checkout.

An :class:`Expert` stores the trained model, its executable unlabelled dataset,
and the trainer that turns model outputs into timestamped probabilities.  The
original run configuration and Git commit document how the expert was trained.
The artifact is intentionally a trusted research artifact, not a portable model
interchange format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any, Optional, Sequence

import cloudpickle
import torch
from torch.utils.data import DataLoader

from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.trainer.utils.disk import NumpyEncoder, json_ready


FORMAT_VERSION = "sleepwalker-expert-v5"


class CloudpickleAdapter:
    """Give PyTorch cloudpickle's pickler and the standard compatible loader."""

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
            entry["normalizers"].add(
                None if normalizer is None else json.dumps(normalizer, sort_keys=True)
            )
    return groups


@dataclass
class Expert:
    """A trained specialist and everything required to run it on an EDF."""

    name: str
    task: str
    model: Any
    dataset: UnlabelledDataset
    trainer: Any
    config: dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None

    @property
    def output_contract(self) -> dict[str, Any]:
        if hasattr(self.trainer, "task_config"):
            return {
                "type": "multitask",
                "tasks": {
                    task: {
                        "classes": list(config["labels"]),
                        "n_steps": int(config["n_steps"]),
                        "target_resolution": str(config["target_resolution"]),
                    }
                    for task, config in self.trainer.task_config.items()
                },
            }
        return {"type": "single-head-multiclass", "classes": list(self.trainer.classes)}

    def freeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def forward(self, batch_or_tensor):
        value = batch_or_tensor["data"] if isinstance(batch_or_tensor, dict) else batch_or_tensor
        return self.model(value)

    def assert_compatible(self, dataset: Any) -> None:
        expected_inputs = list(self.dataset.get_input_channels())
        actual_inputs = list(dataset.get_input_channels())
        if actual_inputs != expected_inputs:
            raise ValueError(f"Expected logical input channels {expected_inputs}, got {actual_inputs}.")
        for attribute in ["sample_frequency", "resample_type", "total_input", "target_resolution", "stride"]:
            expected = str(getattr(self.dataset, attribute))
            actual = str(getattr(dataset, attribute))
            if actual != expected:
                raise ValueError(f"Expected dataset {attribute}={expected}, got {actual}.")
        expected_groups = group_contract(self.dataset)
        actual_groups = group_contract(dataset)
        for group in expected_inputs:
            expected = expected_groups[group]
            actual = actual_groups[group]
            if not actual["units"].issubset(expected["units"]):
                raise ValueError(f"Expected units {expected['units']} for '{group}', got {actual['units']}.")
            if not actual["normalizers"].issubset(expected["normalizers"]):
                raise ValueError(f"Normalizer configuration for '{group}' does not match the expert's stored preprocessing.")

    def predict_dataset(self, dataset: Any, *, batch_size: int = 64, num_workers: int = 0, collate_fn=None):
        self.assert_compatible(dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=batch_collate if collate_fn is None else collate_fn,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
        return self.trainer.predict_loader(self.model, loader)

    def predict_edf(
        self,
        edf_path: str | os.PathLike,
        *,
        channels: Optional[Sequence[ChannelConfig]] = None,
        assume_units_if_missing: Optional[bool] = None,
        batch_size: int = 64,
        num_workers_dataset: int = 0,
        num_workers_loader: int = 0,
        collate_fn=None,
    ):
        dataset = self.dataset.clone(channels=channels, assume_units_if_missing=assume_units_if_missing)
        self.assert_compatible(dataset)
        dataset.initialize([str(edf_path)], num_workers=num_workers_dataset, strict=True)
        if dataset.get_n_patients() != 1:
            raise ValueError(f"Could not prepare EDF file {edf_path}.")
        return self.predict_dataset(dataset, batch_size=batch_size, num_workers=num_workers_loader, collate_fn=collate_fn)

    def save(self, path: str | os.PathLike) -> Path:
        root = Path(path)
        if root.exists() and not root.is_dir():
            raise ValueError(f"Expert path must be a directory, got {root}.")
        root.mkdir(parents=True, exist_ok=True)

        payload_path = root / "expert.pt"
        torch.save(self, payload_path, pickle_module=CloudpickleAdapter)
        manifest = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "task": self.task,
            "payload": payload_path.name,
            "sha256": sha256(payload_path),
            "model_class": class_name(self.model),
            "dataset_class": class_name(self.dataset),
            "trainer_class": class_name(self.trainer),
            "input_channels": list(self.dataset.get_input_channels()),
            "output": self.output_contract,
            "config": self.config,
            "git_commit": self.git_commit,
        }
        with (root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, cls=NumpyEncoder)
            handle.write("\n")
        return root

    @classmethod
    def load(cls, path: str | os.PathLike, *, map_location: str | torch.device = "cpu") -> "Expert":
        root = Path(path)
        with (root / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported expert format {manifest.get('format_version')!r}.")
        payload_path = root / manifest["payload"]
        actual_hash = sha256(payload_path)
        if actual_hash != manifest["sha256"]:
            raise ValueError(f"Expert payload hash mismatch: expected {manifest['sha256']}, got {actual_hash}.")
        expert = torch.load(payload_path, map_location=map_location, pickle_module=CloudpickleAdapter, weights_only=False)
        if not isinstance(expert, cls):
            raise TypeError(f"Expected an Expert payload, got {type(expert).__name__}.")
        expert.model = expert.model.to(map_location)
        if hasattr(expert.trainer, "device"):
            expert.trainer.device = str(map_location)
        if hasattr(expert.trainer, "warmup_device"):
            expert.trainer.warmup_device = str(map_location)
        return expert


def save_expert_package(
    path: str | os.PathLike,
    *,
    expert_name: str,
    task: str,
    model: Any,
    trainer: Any,
    dataset: Any,
    config: dict[str, Any],
    git_commit: Optional[str] = None,
) -> Expert:
    inference_dataset = dataset if isinstance(dataset, UnlabelledDataset) else UnlabelledDataset.from_dataset(dataset)
    expert = Expert(
        name=expert_name,
        task=task,
        model=model,
        dataset=inference_dataset,
        trainer=trainer,
        config=json_ready(config),
        git_commit=git_value("rev-parse", "HEAD") if git_commit is None else git_commit,
    )
    expert.save(path)
    return expert


def load_expert_package(path: str | os.PathLike, *, map_location: str | torch.device = "cpu") -> Expert:
    return Expert.load(path, map_location=map_location)
