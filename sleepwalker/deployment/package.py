"""Expert package save/load path for repo-local research handoff.

An expert package is a directory-based snapshot containing two kinds of data:

1. Binary training artifacts:
   model parameters and, optionally, optimizer/scheduler state.
2. A JSON manifest:
   a structured explanation of what the snapshot expects at inference/load
   time and how to reconstruct the supporting Python objects.

The design intentionally favors explicitness over full portability. A package
is meant to be understandable and reusable inside a compatible Sleepwalker
checkout, not a standalone model interchange format. In particular:

- The package stores tensor state, not source code.
- Reload uses a builder function from the repository to reconstruct the model,
  trainer, and dataset template.
- The manifest records input/output and preprocessing contracts so callers can
  validate whether a dataset or batch matches the saved expert.

This module owns the filesystem layout and torch serialization mechanics,
whereas :mod:`sleepwalker.deployment.manifest` owns the manifest schema. The
modules remain separate because the schema is useful as a pure contract layer
without dragging in package I/O code, but together they define one deployment
story.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Optional

import torch

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.deployment.manifest import BuilderSpec, ExpertManifest, serialize_callable


def _safe_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _normalize_path(path: str | os.PathLike) -> Path:
    out = Path(path)
    if out.exists() and out.is_file():
        raise ValueError(f"Expected expert package directory path, got existing file: {out}")
    return out


def _load_builder(spec: BuilderSpec):
    module = importlib.import_module(spec.module)
    func = getattr(module, spec.function)
    return func


def _class_path(obj: Any) -> str:
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def _dataset_side_contract(dataset: Any) -> dict[str, Any]:
    return {
        "resample_type": getattr(dataset, "resample_type", None),
        "rereference": getattr(dataset, "rereference", None),
        "prepare_patient": serialize_callable(getattr(dataset, "prepare_patient_callback", None)),
        "prepare_target": serialize_callable(getattr(dataset, "prepare_target_callback", None)),
        "prepare_sample": serialize_callable(getattr(dataset, "prepare_sample_callback", None)),
    }


def _model_side_contract(model: Any) -> list[dict[str, Any]]:
    preprocessors = []
    for preprocessor in getattr(model, "preprocessors", []):
        preprocessors.append(
            {
                "class_name": _class_path(preprocessor),
            }
        )
    return preprocessors


def _input_contract(model: Any, dataset_template: Any) -> dict[str, Any]:
    shape, meta = model.input_spec()
    return {
        "tensor_layout": meta.get("layout", "BTC"),
        "channels": list(dataset_template.get_input_channels()) if hasattr(dataset_template, "get_input_channels") else [],
        "sample_frequency": float(getattr(dataset_template, "sample_frequency")),
        "total_input": str(getattr(dataset_template, "total_input")),
        "target_resolution": str(getattr(dataset_template, "target_resolution", getattr(dataset_template, "total_input"))),
        "stride": str(getattr(dataset_template, "stride", getattr(dataset_template, "target_resolution", getattr(dataset_template, "total_input")))),
        "ts_len": int(shape[1]),
        "n_channels": int(shape[2]),
    }


def _output_contract(model: Any, trainer: Any) -> dict[str, Any]:
    if hasattr(trainer, "task_config"):
        return {
            "type": "multitask",
            "keys": list(trainer.task_config.keys()),
            "task_config": {
                task: {
                    "labels": list(cfg["labels"]),
                    "n_steps": int(cfg["n_steps"]),
                    "target_resolution": str(cfg["target_resolution"]),
                }
                for task, cfg in trainer.task_config.items()
            },
        }
    return {
        "type": "single-head-multiclass",
        "keys": ["logits"],
        "classes": list(getattr(trainer, "classes", [])),
    }


@dataclass
class LoadedExpert:
    """Live objects reconstructed from an expert package.

    ``LoadedExpert`` is the in-memory view of a package directory after
    rehydration. It provides the rebuilt model/trainer/template objects plus
    the parsed manifest and any optional optimizer/scheduler state.

    What callers can expect:

    - ``model`` is loaded with persisted weights.
    - ``trainer`` and ``dataset_template`` are rebuilt through the package
      builder when available.
    - ``manifest`` remains the primary source of structural expectations and
      metadata.

    What callers should not expect:

    - The object is not a generic training session checkpoint. Only the pieces
      explicitly persisted into the package are available.
    - Arbitrary runtime state outside the saved tensor files and manifest is
      not restored.
    """

    model: Any
    trainer: Any
    dataset_template: Any
    manifest: ExpertManifest
    optimizer_state: Optional[dict[str, Any]] = None
    scheduler_state: Optional[dict[str, Any]] = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self.manifest.metadata

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, batch_or_tensor):
        if isinstance(batch_or_tensor, dict):
            self.validate_batch(batch_or_tensor)
            x = batch_or_tensor["data"]
        else:
            x = batch_or_tensor
        return self.model(x)

    def validate_batch(self, batch: dict[str, Any]) -> None:
        if "data" not in batch:
            raise ValueError("Batch must contain 'data'.")
        x = batch["data"]
        expected = self.manifest.input_contract
        if x.ndim != 3:
            raise ValueError(f"Expected BTC tensor with ndim=3, got shape {tuple(x.shape)}.")
        if int(x.shape[1]) != int(expected["ts_len"]):
            raise ValueError(f"Expected time axis {expected['ts_len']}, got {x.shape[1]}.")
        if int(x.shape[2]) != int(expected["n_channels"]):
            raise ValueError(f"Expected channel axis {expected['n_channels']}, got {x.shape[2]}.")

    def validate_dataset(self, dataset: Any) -> None:
        expected = self.manifest.input_contract
        channels = list(dataset.get_input_channels()) if hasattr(dataset, "get_input_channels") else []
        if channels != list(expected["channels"]):
            raise ValueError(f"Expected channels {expected['channels']}, got {channels}.")
        if float(getattr(dataset, "sample_frequency")) != float(expected["sample_frequency"]):
            raise ValueError(
                f"Expected sample_frequency={expected['sample_frequency']}, got {getattr(dataset, 'sample_frequency')}."
            )
        if str(getattr(dataset, "total_input")) != str(expected["total_input"]):
            raise ValueError(f"Expected total_input={expected['total_input']}, got {getattr(dataset, 'total_input')}.")

    def build_optimizer(self):
        if self.trainer is None or not hasattr(self.trainer, "optimizer_fn"):
            raise ValueError("Loaded expert does not expose trainer.optimizer_fn.")
        optimizer = self.trainer.optimizer_fn(self.model)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        return optimizer

    def build_scheduler(self, optimizer):
        if self.trainer is None or not hasattr(self.trainer, "lr_scheduler_fn") or self.trainer.lr_scheduler_fn is None:
            return None
        scheduler = self.trainer.lr_scheduler_fn(optimizer)
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)
        return scheduler

    def predict_window(self, batch, n_repeat: int = 1):
        if self.trainer is None:
            raise ValueError("Loaded expert does not include a trainer.")
        return self.trainer.predict_window(self.model, batch, n_repeat=n_repeat)

    def predict_loader(self, loader):
        if self.trainer is None:
            raise ValueError("Loaded expert does not include a trainer.")
        return self.trainer.predict_loader(self.model, loader)

    def predict_patient(
        self,
        edf_path,
        batch_size: int,
        num_workers_dataset: int = 1,
        num_workers_loader: int = 0,
        collate_fn=None,
    ):
        if self.trainer is None or self.dataset_template is None:
            raise ValueError("Loaded expert does not include trainer and dataset_template.")
        return self.trainer.predict_patient(
            self.model,
            self.dataset_template,
            edf_path,
            batch_size=batch_size,
            num_workers_dataset=num_workers_dataset,
            num_workers_loader=num_workers_loader,
            collate_fn=batch_collate if collate_fn is None else collate_fn,
        )


def save_expert_package(
    path: str | os.PathLike,
    *,
    expert_name: str,
    task: str,
    model,
    trainer,
    dataset_template,
    builder: Optional[dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> LoadedExpert:
    """Write a directory-based expert package and return its live view.

    The resulting directory contains:

    - ``model_state.pt``: required model parameters.
    - ``optimizer_state.pt``: optional optimizer snapshot.
    - ``scheduler_state.pt``: optional scheduler snapshot.
    - ``manifest.json``: structured contract and provenance description.

    The layout is intentionally small and inspectable. Users should expect a
    package to preserve enough information to reload the model in the same
    codebase and validate inference compatibility, but not to encapsulate the
    full repository or arbitrary Python behavior.
    """

    out = _normalize_path(path)
    out.mkdir(parents=True, exist_ok=True)

    builder_spec = None if builder is None else BuilderSpec(
        module=str(builder["module"]),
        function=str(builder["function"]),
        config=dict(builder.get("config", {})),
    )

    manifest = ExpertManifest(
        format_version="sleepwalker-expert-v1",
        expert_name=expert_name,
        task=task,
        source_git_commit=_safe_git_commit(),
        builder=builder_spec,
        model={
            "class_name": _class_path(model),
            "state_file": "model_state.pt",
            "input_spec": {
                "shape": list(model.input_spec()[0]),
                "meta": model.input_spec()[1],
            },
        },
        input_contract=_input_contract(model, dataset_template),
        preprocessing_contract={
            "dataset_side": _dataset_side_contract(dataset_template),
            "model_side": _model_side_contract(model),
        },
        output_contract=_output_contract(model, trainer),
        training_state={
            "optimizer_file": "optimizer_state.pt" if optimizer is not None else None,
            "scheduler_file": "scheduler_state.pt" if scheduler is not None else None,
            "trainer_class": _class_path(trainer) if trainer is not None else None,
            "dataset_class": _class_path(dataset_template) if dataset_template is not None else None,
        },
        metadata=dict(metadata or {}),
    )

    torch.save(model.state_dict(), out / "model_state.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out / "optimizer_state.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), out / "scheduler_state.pt")
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        f.write(manifest.to_json() + "\n")

    return LoadedExpert(
        model=model,
        trainer=trainer,
        dataset_template=dataset_template,
        manifest=manifest,
        optimizer_state=None if optimizer is None else optimizer.state_dict(),
        scheduler_state=None if scheduler is None else scheduler.state_dict(),
    )


def _manifest_from_dict(payload: dict[str, Any]) -> ExpertManifest:
    builder_payload = payload.get("builder")
    builder = None
    if builder_payload is not None:
        builder = BuilderSpec(
            module=builder_payload["module"],
            function=builder_payload["function"],
            config=dict(builder_payload.get("config", {})),
        )
    return ExpertManifest(
        format_version=payload["format_version"],
        expert_name=payload["expert_name"],
        task=payload["task"],
        source_git_commit=payload.get("source_git_commit"),
        builder=builder,
        model=dict(payload["model"]),
        input_contract=dict(payload["input_contract"]),
        preprocessing_contract=dict(payload["preprocessing_contract"]),
        output_contract=dict(payload["output_contract"]),
        training_state=dict(payload["training_state"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _prepare_preprocessor_state_for_load(model: Any, state_dict: dict[str, Any]) -> None:
    """Initialize lazy preprocessor buffers before loading saved state."""
    preprocessors = getattr(model, "preprocessors", None)
    if preprocessors is None:
        return

    pattern = re.compile(r"^preprocessors\.(\d+)\.n$")
    for key, saved in state_dict.items():
        match = pattern.match(key)
        if match is None or not hasattr(saved, "numel") or int(saved.numel()) == 0:
            continue
        idx = int(match.group(1))
        if idx >= len(preprocessors):
            continue
        preprocessor = preprocessors[idx]
        current = getattr(preprocessor, "n", None)
        if current is None or not hasattr(current, "numel") or int(current.numel()) != 0:
            continue
        if not hasattr(preprocessor, "push"):
            continue
        dummy = torch.zeros(1, 1, int(saved.numel()), device=saved.device, dtype=torch.float32)
        preprocessor.push(dummy)


def load_expert_package(
    path: str | os.PathLike,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedExpert:
    """Load a previously saved expert package from disk.

    Loading proceeds in three phases:

    1. Parse ``manifest.json`` to recover the schema and builder spec.
    2. Reconstruct Python objects by calling the builder.
    3. Load tensor state files into the reconstructed objects.

    Because builder reconstruction depends on repository code, load-time
    compatibility is a code-level contract, not just a file-format contract.
    """

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Expert package path must be a directory, got {root}")

    with (root / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = _manifest_from_dict(json.load(f))

    if manifest.builder is None:
        raise ValueError("Cannot load expert package without a builder spec.")

    builder_fn = _load_builder(manifest.builder)
    components = builder_fn(manifest.builder.config)

    model = components["model"]
    trainer = components.get("trainer")
    dataset_template = components.get("dataset_template")

    state_dict = torch.load(root / "model_state.pt", map_location=map_location)
    _prepare_preprocessor_state_for_load(model, state_dict)
    model.load_state_dict(state_dict)
    model = model.to(map_location)

    optimizer_state = None
    scheduler_state = None
    optimizer_file = manifest.training_state.get("optimizer_file")
    scheduler_file = manifest.training_state.get("scheduler_file")
    if optimizer_file:
        optimizer_state = torch.load(root / optimizer_file, map_location="cpu")
    if scheduler_file:
        scheduler_state = torch.load(root / scheduler_file, map_location="cpu")

    if trainer is not None and hasattr(trainer, "device"):
        trainer.device = str(map_location)
    if trainer is not None and hasattr(trainer, "warmup_device"):
        trainer.warmup_device = str(map_location)

    return LoadedExpert(
        model=model,
        trainer=trainer,
        dataset_template=dataset_template,
        manifest=manifest,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )


def export_prediction_package(
    path: str | os.PathLike,
    *,
    model,
    trainer,
    dataset,
    metadata: Optional[dict[str, Any]] = None,
    model_card_md: str = "",
) -> LoadedExpert:
    """Compatibility wrapper that saves an inference-oriented expert package.

    This helper preserves the older "prediction package" naming while using the
    same package/manifest layout as :func:`save_expert_package`.
    """

    del model_card_md
    metadata = dict(metadata or {})
    expert_name = str(
        metadata.get("expert_name")
        or metadata.get("experiment_name")
        or Path(path).name
    )
    task = str(
        metadata.get("task")
        or metadata.get("model")
        or "expert"
    )
    return save_expert_package(
        path,
        expert_name=expert_name,
        task=task,
        model=model,
        trainer=trainer,
        dataset_template=dataset,
        metadata=metadata,
    )


def load_prediction_package(path: str | os.PathLike, map_location: str | torch.device = "cpu") -> LoadedExpert:
    """Compatibility wrapper around :func:`load_expert_package`."""
    return load_expert_package(path, map_location=map_location)
