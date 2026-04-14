from __future__ import annotations

import copy
import io
import json
import os
import platform
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import cloudpickle
import numpy as np
import pandas as pd
import torch

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset


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


def _normalize_device_string(device: str | torch.device) -> str:
    return str(device)


def _torch_save_bytes(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _sanitize_trainer_for_export(trainer):
    trainer_copy = copy.copy(trainer)
    if hasattr(trainer_copy, "device"):
        trainer_copy.device = "cpu"
    if hasattr(trainer_copy, "warmup_device"):
        trainer_copy.warmup_device = "cpu"
    for attr in [
        "optimizer_fn",
        "lr_scheduler_fn",
        "loss_function",
        "base_loss_function",
        "train_transform",
        "task_loss_functions",
        "domain_head",
    ]:
        if hasattr(trainer_copy, attr):
            value = {} if attr == "task_loss_functions" else None
            setattr(trainer_copy, attr, value)
    return trainer_copy


def _model_copy_for_export(model):
    return copy.deepcopy(model).to("cpu")


def _normalize_export_path(path: str) -> str:
    normalized = os.fspath(path)
    if normalized.endswith(os.sep):
        raise ValueError("Prediction package export path must be a file path, not a directory path.")
    root, ext = os.path.splitext(normalized)
    if ext != ".swmodel":
        return normalized + ".swmodel"
    else:
        return normalized

def _to_unlabelled_dataset(dataset):
    if isinstance(dataset, UnlabelledDataset):
        return dataset.clone()
    if hasattr(dataset, "to_unlabelled"):
        return dataset.to_unlabelled()
    raise ValueError(
        f"Dataset of type {type(dataset).__name__} does not support export to an UnlabelledDataset."
    )


@dataclass
class PredictionPackage:
    model: Any
    trainer: Any
    unlabelled_dataset: Any
    metadata: dict[str, Any]

    def predict_window(self, batch, n_repeat: int = 1) -> pd.DataFrame:
        return self.trainer.predict_window(self.model, batch, n_repeat=n_repeat)

    def predict_loader(self, loader) -> pd.DataFrame:
        return self.trainer.predict_loader(self.model, loader)

    def predict_patient(
        self,
        edf_path,
        batch_size: int,
        num_workers_dataset: int = 1,
        num_workers_loader: int = 0,
        collate_fn=None,
    ) -> pd.DataFrame:
        return self.trainer.predict_patient(
            self.model,
            self.unlabelled_dataset,
            edf_path,
            batch_size=batch_size,
            num_workers_dataset=num_workers_dataset,
            num_workers_loader=num_workers_loader,
            collate_fn=batch_collate if collate_fn is None else collate_fn,
        )


def export_prediction_package(
    path: str,
    *,
    model,
    trainer,
    dataset,
    model_card_md: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> PredictionPackage:
    path = _normalize_export_path(path)
    unlabelled_dataset = _to_unlabelled_dataset(dataset)
    model_export = _model_copy_for_export(model)
    trainer_export = _sanitize_trainer_for_export(trainer)
    package_metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _safe_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "trainer_class": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}",
        "source_dataset_class": f"{dataset.__class__.__module__}.{dataset.__class__.__name__}",
        "unlabelled_dataset_class": f"{unlabelled_dataset.__class__.__module__}.{unlabelled_dataset.__class__.__name__}",
        "source_event_mapping": dict(getattr(dataset, "event_mapping", {}) or {}),
    }
    if metadata is not None:
        package_metadata.update(metadata)
    package_metadata["format"] = "sleepwalker-swmodel-v1"

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(package_metadata, indent=2, ensure_ascii=True))
        zf.writestr("model_card.md", model_card_md)
        zf.writestr("model.pt", _torch_save_bytes(model.state_dict()))
        zf.writestr("model.pkl", cloudpickle.dumps(model_export))
        zf.writestr("trainer.pkl", cloudpickle.dumps(trainer_export))
        zf.writestr("unlabelled_dataset.pkl", cloudpickle.dumps(unlabelled_dataset))

    return PredictionPackage(
        model=model,
        trainer=trainer,
        unlabelled_dataset=unlabelled_dataset,
        metadata=package_metadata,
    )


def load_prediction_package(path: str, map_location: str | torch.device = "cpu") -> PredictionPackage:
    if os.path.isdir(path):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            model = cloudpickle.load(f)
        with open(os.path.join(path, "trainer.pkl"), "rb") as f:
            trainer = cloudpickle.load(f)
        with open(os.path.join(path, "unlabelled_dataset.pkl"), "rb") as f:
            unlabelled_dataset = cloudpickle.load(f)
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=map_location)
    else:
        with zipfile.ZipFile(path, "r") as zf:
            model = cloudpickle.loads(zf.read("model.pkl"))
            trainer = cloudpickle.loads(zf.read("trainer.pkl"))
            unlabelled_dataset = cloudpickle.loads(zf.read("unlabelled_dataset.pkl"))
            metadata = json.loads(zf.read("meta.json").decode("utf-8"))
            state_dict = torch.load(io.BytesIO(zf.read("model.pt")), map_location=map_location)
    model.load_state_dict(state_dict)
    model = model.to(map_location)
    device_str = _normalize_device_string(map_location)
    if hasattr(trainer, "device"):
        trainer.device = device_str
    if hasattr(trainer, "warmup_device"):
        trainer.warmup_device = device_str
    return PredictionPackage(
        model=model,
        trainer=trainer,
        unlabelled_dataset=unlabelled_dataset,
        metadata=metadata,
    )
