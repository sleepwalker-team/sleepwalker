"""Fit one batch classifier on frozen waveform features."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sleepwalker.models.BatchClassifierModel import BatchClassifierModel
from sleepwalker.trainer.utils.disk import CloudpickleAdapter, load_checkpoint
from sleepwalker.training.loader import build_warmup_loader
from sleepwalker.utils import logger


def hard_class_targets(batch, classes):
    """Accept one hard one-hot target per window; masks remove whole rows."""
    target = torch.as_tensor(batch["target"])
    if target.ndim == 3 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 2 or target.shape[1] != len(classes):
        raise ValueError("Batch classification requires hard one-hot [B,C] or [B,1,C] targets; temporal/soft targets need an explicit adapter.")
    mask = torch.as_tensor(batch.get("target_mask", target.sum(dim=1) != 0), dtype=torch.bool).reshape(-1)
    if mask.shape != (len(target),):
        raise ValueError("target_mask must contain one validity flag per input row.")
    selected = target[mask]
    if not torch.isfinite(selected).all() or not ((selected == 0) | (selected == 1)).all() or not (selected.sum(dim=1) == 1).all():
        raise ValueError("Batch classification accepts only hard one-hot targets in the declared class order.")
    if "sample_weight" in batch or "weight" in batch:
        raise ValueError("This batch-classification target policy does not accept sample weights; supply an explicit weighted backend/target adapter.")
    return mask.cpu(), selected.argmax(dim=1).cpu().numpy().astype(np.int64)


class BatchTrainer:
    """Extract aligned features/targets, then fit the estimator once.

    RAM arrays are the default. Disk mode writes float32 features and int64
    targets directly to feature_path and reads them back through memmaps.
    Each extraction overwrites its files; an interrupted fit restarts extraction.
    A checkpoint retains the encoder and estimator, never the feature arrays.
    """

    def __init__(self, classes, target_resolution, target_offset="0s", store_on_disk=False, feature_path=None, prepare_features="reuse", device="cpu", checkpoint_path=None):
        self.classes = list(classes)
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must contain at least two unique labels.")
        self.target_resolution = pd.Timedelta(target_resolution)
        self.target_offset = pd.Timedelta(target_offset)
        if self.target_resolution <= pd.Timedelta(0) or self.target_offset < pd.Timedelta(0):
            raise ValueError("Target resolution must be positive and offset nonnegative.")
        if prepare_features not in {"reuse", "fit"}:
            raise ValueError("prepare_features must be 'reuse' or 'fit'.")
        if bool(store_on_disk) != (feature_path is not None):
            raise ValueError("Disk extraction requires store_on_disk=True and feature_path.")
        self.store_on_disk = bool(store_on_disk)
        self.feature_path = feature_path
        self.prepare_features = prepare_features
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.prepared = False
        self.extraction_info = {}

    def classification_contract(self):
        return {"type": "single-head-multiclass", "classes": list(self.classes), "sequence_len": 1, "target_resolution": str(self.target_resolution), "target_offset": str(self.target_offset)}

    @torch.no_grad()
    def extract(self, model, loader, split):
        """Keep only valid rows, in loader order, using RAM or two raw files."""
        dimension = model.feature_dim()
        capacity = len(loader.sampler) if hasattr(loader, "sampler") else None
        path = Path(self.feature_path) if self.store_on_disk else None
        X = y = None
        chunks = []
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            for name in ["features", "targets"]:
                (path / f"{split}-{name}.bin").write_bytes(b"")
        elif capacity is not None:
            X, y = np.empty((capacity, dimension), np.float32), np.empty(capacity, np.int64)
        rows = batches = 0
        for batch in loader:
            batches += 1
            if batch is None:
                continue
            mask, targets = hard_class_targets(batch, self.classes)
            if len(mask) != len(batch["data"]):
                raise ValueError("Feature/target batch row counts differ.")
            if not len(targets):
                continue
            data = batch["data"][mask.to(batch["data"].device)].to(self.device)
            values = model.features(data)
            if not isinstance(values, torch.Tensor) or values.shape != (len(targets), dimension) or not torch.isfinite(values).all():
                raise ValueError(f"Expected finite features with shape {(len(targets), dimension)}.")
            features = values.detach().cpu().to(torch.float32).numpy()
            end = rows + len(targets)
            if path is not None:
                for name, array in [("features", features), ("targets", targets)]:
                    with (path / f"{split}-{name}.bin").open("ab") as handle:
                        array.tofile(handle)
            elif X is not None:
                X[rows:end], y[rows:end] = features, targets
            else:
                chunks.append((features, targets))
            rows = end
        if not rows:
            raise ValueError("The feature loader yielded no usable target rows.")
        self.extraction_info[split] = {"rows": rows, "batches": batches, "dimension": dimension}
        logger.info(f"{split}: {rows:,} rows x {dimension:,} float32 features; estimator memory is additional.")
        if path is not None:
            return (np.memmap(path / f"{split}-features.bin", mode="r", dtype=np.float32, shape=(rows, dimension)), np.memmap(path / f"{split}-targets.bin", mode="r", dtype=np.int64, shape=(rows,)))
        if X is not None:
            return X[:rows], y[:rows]
        return tuple(np.concatenate([chunk[index] for chunk in chunks]) for index in [0, 1])

    def fit(self, model, train_loader, val_loader=None):
        if not isinstance(model, BatchClassifierModel) or model.classes != self.classes:
            raise TypeError("BatchTrainer requires a BatchClassifierModel with matching classes.")
        if not model.batch_model.is_fitted():
            encoder = model.embedding_model.eval().to(self.device)
            encoder.requires_grad_(False)
            if not self.prepared and self.prepare_features == "fit":
                warmup_loader = build_warmup_loader(train_loader)
                if hasattr(encoder, "warmup_preprocessor"):
                    encoder.warmup_preprocessor(warmup_loader, self.device)
                else:
                    for index, step in enumerate(encoder.preprocessors):
                        if hasattr(step, "requires_warmup") and step.requires_warmup():
                            for batch in warmup_loader:
                                if batch is not None:
                                    step.update(encoder.apply_preprocessors(batch["data"].to(self.device), index))
            self.prepared = True
            if self.checkpoint_path is not None:
                self.save_checkpoint(self.checkpoint_path, model)
            X, y = self.extract(model, train_loader, "train")
            validation_data = self.extract(model, val_loader, "validation") if val_loader is not None else None
            model.batch_model.fit(X, y, validation_data=validation_data)
            model.validate_fitted()
            if self.checkpoint_path is not None:
                self.save_checkpoint(self.checkpoint_path, model)
        self.prepared = True
        return {"losses": [], "outputs": [], "feature_extraction": self.extraction_info}

    def validate_training_state(self, model):
        if not self.prepared or not isinstance(model, BatchClassifierModel) or model.classes != self.classes:
            raise RuntimeError("Batch checkpoint requires a prepared encoder and matching composite model.")

    def save_checkpoint(self, path, model):
        self.validate_training_state(model)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"trainer": self, "model": model}, temporary, pickle_module=CloudpickleAdapter)
        temporary.replace(path)
        return path

    @classmethod
    def load_checkpoint(cls, path, map_location=None):
        trainer, model = load_checkpoint(path, map_location)
        if not isinstance(trainer, cls):
            raise TypeError(f"Expected a {cls.__name__} checkpoint.")
        return trainer, model

    @torch.no_grad()
    def test(self, model, loader):
        model.to(self.device).eval()
        matrix = np.zeros((len(self.classes), len(self.classes)), dtype=np.int64)
        loss, count = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            mask, targets = hard_class_targets(batch, self.classes)
            if not len(targets):
                continue
            logits = model(batch["data"][mask.to(batch["data"].device)].to(self.device))[:, 0]
            y = torch.as_tensor(targets, device=self.device)
            loss += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            np.add.at(matrix, (targets, logits.argmax(dim=1).cpu().numpy()), 1)
            count += len(targets)
        if count == 0:
            raise ValueError("The test loader yielded no usable rows.")
        return loss / count, matrix
