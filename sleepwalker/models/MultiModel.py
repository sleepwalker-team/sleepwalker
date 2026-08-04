"""Composite models that fuse embeddings from several submodels.

This module provides the model-composition layer used by several task scripts
and by ``MetaModel``. Tests in ``tests/test_metamodel.py`` confirm channel
slicing, embedding fusion, and nested preprocessor warmup behavior.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.utils import logger


@dataclass
class MetaModelEntry:
    """Describe one submodel together with the channels it consumes.

    Attributes:
        model: Submodel instance implementing the ``BaseModel`` interface.
        input_channels: Channel names that should be sliced out of the parent
            input tensor before being passed to ``model``.
        trainable: Whether the submodel participates in optimization. Setting
            this to ``False`` freezes its existing parameters when the entry is
            constructed.
    """

    model: BaseModel
    input_channels: list[str]
    trainable: bool = True

    def __post_init__(self):
        if not self.trainable:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)


class MultiModel(BaseModel):
    """Fuse embeddings from several submodels and apply one shared head.

    Args:
        classes: Output label names for the shared classification head.
        input_channels: Global input channel order expected by the composite
            model. When omitted, the order is inferred from the submodel
            entries.
        models: Submodel entries describing channel routing.
        preprocessors: Optional preprocessors applied before channel slicing.

    Notes:
        Each submodel must expose ``features(x)`` and ``feature_dim()``. The
        composite model concatenates the resulting embeddings and feeds them
        into a single linear head.
    """

    def __init__(
        self,
        *,
        classes: list[str],
        input_channels: Optional[list[str]] = None,
        models: list[MetaModelEntry],
        preprocessors: Optional[list] = None,
    ):
        super().__init__(preprocessors=preprocessors)

        self.classes = list(classes)
        if len(self.classes) == 0:
            raise ValueError("MultiModel requires at least one class.")

        self.input_channels = [] if input_channels is None else list(input_channels)
        if input_channels is None:
            for entry in models:
                for channel in entry.input_channels:
                    if channel not in self.input_channels:
                        self.input_channels.append(channel)

        self.model_entries = []
        feature_sizes = []
        for idx, entry in enumerate(models):
            missing = [c for c in entry.input_channels if c not in self.input_channels]
            if len(missing) > 0:
                raise ValueError(f"Unknown input channels for submodel #{idx}: {missing}")
            if not hasattr(entry.model, "features"):
                raise ValueError(f"Submodel #{idx} must implement features(x).")
            if not hasattr(entry.model, "feature_dim"):
                raise ValueError(f"Submodel #{idx} must implement feature_dim().")

            self.add_module(f"submodel_{idx}", entry.model)
            self.model_entries.append({
                "model": entry.model,
                "indices": [self.input_channels.index(c) for c in entry.input_channels],
            })
            feature_sizes.append(entry.model.feature_dim())

        self._feature_dim = sum(feature_sizes)
        self.head = torch.nn.Linear(self._feature_dim, len(self.classes))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Slice channels for each submodel and concatenate embeddings."""
        embeddings = []

        for entry in self.model_entries:
            model = entry["model"]
            emb = model.features(x[:, :, entry["indices"]])
            if emb.ndim != 2:
                raise ValueError("MultiModel expects each submodel to produce 2D features [B, E].")
            embeddings.append(emb)

        return torch.cat(embeddings, dim=1)

    def feature_dim(self) -> int:
        """Return the size of the fused embedding space."""
        return self._feature_dim

    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared linear head to fused embeddings."""
        return self.head(x)

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        """Describe the composite raw input shape expected by ``forward``."""
        if len(self.model_entries) == 0:
            raise NotImplementedError("MultiModel without submodels cannot infer an input spec.")
        first_shape, _ = self.model_entries[0]["model"].input_spec()
        return (
            (1, first_shape[1], len(self.input_channels)),
            {"layout": "BTC", "ts_len": first_shape[1], "n_channels": len(self.input_channels)},
        )

    def _warmup_preprocessors(self, data_loader: DataLoader, device: str = "cuda"):
        """Warm composite preprocessors and then nested submodel preprocessors.

        Args:
            data_loader: Loader used to stream warmup batches.
            device: Device used while running warmup updates.

        Returns:
            ``self`` after in-place warmup.
        """
        self.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size

        if batch_size is None:
            raise ValueError("batch_size should not be None here.")

        for idx in range(len(self.preprocessors)):
            logger.progress_start(total_batches * batch_size, desc=f"multi {idx}/{len(self.preprocessors) - 1}", leave=True)
            if self.preprocessors[idx].requires_warmup():
                for batch in data_loader:
                    x = batch["data"].to(device)
                    x = self.apply_preprocessors(x, idx)
                    self.preprocessors[idx].update(x)
                    logger.progress_advance(batch_size)
            else:
                logger.progress_advance(total_batches * batch_size)
            logger.progress_close()

        for entry in self.model_entries:
            model = entry["model"]
            for idx in range(len(model.preprocessors)):
                logger.progress_start(total_batches * batch_size, desc=f"{idx}/{len(model.preprocessors) - 1}", leave=True)
                if model.preprocessors[idx].requires_warmup():
                    for batch in data_loader:
                        x = batch["data"].to(device)
                        x = self.apply_preprocessors(x, len(self.preprocessors) + 1)
                        x = x[:, :, entry["indices"]]
                        x = model.apply_preprocessors(x, idx)
                        model.preprocessors[idx].update(x)
                        logger.progress_advance(batch_size)
                else:
                    logger.progress_advance(total_batches * batch_size)
                logger.progress_close()

        return self
