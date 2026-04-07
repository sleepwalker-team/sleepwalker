from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.utils import logger


@dataclass
class MetaModelEntry:
    model: BaseModel
    input_channels: list[str]


class MetaModel(BaseModel):
    def __init__(
        self,
        *,
        task_config: dict[str, dict],
        input_channels: Optional[list[str]] = None,
        models: list[MetaModelEntry],
        preprocessors: Optional[list] = None,
    ):
        super().__init__(preprocessors=preprocessors)

        self.task_config = dict(task_config)
        self.classes = []
        self.input_channels = [] if input_channels is None else list(input_channels)
        for task, cfg in self.task_config.items():
            if "labels" not in cfg or "n_steps" not in cfg:
                raise ValueError(f"Task '{task}' must provide normalized config with 'labels' and 'n_steps'.")
            self.classes.extend(cfg["labels"])

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
        self.heads = torch.nn.ModuleDict({
            task: torch.nn.Linear(self._feature_dim, cfg["n_steps"] * len(cfg["labels"]))
            for task, cfg in self.task_config.items()
        })

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []

        for entry in self.model_entries:
            model = entry["model"]
            emb = model.features(x[:, :, entry["indices"]])
            if emb.ndim != 2:
                raise ValueError("MetaModel expects each submodel to produce 2D features [B, E].")
            embeddings.append(emb)

        return torch.cat(embeddings, dim=1)

    def feature_dim(self) -> int:
        return self._feature_dim

    def _classifier(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for task, cfg in self.task_config.items():
            logits = self.heads[task](x)
            out[task] = logits.view(x.shape[0], cfg["n_steps"], len(cfg["labels"]))
        return out

    def _warmup_preprocessors(self, data_loader: DataLoader, device: str = "cuda"):
        self.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size

        if batch_size is None:
            raise ValueError("batch_size should not be None here.")

        for idx in range(len(self.preprocessors)):
            logger.progress_start(total_batches * batch_size, desc=f"meta {idx}/{len(self.preprocessors) - 1}", leave=True)
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
