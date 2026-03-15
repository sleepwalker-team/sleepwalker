from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.utils import logger


@dataclass
class MetaModelEntry:
    name: str
    model: BaseModel
    input_channels: list[str]


class MetaModel(BaseModel):
    def __init__(
        self,
        *,
        classes: list[str],
        input_channels: list[str],
        models: list[MetaModelEntry],
        sample_frequency: Optional[float] = None,
        ts_len: Optional[int] = None,
    ):
        super().__init__()

        self.classes = list(classes)
        self.input_channels = list(input_channels)
        self.sample_frequency = sample_frequency
        self.ts_len = ts_len

        self.model_entries = OrderedDict()
        embedding_sizes = []
        used_names = set()

        for entry in models:
            if entry.name in used_names:
                raise ValueError(f"Duplicate metamodel entry '{entry.name}'.")
            used_names.add(entry.name)

            missing = [c for c in entry.input_channels if c not in self.input_channels]
            if len(missing) > 0:
                raise ValueError(f"Unknown input channels for '{entry.name}': {missing}")

            if not hasattr(entry.model, "classes") or len(entry.model.classes) <= 0:
                raise ValueError(f"Model '{entry.name}' must be initialized with embedding classes.")

            if hasattr(entry.model, "n_channels") and entry.model.n_channels != len(entry.input_channels):
                raise ValueError(
                    f"Model '{entry.name}' expects {entry.model.n_channels} channels but got {len(entry.input_channels)}."
                )
            if hasattr(entry.model, "nchannel") and entry.model.nchannel != len(entry.input_channels):
                raise ValueError(
                    f"Model '{entry.name}' expects {entry.model.nchannel} channels but got {len(entry.input_channels)}."
                )
            if ts_len is not None and hasattr(entry.model, "ts_len") and entry.model.ts_len != ts_len:
                raise ValueError(f"Model '{entry.name}' expects ts_len={entry.model.ts_len} but metamodel uses {ts_len}.")
            if sample_frequency is not None and hasattr(entry.model, "fs"):
                expected_fs = pd.to_timedelta(1.0 / sample_frequency, unit="s")
                if entry.model.fs != expected_fs:
                    raise ValueError(
                        f"Model '{entry.name}' expects sample spacing {entry.model.fs} but metamodel uses {expected_fs}."
                    )

            embedding_size = len(entry.model.classes)
            entry.model.classes = [f"{entry.name}_{i}" for i in range(embedding_size)]

            self.add_module(f"submodel_{entry.name}", entry.model)
            self.model_entries[entry.name] = {
                "model": entry.model,
                "indices": [self.input_channels.index(c) for c in entry.input_channels],
                "labels": list(entry.input_channels),
            }
            embedding_sizes.append(embedding_size)

        self.fc = torch.nn.Linear(sum(embedding_sizes), len(self.classes))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []

        for entry in self.model_entries.values():
            model = entry["model"]
            emb = model(x[:, :, entry["indices"]])
            if emb.ndim != 2:
                raise ValueError("MetaModel expects each submodel to return [B, E].")
            embeddings.append(emb)

        return self.fc(torch.cat(embeddings, dim=1))

    def _warmup_preprocessors(self, data_loader: DataLoader, device: str = "cuda"):
        self.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size

        if batch_size is None:
            raise ValueError("batch_size should not be None here.")

        for name, entry in self.model_entries.items():
            model = entry["model"]
            for idx in range(len(model.preprocessors)):
                logger.progress_start(total_batches * batch_size, desc=f"{name} {idx}/{len(model.preprocessors) - 1}", leave=True)
                if model.preprocessors[idx].requires_warmup():
                    for batch in data_loader:
                        x = batch["data"][:, :, entry["indices"]].to(device)
                        x = model.apply_preprocessors(x, idx)
                        model.preprocessors[idx].update(x)
                        logger.progress_advance(batch_size)
                else:
                    logger.progress_advance(total_batches * batch_size)
                logger.progress_close()

        return self
