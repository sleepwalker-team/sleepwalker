"""Helpers for jsonl artifacts and model checkpoints."""

import json
import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import torch

from sleepwalker.models.Basemodel import BaseModel


def read_jsonl(filename: str) -> pd.DataFrame:
    """Read newline-delimited JSON records into a DataFrame."""
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(data)


def append_to_jsonl(filename: str, record: dict):
    """Append one JSON record to a `.jsonl` artifact file.

    Args:
        filename: Artifact stem without the `.jsonl` suffix.
        record: Mapping to serialize. Common NumPy and pandas scalar types are
            normalized through a custom encoder.
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.Timedelta):
                return str(o)
            if isinstance(o, pd.Timestamp):
                return o.isoformat()
            return super().default(o)

    with open(f"{filename}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n")


def store_checkpoint(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    folder: str = tempfile.mkdtemp(prefix="sleepwalker_"),
) -> str:
    """Persist model and optimizer state to one checkpoint directory.

    Args:
        model: Model whose `state_dict` should be stored.
        optimizer: Optimizer to store.
        scheduler: Optional scheduler to store.
        folder: Output directory.

    Returns:
        The checkpoint directory path.
    """
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(folder, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(folder, "optimizer.pt"))
    if scheduler:
        torch.save(scheduler.state_dict(), os.path.join(folder, "scheduler.pt"))
    return folder
