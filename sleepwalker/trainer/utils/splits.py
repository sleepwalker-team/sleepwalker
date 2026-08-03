"""Small dataset and split helpers shared by training scripts."""

from pathlib import Path
from typing import Any

import yaml

from sleepwalker.datasets.MultiDataset import combine_datasets


def load_split(path: str | Path) -> dict[str, Any]:
    """Load a YAML split artifact and require a top-level mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        split = yaml.safe_load(handle)
    if not isinstance(split, dict):
        raise ValueError(f"Expected a top-level mapping in split file {path}.")
    return split


__all__ = ["combine_datasets", "load_split"]
