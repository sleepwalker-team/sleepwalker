"""Cache-backed dataset for realized numpy exports.

This dataset reads the directory format produced by
``export_dataloader_to_numpy_dir`` and exposes it through a dataset-like API.
Tests in ``tests/test_datasets.py`` and ``tests/test_deployment.py`` cover the
current round-trip, memmap, and prediction-package export behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig


class NumpyDataset:
    """
    Cache-backed dataset for arrays exported from an initialized dataset.

    The cache format is a directory containing:
    - ``meta.json``
    - ``data.npy`` or ``data.000000.npy`` sharded files
    - optional ``target.npy`` or ``target.000000.npy`` shards
    - optional ``target_extra.npy`` or ``target_extra.000000.npy`` shards
    - ``time.npy`` / ``patient.npy`` or matching shards
    - optional ``extra__{key}.npy`` arrays or shards

    Notes on ``extra__{key}.npy`` fields
    ------------------------------------
    ``extra_keys`` are intentionally limited to values with a stable, numpy-
    serializable representation across every exported item. In practice this
    means:

    - scalars such as ``int``, ``float``, ``bool``, ``str``
    - tensors / numpy arrays with the same shape for every item
    - pandas objects that collapse cleanly to same-shaped numpy arrays

    This cache path does not attempt to preserve arbitrary Python objects or
    ragged nested structures. If an extra field changes shape between items or
    only exists on some items, export fails loudly instead of silently pickling
    heterogeneous payloads.
    """

    def __init__(
        self,
        cache_path: str | Path,
        in_memory: bool = True,
    ) -> None:
        """Load a numpy cache directory as a dataset-like object.

        Args:
            cache_path: Directory containing ``meta.json`` and one or more
                exported array files.
            in_memory: Whether arrays should be fully loaded into memory.
                ``False`` uses NumPy memmap mode where possible.
        """
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise ValueError(f"Cache path does not exist: {self.cache_path}")

        meta_path = self.cache_path / "meta.json"
        if not meta_path.exists():
            raise ValueError(f"Cache metadata not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        mmap_mode = None if in_memory else "r"
        self.data_shards = self.load_array_shards("data", mmap_mode, required=True)
        self.target_shards = self.load_array_shards("target", mmap_mode, required=False)
        self.target_extra_shards = self.load_array_shards("target_extra", mmap_mode, required=False)
        self.time_shards = self.load_array_shards("time", mmap_mode, required=True)
        self.patient_shards = self.load_array_shards("patient", mmap_mode, required=True)

        self.extra_keys = list(meta.get("extra_keys", []))
        self.extra_shards = {
            key: self.load_array_shards(f"extra__{key}", mmap_mode, required=True)
            for key in self.extra_keys
        }

        shard_lengths = [int(shard.shape[0]) for shard in self.data_shards]
        self.cum_lengths = np.cumsum(shard_lengths)

        self.sample_frequency = float(meta["sample_frequency"])
        self.resample_type = str(meta["resample_type"])
        self.total_input = pd.to_timedelta(meta["total_input"])
        self.target_resolution = pd.to_timedelta(meta["target_resolution"])
        self.stride = pd.to_timedelta(meta.get("stride", meta["target_resolution"]))
        self.classes = list(meta.get("classes", []))
        self.label_classes = list(self.classes)
        self.input_channels = list(meta.get("input_channels", []))
        self.channels = [
            ChannelConfig(logical_name=name, physical_names=[name])
            for name in self.input_channels
        ]
        self.initialized = True
        self.all_patients = list(meta.get("all_patients", []))

    def load_array_shards(self, stem: str, mmap_mode: Optional[str], *, required: bool) -> Optional[list[np.ndarray]]:
        """Load one array or a set of sharded arrays from the cache directory."""
        single_path = self.cache_path / f"{stem}.npy"
        if single_path.exists():
            return [np.load(single_path, mmap_mode=mmap_mode)]

        shard_paths = sorted(self.cache_path.glob(f"{stem}.*.npy"))
        if shard_paths:
            return [np.load(path, mmap_mode=mmap_mode) for path in shard_paths]

        if required:
            raise ValueError(f"Cache array not found for '{stem}' in {self.cache_path}")
        return None

    def resolve_index(self, idx: int) -> tuple[int, int]:
        """Map a global row index to ``(shard_idx, local_idx)``."""
        shard_idx = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        prev = 0 if shard_idx == 0 else int(self.cum_lengths[shard_idx - 1])
        return shard_idx, idx - prev

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def has_extra_target(self) -> bool:
        """Report whether the cache contains a secondary target array."""
        return self.target_extra_shards is not None

    def get_classes(self) -> list[str]:
        """Return the cached class ordering."""
        return self.classes

    def get_timeseries_len(self) -> int:
        """Return the number of timesteps per cached sample."""
        return int(self.data_shards[0].shape[1])

    def get_n_patients(self) -> int:
        """Return the number of distinct patient identifiers in the cache."""
        return len(set(str(p) for shard in self.patient_shards for p in shard.tolist()))

    def get_input_channels(self) -> list[str]:
        """Return the cached input channel names."""
        return list(self.input_channels)

    def _to_tensor(self, value: np.ndarray) -> torch.Tensor:
        arr = np.asarray(value)
        if not arr.flags.writeable:
            arr = np.array(arr, copy=True)
        return torch.as_tensor(arr)

    def _restore_extra_value(self, value: Any) -> Any:
        arr = np.asarray(value)
        if arr.ndim == 0:
            scalar = arr.item()
            if isinstance(scalar, np.generic):
                scalar = scalar.item()
            return scalar
        if arr.dtype.kind in {"U", "S", "O"}:
            return arr.tolist()
        return self._to_tensor(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one cached sample.

        Args:
            idx: Global sample index.

        Returns:
            A dictionary containing ``data``, ``patient``, ``time``, optional
            targets, and any exported extra keys.

        Raises:
            IndexError: If ``idx`` is outside the dataset range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        shard_idx, local_idx = self.resolve_index(idx)

        item: Dict[str, Any] = {
            "data": self._to_tensor(self.data_shards[shard_idx][local_idx]).float(),
            "patient": str(self.patient_shards[shard_idx][local_idx]),
            "time": pd.Timestamp(int(self.time_shards[shard_idx][local_idx])),
        }

        if self.target_shards is not None:
            item["target"] = self._to_tensor(self.target_shards[shard_idx][local_idx])
        if self.target_extra_shards is not None:
            item["target_extra"] = self._to_tensor(self.target_extra_shards[shard_idx][local_idx])

        for key, shards in self.extra_shards.items():
            item[key] = self._restore_extra_value(shards[shard_idx][local_idx])

        return item
