from __future__ import annotations

from pathlib import Path
from typing import Sequence

from torch.utils.data import DataLoader

from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.NumpyDataset import NumpyDataset
from sleepwalker.datasets.utils import export_dataloader_to_numpy_dir, random_split
from sleepwalker.utils import logger


def split_patients_train_val_test(patients: list[str], test_frac: float, val_frac: float):
    train_patients, test_patients = random_split(patients, test_frac=test_frac)
    train_patients, val_patients = random_split(train_patients, test_frac=val_frac)
    return train_patients, val_patients, test_patients


def combine_datasets(parts: Sequence[object]):
    if len(parts) == 0:
        raise ValueError("Cannot combine an empty dataset list.")
    if len(parts) == 1:
        return parts[0]
    return MultiDataset(list(parts))


def load_or_build_numpy_cache(
    export_loader: DataLoader,
    cache_path: str | Path,
    *,
    in_memory: bool = False,
    extra_keys: Sequence[str] = (),
    n_samples_per_file: int | None = None,
):
    cache_path = Path(cache_path)
    if (cache_path / "meta.json").exists() and ((cache_path / "data.npy").exists() or len(list(cache_path.glob("data.*.npy"))) > 0):
        logger.info(f"Loading frozen sample cache from {cache_path}")
        return NumpyDataset(cache_path, in_memory=in_memory)

    if len(export_loader.dataset) == 0:
        raise ValueError(f"Cannot build numpy cache at {cache_path}: export loader dataset is empty.")

    logger.info(f"Frozen sample cache not found at {cache_path}. Building it from EDF files.")
    export_dataloader_to_numpy_dir(
        export_loader,
        cache_path,
        extra_keys=list(extra_keys),
        n_samples_per_file=n_samples_per_file,
    )
    return NumpyDataset(cache_path, in_memory=in_memory)
