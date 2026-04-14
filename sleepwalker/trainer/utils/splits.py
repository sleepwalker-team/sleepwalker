from __future__ import annotations

from typing import Sequence

from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.utils import random_split


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
