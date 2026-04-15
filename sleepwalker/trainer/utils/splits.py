"""Small helpers for patient splitting and dataset combination.

These helpers are used by training scripts and the shared run pipeline. They
are intentionally minimal and reflect current lab workflows rather than a
stable experiment-management API.
"""

from __future__ import annotations

from typing import Sequence

from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.utils import random_split


def split_patients_train_val_test(patients: list[str], test_frac: float, val_frac: float):
    """Split patients into train, validation, and test partitions.

    Args:
        patients: Patient identifiers or EDF paths.
        test_frac: Fraction reserved for the test split.
        val_frac: Fraction reserved for validation after removing the test
            split.

    Returns:
        A tuple `(train_patients, val_patients, test_patients)`.
    """
    train_patients, test_patients = random_split(patients, test_frac=test_frac)
    train_patients, val_patients = random_split(train_patients, test_frac=val_frac)
    return train_patients, val_patients, test_patients


def combine_datasets(parts: Sequence[object]):
    """Return one dataset-like object from one or more parts.

    Args:
        parts: Dataset objects to combine.

    Returns:
        The single dataset unchanged when only one part is provided, otherwise
        a `MultiDataset` wrapper.
    """
    if len(parts) == 0:
        raise ValueError("Cannot combine an empty dataset list.")
    if len(parts) == 1:
        return parts[0]
    return MultiDataset(list(parts))
