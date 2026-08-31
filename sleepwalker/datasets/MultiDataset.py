"""Dataset wrapper that concatenates several initialized datasets.

`MultiDataset` is used when training should sample from several prepared
datasets while preserving dataset identity in each returned item.
"""

from __future__ import annotations

import bisect
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from torch.utils.data import Dataset
from sleepwalker.datasets.Basedataset import BaseDataset

class MultiDataset(Dataset):
    """Concatenate several initialized datasets into one dataset-like object.

    Args:
        datasets: Initialized datasets with matching timing and class
            definitions.

    Notes:
        Current code enforces matching `sample_frequency`, `target_resolution`,
        `total_input`, `stride`, and class sets across all parts.
    """
    def __init__(self, 
            datasets: list[BaseDataset]
        ): 
        
        if not datasets:
            raise ValueError(f"Datasets must at-least contain one dataset")

        inits = [d.initialized for d in datasets]
        if not all(inits):
            raise ValueError(f"All datasets must be initialized")
        
        sample_frequency = [d.sample_frequency for d in datasets]
        if len(set(sample_frequency)) > 1:
            raise ValueError(f"All datasets must have the same sample_frequency")

        target_resolution = [d.target_resolution for d in datasets]
        if len(set(target_resolution)) > 1:
            raise ValueError(f"All datasets must have the same target_resolution")
        
        total_input = [d.total_input for d in datasets]
        if len(set(total_input)) > 1:
            raise ValueError(f"All datasets must have the same total_input")

        strides = [getattr(d, "stride", d.target_resolution) for d in datasets]
        if len(set(strides)) > 1:
            raise ValueError(f"All datasets must have the same stride")
        
        classes = [set(d.classes) for d in datasets]
        if not all(set(lst) == set(classes[0]) for lst in classes):
            raise ValueError(f"All datasets must have the same classes")

        self.lower_bound = []
        self.upper_bound = []

        lower = 0
        upper = 0
        for d in datasets:
            upper += len(d)
            self.lower_bound.append(lower)
            self.upper_bound.append(upper)
            lower += len(d)

        self.datasets = datasets
        self.len = sum([len(d) for d in datasets])
        self.extra_target = all([d.has_extra_target() for d in datasets])
        self.sample_frequency = datasets[0].sample_frequency
        self.target_resolution = datasets[0].target_resolution
        self.total_input = datasets[0].total_input
        self.stride = getattr(datasets[0], "stride", datasets[0].target_resolution)
        self.channels = datasets[0].channels
        input_channels = [dataset.get_input_channels() for dataset in datasets]
        if any(len(channels) != len(input_channels[0]) for channels in input_channels[1:]):
            raise ValueError("All datasets must expose the same number of input channels")
        self.input_channels = input_channels[0]
        self.n_views = 1

    def get_n_datasets(self):
        """Return the number of component datasets."""
        return len(self.datasets)
    
    def get_classes(self):
        """Return the shared class list."""
        # We enforced in the c'tor that all datasets have the same classes, so pick one here
        return self.datasets[0].get_classes()

    def get_timeseries_len(self):
        """Return the shared timeseries length."""
        # We enforced in the c'tor that all datasets have the same classes, so pick one here
        return self.datasets[0].get_timeseries_len()

    def get_input_channels(self):
        """Return the first dataset's reference input-channel order."""
        return list(self.input_channels)

    def has_extra_target(self):
        """Return whether every component dataset exposes `target_extra`."""
        return self.extra_target 

    def get_n_patients(self):
        """Return the total number of patients across all component datasets."""
        return sum([d.get_n_patients() for d in self.datasets])

    def set_rejection_strategy(self, strategy: str) -> None:
        for dataset in self.datasets:
            dataset.set_rejection_strategy(strategy)

    def set_n_views(self, n_views: int) -> None:
        for dataset in self.datasets:
            dataset.set_n_views(n_views)
        self.n_views = int(n_views)

    def get_patient_ranges(self) -> list[tuple[int, int]]:
        """Return component patient ranges shifted into the combined index."""
        ranges = []
        for dataset, offset in zip(self.datasets, self.lower_bound):
            ranges.extend((lower + offset, upper + offset) for lower, upper in dataset.get_patient_ranges())
        return ranges

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Return one item plus its originating dataset index."""
        d_idx = bisect.bisect_right(self.upper_bound, idx)
        new_idx = idx - self.lower_bound[d_idx]
        item = self.datasets[d_idx][new_idx]
        if item is None:
            return None

        return {"dataset": d_idx, **item}


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
