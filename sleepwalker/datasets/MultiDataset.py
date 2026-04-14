from __future__ import annotations

import bisect
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from torch.utils.data import Dataset
from sleepwalker.datasets.Basedataset import BaseDataset

class MultiDataset(Dataset):
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

    def get_n_datasets(self):
        return len(self.datasets)
    
    def get_classes(self):
        # We enforced in the c'tor that all datasets have the same classes, so pick one here
        return self.datasets[0].get_classes()

    def get_timeseries_len(self):
        # We enforced in the c'tor that all datasets have the same classes, so pick one here
        return self.datasets[0].get_timeseries_len()

    def has_extra_target(self):
        return self.extra_target 

    def get_n_patients(self):
        return sum([d.get_n_patients() for d in self.datasets])

    def get_patient_index_spans(self) -> list[tuple[int, int]]:
        spans = []
        dataset_offset = 0
        for dataset in self.datasets:
            for lower, upper in dataset.get_patient_index_spans():
                spans.append((dataset_offset + lower, dataset_offset + upper))
            dataset_offset += len(dataset)
        return spans

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d_idx = bisect.bisect_right(self.upper_bound, idx)
        new_idx = idx - self.lower_bound[d_idx]

        return {"dataset":d_idx, **self.datasets[d_idx].__getitem__(new_idx)}
