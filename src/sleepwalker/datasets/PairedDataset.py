"""Timestamp-aligned native inputs for a composition of packaged experts."""

import bisect
import copy

import cloudpickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sleepwalker.datasets.Basedataset import stack_repeated_views
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.utils import logger


def initialization_key(dataset):
    """Identify datasets that prepare identical source inputs and recording fits."""
    options = dataset.dataset_kwargs()
    for name in ("stride", "prepare_channels", "input_channels", "prepare_sample", "online_max_tries", "rejection_strategy", "n_views"):
        options.pop(name, None)
    return cloudpickle.dumps(options, protocol=5)


def sample_key(dataset):
    """Identify datasets that build identical input samples from one timestamp."""
    options = dataset.dataset_kwargs()
    for name in ("stride", "online_max_tries", "rejection_strategy", "n_views"):
        options.pop(name, None)
    return cloudpickle.dumps(options, protocol=5)


class PairedDataset(Dataset):
    """Fetch the scheduled native input calls for every expert at one target anchor.

    The paired dataset plans timestamps but delegates all signal I/O and
    preprocessing to each component's ``BaseDataset.get_items`` method.
    Experts with an identical serialized sample contract share window requests
    and reuse the resulting prepared samples.
    """

    def __init__(self, datasets, *, base, input_offsets):
        self.datasets = dict(datasets)
        self.sample_keys = {name: sample_key(dataset) for name, dataset in self.datasets.items()}
        self.base = base
        self.input_offsets = {name: [int(offset) for offset in offsets] for name, offsets in input_offsets.items()}
        starts = [offset for offsets in self.input_offsets.values() for offset in offsets]
        ends = [offset + int(self.datasets[name].total_input.value) for name, offsets in self.input_offsets.items() for offset in offsets]
        self.input_start = pd.Timedelta(min(starts), unit="ns")
        self.input_end = pd.Timedelta(max(ends), unit="ns")
        self.total_input = self.input_end - self.input_start
        self.stride = base.stride
        self.sample_frequency = None
        self.label_classes = list(base.label_classes)
        self.classes = list(base.classes)
        self.online_max_tries = base.online_max_tries
        self.rejection_strategy = base.rejection_strategy
        self.n_views = 1
        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.component_files = {}
        self.initialized = False
        self.all_patients = []

    def dataset_kwargs(self):
        return self.base.dataset_kwargs()

    def get_input_channels(self):
        return {name: dataset.get_input_channels() for name, dataset in self.datasets.items()}

    def input_spec(self):
        channels = self.get_input_channels()
        return {name: (1, len(self.input_offsets[name]), dataset.get_timeseries_len(), len(channels[name])) for name, dataset in self.datasets.items()}

    def get_n_patients(self):
        return len(self.edf_files)

    def get_patient_ranges(self):
        if not self.initialized:
            raise ValueError("PairedDataset is not initialized.")
        return list(zip(self.lower_bounds, self.upper_bounds))

    def get_classes(self):
        return list(self.classes)

    def has_extra_target(self):
        return self.base.has_extra_target()

    def set_rejection_strategy(self, strategy):
        self.base.set_rejection_strategy(strategy)
        self.rejection_strategy = self.base.rejection_strategy

    def set_n_views(self, n_views):
        self.n_views = int(n_views)
        if self.n_views < 1:
            raise ValueError("n_views must be positive.")

    def set_edf_cache(self, cache):
        """Share one loading-session cache across the base and all experts."""
        self.base.set_edf_cache(cache)
        for dataset in self.datasets.values():
            dataset.set_edf_cache(cache)

    def clone(self):
        datasets = {name: dataset.clone() if isinstance(dataset, UnlabelledDataset) else copy.deepcopy(dataset) for name, dataset in self.datasets.items()}
        base = self.base.clone() if isinstance(self.base, UnlabelledDataset) else copy.deepcopy(self.base)
        return PairedDataset(datasets, base=base, input_offsets=self.input_offsets)

    def with_labels(self, labelled_dataset):
        channels = labelled_dataset.channels
        datasets = {}
        for name, dataset in self.datasets.items():
            source = dataset if isinstance(dataset, UnlabelledDataset) else UnlabelledDataset.from_dataset(dataset)
            if channels:
                expected = [channel.logical_name for channel in source.channels]
                actual = [channel.logical_name for channel in channels]
                if actual != expected:
                    raise ValueError(f"Evaluation channels for expert '{name}' must be {expected}, got {actual}.")
                datasets[name] = source.clone(channels=copy.deepcopy(channels))
            else:
                datasets[name] = source.clone()
        return PairedDataset(datasets, base=labelled_dataset, input_offsets=self.input_offsets)

    def to_unlabelled(self):
        datasets = {name: dataset.clone() if isinstance(dataset, UnlabelledDataset) else UnlabelledDataset.from_dataset(dataset) for name, dataset in self.datasets.items()}
        base = self.base.clone() if isinstance(self.base, UnlabelledDataset) else UnlabelledDataset.from_dataset(self.base)
        return PairedDataset(datasets, base=base, input_offsets=self.input_offsets)

    def set_input_offsets(self, input_offsets):
        normalized = {name: [int(offset) for offset in offsets] for name, offsets in input_offsets.items()}
        if normalized != self.input_offsets:
            raise ValueError("Input offsets are fixed by the expert prediction contracts.")

    def initialize(self, patients, num_workers=4, *, strict=False):
        self.all_patients = list(patients)
        self.initialized = False
        start_method = "forkserver" if num_workers > 1 else None
        self.base.initialize(patients, num_workers=num_workers, strict=strict, multiprocessing_start_method=start_method)
        prepared_files = {}
        self.component_files = {}
        for name, dataset in self.datasets.items():
            key = initialization_key(dataset)
            if key not in prepared_files:
                dataset.initialize(patients, num_workers=num_workers, strict=strict, multiprocessing_start_method=start_method)
                prepared_files[key] = {str(file.path): file for file in dataset.edf_files}
            else:
                logger.info(f"Reusing prepared EDF channels for expert '{name}'.")
            self.component_files[name] = prepared_files[key]
        common_paths = set(str(file.path) for file in self.base.edf_files)
        for files in self.component_files.values():
            common_paths &= set(files)
        base_files = [file for file in self.base.edf_files if str(file.path) in common_paths]
        if not base_files:
            raise ValueError("Paired datasets have no initialized recordings in common.")

        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        lower = 0
        for base_file in base_files:
            offsets = np.arange(base_file.length, dtype=np.int64) if base_file.start_offsets is None else np.asarray(base_file.start_offsets, dtype=np.int64)
            anchors = pd.DatetimeIndex(base_file.start_date + pd.to_timedelta(offsets * int(self.base.stride.value), unit="ns"))
            valid = np.ones(len(offsets), dtype=bool)
            for name, dataset in self.datasets.items():
                local = self.component_files[name][str(base_file.path)]
                if local.end_date is None:
                    local_offsets = np.arange(local.length, dtype=np.int64) if local.start_offsets is None else np.asarray(local.start_offsets, dtype=np.int64)
                    first = local.start_date + int(local_offsets.min()) * dataset.stride
                    last = local.start_date + int(local_offsets.max()) * dataset.stride
                else:
                    first = local.start_date
                    last = local.end_date - dataset.total_input
                for offset in self.input_offsets[name]:
                    starts = anchors + pd.Timedelta(offset, unit="ns")
                    valid &= (starts >= first) & (starts <= last)
            if valid.any():
                copied = copy.copy(base_file)
                copied.start_offsets = offsets[valid]
                copied.length = int(valid.sum())
                self.edf_files.append(copied)
                self.lower_bounds.append(lower)
                lower += copied.length
                self.upper_bounds.append(lower)
        if not self.edf_files:
            raise ValueError("No reference windows can satisfy every expert input contract.")
        self.label_classes = list(self.base.label_classes)
        self.classes = list(self.base.classes)
        self.initialized = True

    def __len__(self):
        return self.upper_bounds[-1] if self.upper_bounds else 0

    def get_items(self, locations):
        """Build aligned samples while sharing native reads within each patient."""
        requests = {}
        plans = []
        for base_file, start in locations:
            base_item = self.base.get_target_item(base_file, start)
            if base_item is None:
                plans.append(None)
                continue
            patient = str(base_file.path)
            expert_starts = {}
            for name, dataset in self.datasets.items():
                requested_starts = [pd.Timestamp(base_item["time"]) + pd.Timedelta(offset, unit="ns") for offset in self.input_offsets[name]]
                expert_starts[name] = requested_starts
                key = (patient, self.sample_keys[name])
                if key not in requests:
                    requests[key] = {"dataset": dataset, "file": self.component_files[name][patient], "starts": []}
                requests[key]["starts"].extend(requested_starts)
            plans.append((patient, base_item, expert_starts))

        loaded = {}
        for key, request in requests.items():
            requested_starts = list(dict.fromkeys(request["starts"]))
            local_items = request["dataset"].get_items(request["file"], requested_starts)
            loaded[key] = dict(zip(requested_starts, local_items))

        results = []
        for plan in plans:
            if plan is None:
                results.append(None)
                continue
            patient, base_item, expert_starts = plan
            inputs = {}
            rejected = False
            for name, requested_starts in expert_starts.items():
                local_items = [loaded[(patient, self.sample_keys[name])][requested] for requested in requested_starts]
                if any(local_item is None for local_item in local_items):
                    rejected = True
                    break
                for requested, local_item in zip(requested_starts, local_items):
                    if pd.Timestamp(local_item["time"]) != requested:
                        raise ValueError(f"Expert '{name}' returned {local_item['time']} for requested time {requested}.")
                inputs[name] = local_items[0]["data"].unsqueeze(0) if len(local_items) == 1 else torch.stack([item["data"] for item in local_items])
            if rejected:
                results.append(None)
                continue
            result = dict(base_item)
            result["data"] = inputs
            results.append(result)
        return results

    def get_item(self, base_file, start):
        """Build one aligned sample while leaving window loading to BaseDataset."""
        return self.get_items([(base_file, start)])[0]

    def candidate_indices(self, original_index):
        patient_index = bisect.bisect_right(self.upper_bounds, original_index)
        yield original_index
        if self.rejection_strategy in {"patient", "patient_then_global"}:
            for _ in range(max(0, self.online_max_tries - 1)):
                yield int(np.random.randint(self.lower_bounds[patient_index], self.upper_bounds[patient_index]))
        if self.rejection_strategy in {"global", "patient_then_global"}:
            for _ in range(self.online_max_tries):
                yield int(np.random.choice(len(self)))

    def item_from_index(self, index):
        base_file, start = self.location_from_index(index)
        views = []
        for _ in range(self.n_views):
            item = self.get_item(base_file, start)
            if item is None:
                return None
            views.append(item)
        return views[0] if self.n_views == 1 else stack_repeated_views(views)

    def location_from_index(self, index):
        patient_index = bisect.bisect_right(self.upper_bounds, index)
        base_file = self.edf_files[patient_index]
        local_index = index - self.lower_bounds[patient_index]
        start = base_file.start_date + self.stride * int(base_file.start_offsets[local_index])
        return base_file, start

    def __getitem__(self, index):
        if not self.initialized:
            raise ValueError("PairedDataset is not initialized.")
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for candidate_index in self.candidate_indices(index):
            item = self.item_from_index(candidate_index)
            if item is not None:
                return item
        return None

    def __getitems__(self, indices):
        """Fetch one evaluation batch, sharing requests by patient and sample contract."""
        indices = list(indices)
        if not self.initialized:
            raise ValueError("PairedDataset is not initialized.")
        if any(index < 0 or index >= len(self) for index in indices):
            raise IndexError("PairedDataset batch contains an out-of-range index.")
        if self.n_views != 1 or self.rejection_strategy != "none":
            return [self[index] for index in indices]
        return self.get_items([self.location_from_index(index) for index in indices])
