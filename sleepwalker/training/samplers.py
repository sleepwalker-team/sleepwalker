"""Samplers used by the shared training loader."""

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class PatientSampler(Sampler[int]):
    """Sample an equal number of windows from a rotating patient subset.

    Patient ranges are grouped in the yielded index stream so nearby batches
    reuse a small set of EDF files. Within each group, patients are interleaved
    to preserve patient diversity. A patient is exhausted before any of its
    windows are sampled a second time.
    """

    def __init__(self, dataset, *, num_samples: int | None, patients_per_epoch: int, patient_group_size: int | None = None, generator: torch.Generator | None = None):
        if patients_per_epoch < 1:
            raise ValueError("patients_per_epoch must be positive.")
        if patient_group_size is not None and patient_group_size < 1:
            raise ValueError("patient_group_size must be positive when provided.")
        if patient_group_size is not None and patient_group_size > patients_per_epoch:
            raise ValueError("patient_group_size must not exceed patients_per_epoch.")
        if not hasattr(dataset, "get_patient_ranges"):
            raise ValueError(f"{dataset.__class__.__name__} does not expose get_patient_ranges() required by PatientSampler.")

        patient_ranges = list(dataset.get_patient_ranges())
        if not patient_ranges:
            raise ValueError("PatientSampler requires at least one patient range.")

        expected_lower = 0
        for patient_index, bounds in enumerate(patient_ranges):
            if len(bounds) != 2:
                raise ValueError(f"Patient range {patient_index} must contain exactly two bounds.")
            lower, upper = bounds
            if lower != expected_lower or upper <= lower:
                raise ValueError(f"Patient ranges must be a contiguous partition of the dataset; got range {bounds} at position {patient_index}.")
            expected_lower = upper
        if expected_lower != len(dataset):
            raise ValueError(f"Patient ranges end at {expected_lower}, but the dataset contains {len(dataset)} samples.")

        self.patient_ranges = patient_ranges
        self.patients_per_epoch = min(int(patients_per_epoch), len(patient_ranges))
        self.patient_group_size = min(int(patient_group_size or self.patients_per_epoch), self.patients_per_epoch)
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if self.num_samples < self.patients_per_epoch:
            raise ValueError(f"num_samples={self.num_samples} must be at least the effective patients_per_epoch={self.patients_per_epoch}.")
        self.seed = int(generator.initial_seed()) if generator is not None else int(torch.initial_seed())
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative.")
        self.epoch = int(epoch)

    def selected_patients(self) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        patient_order = torch.randperm(len(self.patient_ranges), generator=generator).tolist()
        start = self.epoch * self.patients_per_epoch % len(patient_order)
        return [patient_order[(start + offset) % len(patient_order)] for offset in range(self.patients_per_epoch)]

    def sample_patient(self, patient_index: int, count: int, generator: torch.Generator) -> list[int]:
        lower, upper = self.patient_ranges[patient_index]
        patient_length = upper - lower
        unique_count = min(count, patient_length)
        indices = (torch.randperm(patient_length, generator=generator)[:unique_count] + lower).tolist()
        if count > patient_length:
            indices.extend(torch.randint(lower, upper, (count - patient_length,), generator=generator).tolist())
        return indices

    def __iter__(self) -> Iterator[int]:
        selected_patients = self.selected_patients()
        samples_per_patient, extra_samples = divmod(self.num_samples, len(selected_patients))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + 1)

        for group_start in range(0, len(selected_patients), self.patient_group_size):
            group = selected_patients[group_start:group_start + self.patient_group_size]
            sampled_indices = []
            for position, patient_index in enumerate(group, start=group_start):
                count = samples_per_patient + int(position < extra_samples)
                sampled_indices.append(self.sample_patient(patient_index, count, generator))

            for sample_position in range(max(map(len, sampled_indices))):
                for patient_indices in sampled_indices:
                    if sample_position < len(patient_indices):
                        yield patient_indices[sample_position]
