from bisect import bisect_right

import pytest
import torch
from torch.utils.data import RandomSampler

from sleepwalker.training.loader import build_loader
from sleepwalker.training.samplers import PatientSampler


class RangeDataset:
    def __init__(self, patient_lengths):
        self.patient_ranges = []
        lower = 0
        for length in patient_lengths:
            self.patient_ranges.append((lower, lower + length))
            lower += length
        self.length = lower

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index

    def get_patient_ranges(self):
        return list(self.patient_ranges)


def patient_for_index(dataset, index):
    return bisect_right([upper for _, upper in dataset.patient_ranges], index)


def build_patient_sampler(dataset, **kwargs):
    generator = torch.Generator()
    generator.manual_seed(17)
    return PatientSampler(dataset, generator=generator, **kwargs)


def test_patient_sampler_balances_patients_and_repeats_only_when_needed():
    dataset = RangeDataset([1, 2, 4, 5])
    sampler = build_patient_sampler(dataset, num_samples=12, patients_per_epoch=2, patient_group_size=2)

    indices = list(sampler)
    patient_indices = [patient_for_index(dataset, index) for index in indices]
    selected = sampler.selected_patients()

    assert len(indices) == len(sampler) == 12
    assert patient_indices == selected * 6
    assert {patient: patient_indices.count(patient) for patient in selected} == {patient: 6 for patient in selected}
    for patient in selected:
        patient_windows = [index for index in indices if patient_for_index(dataset, index) == patient]
        assert len(set(patient_windows)) == min(6, dataset.patient_ranges[patient][1] - dataset.patient_ranges[patient][0])


def test_patient_sampler_rotates_patient_coverage_between_epochs():
    dataset = RangeDataset([3, 3, 3, 3])
    sampler = build_patient_sampler(dataset, num_samples=8, patients_per_epoch=2, patient_group_size=2)

    epoch_zero = sampler.selected_patients()
    sampler.set_epoch(1)
    epoch_one = sampler.selected_patients()

    assert set(epoch_zero).isdisjoint(epoch_one)
    assert set(epoch_zero + epoch_one) == set(range(4))


def test_patient_sampler_keeps_groups_local_and_is_repeatable_within_epoch():
    dataset = RangeDataset([4, 4, 4, 4])
    sampler = build_patient_sampler(dataset, num_samples=12, patients_per_epoch=4, patient_group_size=2)

    first = list(sampler)
    second = list(sampler)
    patients = [patient_for_index(dataset, index) for index in first]

    assert first == second
    assert set(patients[:6]) == set(sampler.selected_patients()[:2])
    assert set(patients[6:]) == set(sampler.selected_patients()[2:])


def test_build_loader_selects_patient_sampler_only_when_configured():
    dataset = RangeDataset([4, 4, 4])
    patient_loader = build_loader(dataset, batch_size=2, num_workers=0, n_samples=9, collate_fn=lambda samples: samples, sampling="patient_balanced", seed=17, patients_per_epoch=2, patient_group_size=2)
    legacy_loader = build_loader(dataset, batch_size=2, num_workers=0, n_samples=9, collate_fn=lambda samples: samples, sampling="random", seed=17)

    assert isinstance(patient_loader.sampler, PatientSampler)
    assert isinstance(legacy_loader.sampler, RandomSampler)


def test_build_loader_derives_patient_group_size_from_loader_capacity():
    dataset = RangeDataset([100] * 64)

    loader = build_loader(dataset, batch_size=32, num_workers=4, n_samples=1024, collate_fn=lambda samples: samples, sampling="patient_balanced", seed=17, patients_per_epoch=64)

    assert isinstance(loader.sampler, PatientSampler)
    assert loader.sampler.patient_group_size == 16


def test_build_loader_preserves_explicit_patient_group_size():
    dataset = RangeDataset([100] * 64)

    loader = build_loader(dataset, batch_size=32, num_workers=4, n_samples=1024, collate_fn=lambda samples: samples, sampling="patient_balanced", seed=17, patients_per_epoch=64, patient_group_size=7)

    assert isinstance(loader.sampler, PatientSampler)
    assert loader.sampler.patient_group_size == 7


def test_patient_sampler_rejects_datasets_without_patient_ranges():
    with pytest.raises(ValueError, match="does not expose get_patient_ranges"):
        build_patient_sampler(list(range(10)), num_samples=8, patients_per_epoch=2)
