"""DataLoader construction, including independent repeated views."""

from functools import partial

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.training.samplers import PatientSampler


def collapse_repeated_value(key: str, value, n_repeat: int):
    count = value.shape[0] if isinstance(value, torch.Tensor) else len(value)
    if count % n_repeat != 0:
        raise ValueError(f"Batch field '{key}' contains {count} values, which is not divisible by n_repeat={n_repeat}.")
    batch_size = count // n_repeat
    if isinstance(value, torch.Tensor):
        grouped = value.reshape(batch_size, n_repeat, *value.shape[1:])
        if key == "data":
            return grouped
        if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
            raise ValueError(f"Repeated target field '{key}' differs between views.")
        return grouped[:, 0]

    grouped = [value[index:index + n_repeat] for index in range(0, count, n_repeat)]
    if any(any(item != group[0] for item in group[1:]) for group in grouped):
        raise ValueError(f"Repeated metadata field '{key}' differs between views.")
    collapsed = [group[0] for group in grouped]
    return tuple(collapsed) if isinstance(value, tuple) else collapsed


def collate_repeated(samples, *, collate_fn, n_repeat: int):
    batch = collate_fn(samples)
    if "data" not in batch:
        raise ValueError("Repeated batches require a 'data' field.")
    return {key: collapse_repeated_value(key, value, n_repeat) for key, value in batch.items()}


def build_loader(dataset, *, batch_size: int, num_workers: int, n_samples: int | None, collate_fn, shuffle: bool, seed: int, n_repeat: int = 1, patients_per_epoch: int | None = None, patient_group_size: int | None = None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if n_repeat < 1:
        raise ValueError("n_repeat must be positive.")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if patients_per_epoch is not None:
        if not shuffle:
            raise ValueError("PatientSampler is only supported for shuffled training loaders.")
        sampler = PatientSampler(dataset, num_samples=n_samples, patients_per_epoch=patients_per_epoch, patient_group_size=patient_group_size, generator=generator)
    elif patient_group_size is not None:
        raise ValueError("patient_group_size requires patients_per_epoch.")
    elif n_samples is not None and len(dataset) > n_samples:
        sampler = RandomSampler(dataset, num_samples=n_samples, generator=generator)
    elif shuffle:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    effective_collate = collate_fn
    effective_batch_size = batch_size
    if n_repeat > 1:
        sampler = RepeatSampler(sampler, n_repeat=n_repeat)
        effective_batch_size = batch_size * n_repeat
        effective_collate = partial(collate_repeated, collate_fn=collate_fn, n_repeat=n_repeat)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": effective_batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": effective_collate,
        "drop_last": False,
        "pin_memory": True,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)
