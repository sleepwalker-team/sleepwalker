"""DataLoader construction with explicit candidate-index policies."""

import math
from functools import partial

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sleepwalker.training.samplers import PatientSampler
from sleepwalker.utils import logger


PATIENT_GROUP_DIVERSITY_FLOOR = 16
PATIENT_GROUP_PREFETCH_WINDOWS = 1
PREFETCH_FACTOR = 2


def derive_patient_group_size(*, num_samples: int, patients_per_epoch: int, batch_size: int, num_workers: int) -> int:
    """Choose a patient locality group that stays active across prefetched work."""
    batches_per_prefetch_window = PREFETCH_FACTOR * num_workers if num_workers > 0 else 1
    target_batches_per_group = PATIENT_GROUP_PREFETCH_WINDOWS * batches_per_prefetch_window
    group_size_for_prefetch = math.ceil(target_batches_per_group * batch_size * patients_per_epoch / num_samples)
    diversity_floor = min(PATIENT_GROUP_DIVERSITY_FLOOR, batch_size, patients_per_epoch)
    return min(patients_per_epoch, max(diversity_floor, group_size_for_prefetch))


def collate_valid_samples(samples, *, collate_fn, drop_incomplete: bool):
    valid_samples = [sample for sample in samples if sample is not None]
    if len(valid_samples) == 0 or (drop_incomplete and len(valid_samples) != len(samples)):
        return None
    return collate_fn(valid_samples)


def configure_dataset_execution(dataset, *, rejection_strategy: str | None, n_views: int) -> None:
    if rejection_strategy is not None:
        if not hasattr(dataset, "set_rejection_strategy"):
            raise TypeError(f"{dataset.__class__.__name__} does not support rejection strategies.")
        dataset.set_rejection_strategy(rejection_strategy)
    if hasattr(dataset, "set_n_views"):
        dataset.set_n_views(n_views)
    elif n_views != 1:
        raise TypeError(f"{dataset.__class__.__name__} does not support repeated views.")


def build_loader(dataset, *, batch_size: int, num_workers: int, n_samples: int | None, collate_fn, sampling: str, seed: int, patients_per_epoch: int | None = None, patient_group_size: int | None = None, rejection_strategy: str | None = None, n_views: int = 1, drop_last: bool = False, pin_memory: bool = True, persistent_workers: bool = True):
    """Build a loader with explicit index sampling and dataset execution policies.

    ``n_samples`` controls how many candidate indices are requested. Expected
    dataset rejection is handled afterwards by ``collate_valid_samples``.
    ``drop_last`` therefore drops both a short final candidate batch and any
    batch made incomplete by rejected samples.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if n_samples is not None and n_samples < 1:
        raise ValueError("n_samples must be positive when provided.")
    if n_views < 1:
        raise ValueError("n_views must be positive.")
    if sampling not in {"sequential", "random", "patient_balanced"}:
        raise ValueError("sampling must be 'sequential', 'random', or 'patient_balanced'.")
    configure_dataset_execution(dataset, rejection_strategy=rejection_strategy, n_views=n_views)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if sampling == "patient_balanced":
        if patients_per_epoch is None:
            raise ValueError("patient_balanced sampling requires patients_per_epoch.")
        sampler = PatientSampler(dataset, num_samples=n_samples, patients_per_epoch=patients_per_epoch, patient_group_size=patient_group_size, generator=generator)
        if patient_group_size is None:
            sampler.patient_group_size = derive_patient_group_size(
                num_samples=sampler.num_samples,
                patients_per_epoch=sampler.patients_per_epoch,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            logger.info(
                f"Automatically selected patient_group_size={sampler.patient_group_size} for "
                f"patients_per_epoch={sampler.patients_per_epoch}, batch_size={batch_size}, "
                f"num_workers={num_workers}, and num_samples={sampler.num_samples}."
            )
    elif patients_per_epoch is not None or patient_group_size is not None:
        raise ValueError("patients_per_epoch and patient_group_size require patient_balanced sampling.")
    elif sampling == "random":
        sampler = RandomSampler(dataset, num_samples=n_samples, generator=generator) if n_samples is not None else RandomSampler(dataset, generator=generator)
    else:
        sample_count = min(len(dataset), n_samples) if n_samples is not None else len(dataset)
        sampler = SequentialSampler(range(sample_count))

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": partial(collate_valid_samples, collate_fn=collate_fn, drop_incomplete=drop_last),
        "drop_last": drop_last,
        "pin_memory": bool(pin_memory),
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**loader_kwargs)
