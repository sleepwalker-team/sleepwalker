# Numpy Cache Format

This page documents the cache format used by `export_dataloader_to_numpy_dir(...)` and `NumpyDataset`.

## Purpose

The cache format freezes one realized pass over a dataloader into a directory of numpy arrays. It is useful when:

- live EDF-backed datasets are too expensive to re-read repeatedly
- one training or evaluation pass should be reused deterministically
- an unlabelled inference template should be reconstructed from cache metadata

## Required Files

A cache directory contains:

- `meta.json`
- `data.npy` or `data.000000.npy`, `data.000001.npy`, ...
- `time.npy`
- `patient.npy`

Optional files:

- `target.npy`
- `target_extra.npy`
- `extra__{key}.npy`

Arrays may be sharded across multiple files along axis 0.

## `meta.json`

Observed metadata fields include:

- `sample_frequency`
- `resample_type`
- `total_input`
- `target_resolution`
- `classes`
- `input_channels`
- `all_patients`
- `extra_keys`

Additional fields may appear over time, but the items above are the core fields currently consumed by `NumpyDataset`.

## `NumpyDataset`

`sleepwalker.datasets.NumpyDataset.NumpyDataset` reads this directory and exposes dataset-like access.

Observed behavior:

- can load fully in memory or through NumPy memmap
- reconstructs `data`, `patient`, `time`, and optional targets per item
- restores exported extra keys
- can build an `UnlabelledDataset` template from cache metadata

## What The Cache Represents

The export intentionally captures one realized dataloader pass, not an abstract dataset definition.

That means the cache already reflects:

- any sampler choices
- any grouped-channel randomness already realized
- any dataset-side filtering or shaping that happened before export

Tests in `tests/test_datasets.py` confirm this behavior, including the case where a grouped live dataset stays random while the cached export becomes stable.

## Limits

The cache is intentionally narrow:

- extra keys must have stable, numpy-serializable shapes
- arbitrary Python objects are not preserved
- ragged extra fields fail loudly instead of being pickled silently
