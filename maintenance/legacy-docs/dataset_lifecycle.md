# Dataset Lifecycle

This page summarizes the current dataset lifecycle used throughout the repository.

## Core Flow

The main dataset flow is:

1. configure a dataset object with channels, timing, mappings, and callbacks
2. optionally inspect patient-level statistics
3. initialize the dataset on a chosen patient list
4. sample sliding windows lazily during training or inference

This flow is implemented primarily by `BaseDataset`.

## Main Hooks

`BaseDataset` currently exposes three important callback stages:

- `prepare_patient`
- `prepare_target`
- `prepare_sample`

Observed intent:

- `prepare_patient` is for whole-recording filtering or trimming
- `prepare_target` is for cheap label-based rejection and target building
- `prepare_sample` is for final tensor conversion and signal-quality checks

Tests confirm that `prepare_target` can reject windows before signal loading, which is important for performance.

## Labelled vs Unlabelled

Two important dataset forms now documented:

- labelled EDF-backed datasets derived from `BaseDataset`
- `UnlabelledDataset`, which keeps the signal-loading path but disables label extraction

`UnlabelledDataset` is used by prediction packages and inference helpers.

## Cached Datasets

`NumpyDataset` represents a different lifecycle stage:

- a realized dataloader pass has already been exported
- the cache is then replayed as a dataset-like object

This is a frozen data representation rather than a live EDF-backed dataset.

## Dataset-Specific Adapters

Adapters such as `Ruhrlandklinik`, `SleepEDFx`, `CAP`, and others implement dataset-specific event parsing on top of the shared lifecycle.

Current documentation should treat those adapters as integrations, not as generic file-format parsers.
