# Preprocessors

This page documents the currently visible preprocessor layer used by Sleepwalker models.

## Interface

All preprocessors derive from `sleepwalker.models.preprocessors.Preprocessor.Preprocessor`.

Current interface expectations:

- input tensors use the repository convention `[B, T, C]`
- `requires_warmup()` declares whether running statistics must be collected
- `update(x)` accumulates warmup information
- calling the preprocessor transforms the tensor

Warmup is orchestrated by trainer code, not by the preprocessors themselves.

## Documented Preprocessors

The following preprocessors are currently documented:

- `Normalize`
- `NormalizeAlongDim`
- `Spectrogram`
- `Crop`
- `EmpiricalClipScaler`
- `RobustScaler`

## Stateful vs Stateless

Stateless preprocessors:

- `Spectrogram`
- `NormalizeAlongDim`
- `Crop`

Stateful preprocessors that require warmup:

- `Normalize`
- `EmpiricalClipScaler`
- `RobustScaler`

## Observed Semantics

- `Normalize` uses running mean and variance statistics
- `NormalizeAlongDim` normalizes per call along a chosen dimension
- `Spectrogram` converts time-domain data to log-magnitude spectrograms
- `Crop` extracts a fixed-length segment from each sequence
- `EmpiricalClipScaler` clips to learned quantile bounds and rescales to `[0, 1]`
- `RobustScaler` centers and scales using running median/IQR-style estimates

## Test Coverage

`tests/test_preprocessors.py` gives the clearest current contract for these modules:

- shape preservation where applicable
- warmup requirements
- basic numerical stability
- crop behavior
- CPU/CUDA consistency for selected preprocessors

## Caveat

The preprocessor layer is consistent and useful, but still internal. It should be documented as current model infrastructure rather than as a stable standalone API.
