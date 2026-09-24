# Data pipeline components

The building blocks that sit between an EDF file and a model tensor: normalizers, augmentation, samplers, sample/patient callbacks, file selection, and the evaluation helpers that turn confusion matrices into metrics. Dataset classes compose these; the [Python API](api.md) documents the dataset and run objects that use them.

## Normalizers

A normalizer maps a raw `[N, 1]` physical-channel array to a processed array of the same shape. It is attached to a [`ChannelConfig`](api.md#channel-config) and runs after the channel is read and unit-converted. See [Load and prepare data](../how-to/data.md) for where normalizers sit in the pipeline.

### `Normalizer` {#normalizer}

::: sleepwalker.datasets.normalizer.base.Normalizer
    options:
      show_root_heading: false

### `SignalFilterNormalizer` {#signal-filter-normalizer}

::: sleepwalker.datasets.normalizer.SignalFilterNormalizer.SignalFilterNormalizer
    options:
      show_root_heading: false

### `EEGFilterNormalizer` {#eeg-filter-normalizer}

::: sleepwalker.datasets.normalizer.EEGFilterNormalizer.EEGFilterNormalizer
    options:
      show_root_heading: false

### `PulseFilterNormalizer` {#pulse-filter-normalizer}

::: sleepwalker.datasets.normalizer.PulseFilterNormalizer.PulseFilterNormalizer
    options:
      show_root_heading: false

### `RespirationFilterNormalizer` {#respiration-filter-normalizer}

::: sleepwalker.datasets.normalizer.RespirationFilterNormalizer.RespirationFilterNormalizer
    options:
      show_root_heading: false

### `SaturationFilterNormalizer` {#saturation-filter-normalizer}

::: sleepwalker.datasets.normalizer.SaturationFilterNormalizer.SaturationFilterNormalizer
    options:
      show_root_heading: false

## Augmentation

Random transforms applied to a sample during training and never at inference. Each is a plain callable that takes a `pandas.DataFrame` of window samples (rows are time steps, columns are channels) and returns a transformed frame. Wire one into a dataset through its augmentation callback; see [Load and prepare data](../how-to/data.md).

### `AmplitudeScale` {#amplitude-scale}

::: sleepwalker.datasets.augmentation.AmplitudeScale.AmplitudeScale
    options:
      show_root_heading: false

### `ChannelDropout` {#channel-dropout}

::: sleepwalker.datasets.augmentation.ChannelDropout.ChannelDropout
    options:
      show_root_heading: false

### `ChannelShift` {#channel-shift}

::: sleepwalker.datasets.augmentation.ChannelShift.ChannelShift
    options:
      show_root_heading: false

### `FrequencyNoise` {#frequency-noise}

::: sleepwalker.datasets.augmentation.FrequencyNoise.FrequencyNoise
    options:
      show_root_heading: false

### `GaussianNoise` {#gaussian-noise}

::: sleepwalker.datasets.augmentation.GaussianNoise.GaussianNoise
    options:
      show_root_heading: false

### `RandomPolarityFlip` {#random-polarity-flip}

::: sleepwalker.datasets.augmentation.RandomPolarityFlip.RandomPolarityFlip
    options:
      show_root_heading: false

### `RandomResampleJitter` {#random-resample-jitter}

::: sleepwalker.datasets.augmentation.RandomResampleJitter.RandomResampleJitter
    options:
      show_root_heading: false

### `TimeShiftAndCrop` {#time-shift-and-crop}

::: sleepwalker.datasets.augmentation.TimeShiftAndCrop.TimeShiftAndCrop
    options:
      show_root_heading: false

## Samplers

### `PatientSampler` {#patient-sampler}

::: sleepwalker.training.samplers.PatientSampler
    options:
      show_root_heading: false

## Sample and patient callbacks

Callbacks hook into dataset preparation. `prepare_patient` runs once per patient at initialization; `prepare_target` runs per window. See [Load and prepare data](../how-to/data.md).

### `trim_patient_to_events()` {#trim-patient-to-events}

::: sleepwalker.training.callbacks.trim_patient_to_events
    options:
      show_root_heading: false

### `prepare_patient_events()` {#prepare-patient-events}

::: sleepwalker.training.callbacks.prepare_patient_events
    options:
      show_root_heading: false

### `robust_scaler()` {#robust-scaler-callback}

::: sleepwalker.training.callbacks.robust_scaler
    options:
      show_root_heading: false

## File selection

Helpers that resolve directories into the EDF files a dataset should use, and split patient lists deterministically.

### `filter_edf_files()` {#filter-edf-files}

::: sleepwalker.training.files.filter_edf_files
    options:
      show_root_heading: false

### `edf_filter_result()` {#edf-filter-result}

::: sleepwalker.training.files.edf_filter_result
    options:
      show_root_heading: false

### `get_edf_files_in_repo()` {#get-edf-files-in-repo}

::: sleepwalker.datasets.utils.get_edf_files_in_repo
    options:
      show_root_heading: false

### `random_split()` {#random-split}

::: sleepwalker.datasets.utils.random_split
    options:
      show_root_heading: false

## Metrics

Confusion-matrix helpers used by the trainers to report classification quality.

### `validate_confusion_matrix()` {#validate-confusion-matrix}

::: sleepwalker.metrics.validate_confusion_matrix
    options:
      show_root_heading: false

### `accuracy_from_confusion_matrix()` {#accuracy-from-confusion-matrix}

::: sleepwalker.metrics.accuracy_from_confusion_matrix
    options:
      show_root_heading: false

### `precision_from_confusion_matrix()` {#precision-from-confusion-matrix}

::: sleepwalker.metrics.precision_from_confusion_matrix
    options:
      show_root_heading: false

### `recall_from_confusion_matrix()` {#recall-from-confusion-matrix}

::: sleepwalker.metrics.recall_from_confusion_matrix
    options:
      show_root_heading: false

### `f1_per_class_from_confusion_matrix()` {#f1-per-class-from-confusion-matrix}

::: sleepwalker.metrics.f1_per_class_from_confusion_matrix
    options:
      show_root_heading: false

### `support_from_confusion_matrix()` {#support-from-confusion-matrix}

::: sleepwalker.metrics.support_from_confusion_matrix
    options:
      show_root_heading: false

### `cohen_kappa_from_confusion_matrix()` {#cohen-kappa-from-confusion-matrix}

::: sleepwalker.metrics.cohen_kappa_from_confusion_matrix
    options:
      show_root_heading: false

### `f1_score_from_confusion_matrix()` {#f1-score-from-confusion-matrix}

::: sleepwalker.metrics.f1_score_from_confusion_matrix
    options:
      show_root_heading: false

## Prediction transforms

Post-processing functions applied to a long-format prediction `DataFrame` before evaluation.

### `apply_pipeline()` {#apply-pipeline}

::: sleepwalker.prediction_transforms.apply_pipeline
    options:
      show_root_heading: false

### `filter_patient_signals()` {#filter-patient-signals}

::: sleepwalker.prediction_transforms.filter_patient_signals
    options:
      show_root_heading: false

### `filter_annotation()` {#filter-annotation}

::: sleepwalker.prediction_transforms.filter_annotation
    options:
      show_root_heading: false

### `resolve_overlaps()` {#resolve-overlaps}

::: sleepwalker.prediction_transforms.resolve_overlaps
    options:
      show_root_heading: false

### `smooth_probabilities()` {#smooth-probabilities}

::: sleepwalker.prediction_transforms.smooth_probabilities
    options:
      show_root_heading: false
