# Python API

This reference is generated from Sleepwalker's Python signatures, type annotations, and docstrings. It documents the public objects used throughout the guides; internal helpers are intentionally omitted.

Sleepwalker does not currently expose a public dataset-download function. See the [roadmap](../roadmap.md).

## Datasets

### `ChannelConfig` {#channel-config}

::: sleepwalker.datasets.Basedataset.ChannelConfig

### `BaseDataset` {#base-dataset}

<span id="dataset-signal-settings"></span>

::: sleepwalker.datasets.Basedataset.BaseDataset
    options:
      show_root_heading: false
      members:
        - initialize
        - __getitem__
        - get_n_patients
        - get_timeseries_len
        - get_patient_ranges
        - get_input_channels
        - get_classes
        - input_spec
        - set_rejection_strategy
        - set_n_views

### `SleepEDFx` {#sleep-edfx}

::: sleepwalker.datasets.SleepEDFx.SleepEDFx
    options:
      show_root_heading: false
      members:
        - get_event_df

### `UnlabelledDataset` {#unlabelled-dataset}

::: sleepwalker.datasets.UnlabelledDataset.UnlabelledDataset
    options:
      show_root_heading: false
      members:
        - from_dataset
        - clone
        - dataset_kwargs

### Dataset utilities

<span id="get-edf-files-in-repo"></span>

::: sleepwalker.datasets.utils.get_edf_files_in_repo

::: sleepwalker.datasets.Basedataset.batch_collate

### `prepare_tensor_sample()` {#prepare-tensor-sample}

::: sleepwalker.datasets.Basedataset.prepare_tensor_sample
    options:
      show_root_heading: false

## Data loading and targets

### `build_loader()` {#build-loader}

::: sleepwalker.training.loader.build_loader
    options:
      show_root_heading: false

### `prepare_multiclass_target()` {#prepare-multiclass-target}

::: sleepwalker.trainer.utils.targets.prepare_multiclass_target
    options:
      show_root_heading: false

### `prepare_multitask_target()` {#prepare-multitask-target}

::: sleepwalker.trainer.utils.targets.prepare_multitask_target
    options:
      show_root_heading: false

## Training

### `MulticlassTrainer` {#multiclass-trainer}

::: sleepwalker.trainer.MulticlassTrainer.MulticlassTrainer
    options:
      show_root_heading: false
      members:
        - fit
        - predict
        - test
        - classification_contract

### `RunCfg` {#run-cfg}

::: sleepwalker.trainer.Run.RunCfg
    options:
      show_root_heading: false

### `RunResult` {#run-result}

::: sleepwalker.trainer.Run.RunResult
    options:
      show_root_heading: false

### `run()` {#run}

::: sleepwalker.trainer.Run.run
    options:
      show_root_heading: false

### Patient callbacks {#patient-callbacks}

::: sleepwalker.training.callbacks.trim_patient_to_events

::: sleepwalker.training.callbacks.prepare_patient_events

### `class_weights_for_loss()` {#class-weights-for-loss}

::: sleepwalker.trainer.losses.class_weights_for_loss
    options:
      show_root_heading: false

## Logging and telemetry

### `logger.batch_received()` {#batch-received}

::: sleepwalker.utils.UnifiedLogger.batch_received
    options:
      show_root_heading: false

The module-level `sleepwalker.utils.logger` forwards the batch to every registered sink as `sink.batch_received(batch, context)`. The logger supplies its current context; no phase, epoch or timing arguments are required. Multiple sinks can observe the same batch. Sinks must treat it as read-only and avoid retaining tensors.

`MulticlassTrainer` and `MultiLabelTrainer` emit one notification for each non-`None` batch at the beginning of their epoch loop, before device transfer and model execution. `BaseTrainer.fit()` and `test()` surround those loops with training/validation epoch or test context. Direct `run_epoch()` callers supply their own surrounding context. Warmup iteration does not emit batch notifications.

This hook reports delivery, not successful processing or an optimizer step. It does not measure loading or GPU computation time. Built-in logging and resource-monitoring sinks ignore it; custom sinks may inspect it. Notification errors propagate to the caller.

### `TelemetrySink` {#telemetry-sink}

::: sleepwalker.telemetry.TelemetrySink
    options:
      show_root_heading: false
      members:
        - __init__

See [Optional telemetry](../how-to/runcfg-cli.md#optional-telemetry) for native runner configuration.

## Model packages

### `PackagedModel` {#packaged-model}

::: sleepwalker.deployment.package.PackagedModel
    options:
      show_root_heading: false
      members:
        - capabilities
        - assert_compatible
        - predict_dataset
        - save
        - load

### `PackagedModel.predict_patient()` {#predict-patient}

::: sleepwalker.deployment.package.PackagedModel.predict_patient
    options:
      show_root_heading: false

### `load_packaged_model()` {#load-packaged-model}

::: sleepwalker.deployment.package.load_packaged_model

### `save_packaged_model()` {#save-packaged-model}

::: sleepwalker.deployment.package.save_packaged_model
