# Trainers and targets

A trainer owns everything around the model's forward pass: the loss function, the optimizer and learning-rate schedule, the epoch and validation cadence, early stopping, checkpointing, and the metrics a run reports. The user-facing guide is [Train a multiclass model](../how-to/train-multiclass.md); this page renders the generated reference for the trainer classes, the target-preparation callbacks, and the loss helpers.

## Trainer classes

### `BaseTrainer` {#base-trainer}

::: sleepwalker.trainer.BaseTrainer.BaseTrainer
    options:
      show_root_heading: false
      members:
        - fit
        - test
        - run_epoch
        - warmup_trainer
        - set_loader_epoch
        - validate_training_state
        - save_checkpoint
        - load_checkpoint

### `MulticlassTrainer` {#multiclass-trainer}

::: sleepwalker.trainer.MulticlassTrainer.MulticlassTrainer
    options:
      show_root_heading: false
      members:
        - fit
        - test
        - classification_contract

### `MultiLabelTrainer` {#multilabel-trainer}

::: sleepwalker.trainer.MultiLabelTrainer.MultiLabelTrainer
    options:
      show_root_heading: false
      members:
        - fit
        - test
        - classification_contract
        - run_epoch

### `MaskedAutoencoderTrainer` {#masked-autoencoder-trainer}

::: sleepwalker.trainer.MaskedAutoencoderTrainer.MaskedAutoencoderTrainer
    options:
      show_root_heading: false

## Target preparation callbacks

These functions plug into the dataset's `prepare_target` slot. They receive the label-activity window (`target`, optional `target_extra`) and return a prepared payload dictionary or `None` to reject the candidate. See [Loading data with labels](../how-to/data.md#loading-data-with-labels) and [Multilabel targets](../how-to/multilabel.md).

### `prepare_multiclass_target()` {#prepare-multiclass-target}

::: sleepwalker.trainer.utils.targets.prepare_multiclass_target
    options:
      show_root_heading: false

### `prepare_multitask_target()` {#prepare-multitask-target}

::: sleepwalker.trainer.utils.targets.prepare_multitask_target
    options:
      show_root_heading: false

### `build_multitask_target()` {#build-multitask-target}

::: sleepwalker.trainer.utils.targets.build_multitask_target
    options:
      show_root_heading: false

### `normalize_multitask_config()` {#normalize-multitask-config}

::: sleepwalker.trainer.utils.targets.normalize_multitask_config
    options:
      show_root_heading: false

### `prepare_single_target()` {#prepare-single-target}

::: sleepwalker.trainer.utils.targets.prepare_single_target
    options:
      show_root_heading: false

### `slice_target_interval()` {#slice-target-interval}

::: sleepwalker.trainer.utils.targets.slice_target_interval
    options:
      show_root_heading: false

### `build_multiclass_sequence()` {#build-multiclass-sequence}

::: sleepwalker.trainer.utils.targets.build_multiclass_sequence
    options:
      show_root_heading: false

### `build_sequence_mask()` {#build-sequence-mask}

::: sleepwalker.trainer.utils.targets.build_sequence_mask
    options:
      show_root_heading: false

### `passes_filters()` {#passes-filters}

::: sleepwalker.trainer.utils.targets.passes_filters
    options:
      show_root_heading: false

### `annotation_coverage()` {#annotation-coverage}

::: sleepwalker.trainer.utils.targets.annotation_coverage
    options:
      show_root_heading: false

## Loss helpers

### `class_weights_for_loss()` {#class-weights-for-loss}

::: sleepwalker.trainer.losses.class_weights_for_loss
    options:
      show_root_heading: false

### `build_multilabel_task_masks()` {#build-multilabel-task-masks}

::: sleepwalker.trainer.losses.build_multilabel_task_masks
    options:
      show_root_heading: false

### `estimate_multilabel_class_cnts()` {#estimate-multilabel-class-cnts}

::: sleepwalker.trainer.losses.estimate_multilabel_class_cnts
    options:
      show_root_heading: false
