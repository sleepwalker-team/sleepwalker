# Target Construction

This page summarizes the target-building logic currently documented in the repository.

## Multiclass Targets

`sleepwalker.trainer.utils.targets` provides the multiclass target path.

Observed behavior:

- input is a time-indexed label-activity window
- optional filters can reject a window before target construction
- a class becomes active if it covers at least a configured fraction of the window
- if exactly one expected class is absent from the input columns, it may serve as an implicit fallback class
- ambiguous windows are rejected rather than force-assigned

Tests in `tests/test_targets.py` cover the current fallback and filtering behavior.

## Multitask Targets

`MultiLabelTrainer` provides the multitask path.

Observed behavior:

- each task defines its own labels and target resolution
- smaller task resolutions are expanded into several steps within the largest target window
- outputs are integer class indices shaped `[n_tasks, max_steps]`
- unresolved or ambiguous targets use `-1` internally and may cause rejection depending on the call path

## Conditioning

`sleepwalker.trainer.losses.build_multilabel_task_masks(...)` supports conditioned multitask training.

Observed behavior:

- one task can act as a gate for several other tasks
- only samples whose condition task contains selected labels contribute to the conditioned task losses

Tests in `tests/test_multilabel_trainer.py` cover the current masking and class-count behavior.

## Practical Interpretation

The target-building code is intentionally conservative. Current behavior favors dropping unclear windows over inventing targets.
