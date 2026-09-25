# Multi-Modeling

This page describes the current composite-model layer built around `MultiModel` and `MetaModel`.

## `MetaModelEntry`

`MetaModelEntry` pairs:

- a submodel implementing the `BaseModel` interface
- a list of input channel names that should be routed to that submodel

This is the routing unit used by both `MultiModel` and `MetaModel`.

## `MultiModel`

`sleepwalker.models.MultiModel.MultiModel` fuses embeddings from multiple submodels and applies one shared linear head.

Observed behavior from code and tests:

- the parent model receives the full `[B, T, C]` tensor
- each submodel gets only its configured channel slice
- each submodel must return a 2D embedding `[B, E]`
- embeddings are concatenated along the feature dimension
- a shared linear head maps the fused embedding to one multiclass output

This pattern is used in current task scripts such as `train_noisy.py` and `train_arousal.py`.

## `MetaModel`

`sleepwalker.models.MetaModel.MetaModel` reuses the same embedding fusion path but replaces the shared head with one head per task.

Each task config entry is expected to provide at least:

- `labels`
- `n_steps`

The output for each task is shaped:

- `[B, n_steps, n_labels]`

This matches the expectations of `MultiLabelTrainer`.

## Preprocessor Warmup

Composite models support two warmup layers:

- preprocessors attached to the composite model itself
- preprocessors attached to individual submodels

`tests/test_metamodel.py` confirms that both layers can be warmed and that channel slicing still happens correctly.

## Current Constraints

- Channel routing is name-based, not schema-based.
- The composite model assumes submodels agree on the general `[B, T, C]` input convention.
- Task config normalization is handled outside the model, typically by `MultiLabelTrainer.normalize_task_config(...)`.
