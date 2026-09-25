# Run Artifacts

This page summarizes the lightweight artifact formats used during training runs.

## Jsonl Records

`sleepwalker.trainer.utils.disk.append_to_jsonl(...)` writes newline-delimited JSON records.

Current uses include:

- test-result summaries emitted by `Run.py`
- experiment tracking outside heavier MLflow-style sinks

The helper normalizes several NumPy and pandas scalar types before writing.

## Checkpoints

`sleepwalker.trainer.utils.disk.store_checkpoint(...)` stores:

- `model.pt`
- `optimizer.pt`
- optionally `scheduler.pt`

These checkpoint folders are used by `BaseTrainer` for:

- periodic intermediate checkpoints
- best-validation-model snapshots

## Current Scope

These are pragmatic lab-internal artifacts:

- simple enough for current workflows
- not versioned as a standalone artifact spec
- documented here to explain repository behavior, not to promise long-term compatibility
