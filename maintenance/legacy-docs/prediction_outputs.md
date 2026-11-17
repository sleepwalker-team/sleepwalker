# Prediction Outputs

This page documents the current prediction-output reshaping path used by `predict.py`.

## Input To The Reshaping Step

Each loaded package produces a trainer-specific prediction frame.

Observed cases:

- multiclass trainers produce columns such as `prob__wake`
- multitask trainers produce task-scoped columns such as `task__prob__label` and `task__time`

`predict.py` normalizes those frames into a shared long-form representation.

## Long-Form Probability Records

The shared intermediate format has columns:

- `time`
- `group`
- `label`
- `prob`

This format is internal to the script but makes it easier to merge outputs from several models.

## Final Output

The final table is time-indexed and wide:

- one column per `group__label`
- values are one-hot winner indicators

If duplicate `time/group/label` rows occur, current code averages probabilities before picking the winner. This behavior is covered by `tests/test_predict.py`.

## Caveat

This output format is clear from code and tests, but it should still be treated as an internal current contract rather than a stable public prediction schema.
