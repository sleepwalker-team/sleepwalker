# Ruhrland Dataset Notes

This page summarizes what is currently clear about `sleepwalker.datasets.Ruhrlandklinik`.

## Scope

`Ruhrlandklinik` is a dataset adapter for EDF recordings with sidecar annotation files. It is used by multiple task scripts in the repository, including sleep staging, arousal, desaturation, body position, noisy-window, and multitask Ruhrland experiments.

## Annotation Sources

The current adapter reads:

- main annotations from `*_MS.xls` or `*_MS.xlsx`
- optional auxiliary annotations from `*_AS.csv`

The exact naming convention is hard-coded in the adapter.

## Main Annotation Handling

Observed behaviors in `get_event_df(...)`:

- spreadsheet rows are normalized to lowercase event labels
- `lm` intervals are merged when they overlap
- some non-sleep events are removed when they overlap with merged wake intervals
- the result is converted into a DataFrame with `Label`, `Starttime`, `Endtime`, and `Duration`

These are explicit code paths. The broader clinical or annotation-policy rationale is not fully documented in the repository.

## Auxiliary Annotation Handling

When `return_nox=True`, the adapter reads `_AS.csv` and exposes it through `get_extra_event_df(...)`.

Observed behaviors:

- a small encoding-fix map is applied for known broken labels
- rows are normalized into the same event-table shape
- body-position events are expanded from change markers into intervals

## Body Position Expansion

Tests in `tests/test_ruhrlandklinik.py` confirm:

- body-position changes are converted into interval events
- duplicate consecutive changes are ignored
- an initial position can be seeded through `initial_body_position_event`

This is one of the clearest, directly tested pieces of Ruhrland-specific behavior.

## Caution

This adapter is heavily tied to local Ruhrland annotation conventions. It should be documented and maintained as a lab-specific integration surface, not as a generic spreadsheet parser.
