# Ruhrland Comparison

This page documents the current comparison utility in `compare_predictions_ruhrland.py`.

## Purpose

The script compares prediction outputs from `predict.py` against Ruhrland sidecar annotations and writes CSV summaries.

## Inputs

The current script expects:

- one or more `.swmodel` files
- EDF files
- either explicit prediction files or a prediction directory

It loads prediction-package metadata to infer:

- task labels
- event mappings
- prediction window durations

## Comparison Logic

The script currently computes:

- exact interval matches
- overlap duration in seconds
- per-class metrics
- per-task aggregates
- overlap matrices across predicted and ground-truth labels

Tests in `tests/test_compare_predictions_ruhrland.py` cover the core interval merge and overlap behavior.

## Outputs

The script writes:

- `per_class_aggregate.csv`
- `per_task_aggregate.csv`
- `overlap_matrix_aggregate.csv`
- `per_class_by_file.csv`
- `per_task_by_file.csv`
- `overlap_matrix_by_file.csv`

## Caveat

This is a useful internal evaluation tool, but it is specialized to Ruhrland annotation handling and the current `predict.py` output format.
