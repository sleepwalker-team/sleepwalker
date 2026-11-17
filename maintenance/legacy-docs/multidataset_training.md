# Multi-Dataset Training

This page documents the current dataset-composition path used for some training workflows.

## `MultiDataset`

`sleepwalker.datasets.MultiDataset.MultiDataset` concatenates several initialized datasets into one dataset-like object.

Current enforced constraints:

- matching `sample_frequency`
- matching `target_resolution`
- matching `total_input`
- matching `stride`
- matching class sets

Each returned item includes an extra `dataset` field identifying the source dataset index.

## Why It Exists

Observed current uses:

- training across several datasets through one shared trainer
- domain-adversarial training where the source dataset ID becomes a supervision signal

## Combined Dataset Identity

The `dataset` field matters because some trainers use it directly.

Most notably:

- `GradReverseTrainer` treats dataset identity as a domain label

## Caveat

`MultiDataset` is a practical concatenation layer, not a full multi-domain experiment framework. It assumes strong compatibility between the combined datasets.
