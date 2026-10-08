# How-to guides

These guides each solve one task. Install [Sleepwalker](../getting-started/index.md) first; the examples use [Sleep-EDFx](sleep-edfx.md) as a concrete dataset.

!!! info
    Sleepwalker currently supports EDF recordings.

**Data**

- [Load and prepare data](data.md) — channels, units, windowing, labels, and normalization.

**Training (API first)**

- [Models: classification and embeddings](models.md) — choosing and configuring a model.
- [Train and export a sleep-staging model](train-sleep-staging.md) — the canonical Python tutorial.
- [Train a multiclass model](train-multiclass.md) — the in-depth Python tutorial: targets, rejection, and class balance.
- [Train with multiple labels](multilabel.md) — multi-task and multi-label objectives.
- [Train from the CLI](train-from-cli.md) — turn the Python examples into a YAML config and run them.
- [RunCfg and the CLI](runcfg-cli.md) — every run field and CLI command.

**Model packages**

- [Model packages](packaged-model.md) — what a package is and why it carries a dataset.
- [Import and export model packages](package-import-export.md) — create and load packages.
- [Predict sleep stages](predict-patient.md) — score one recording.
- [Extract embeddings](extract-embeddings.md) — window-level features for one recording.
- [Train a head on rocket features](features.md) — MiniRocket, MultiRocket and ROCKET features as a frozen encoder package.
- [Inspect a model package](inspect-package.md) — read a package's settings and capabilities.

**Advanced**

- [Tune data loading performance](performance.md) — workers, thread limits, the `PatientSampler`, and in-memory caches.
- [How Sleepwalker works](overview.md) — the full data flow from EDF files to predictions.

**Troubleshooting**

- [Troubleshoot EDF and package errors](troubleshoot.md)
