# Architecture Overview

This page summarizes the repository structure from observed code and tests. It intentionally favors accurate, narrow claims over a complete system description.

## Core Flow

The central workflow is:

1. EDF metadata and signal windows are read through `sleepwalker/core/signal.py`.
2. A dataset adapter derived from `sleepwalker.datasets.Basedataset.BaseDataset` maps raw labels into canonical task labels and exposes sliding-window samples.
3. A model from `sleepwalker/models/` consumes `[B, T, C]` tensors, optionally after preprocessor warmup.
4. A trainer from `sleepwalker/trainer/` runs fit, validation, test, and prediction helpers.
5. `sleepwalker/deployment/package.py` can export the trained model, trainer, and an unlabelled dataset template as a `.swmodel` bundle.
6. `predict.py` loads one or more exported bundles and writes prediction tables for EDF files.

## Dataset Layer

`BaseDataset` is the main lifecycle abstraction. It is responsible for:

- loading EDF-backed signals lazily
- mapping dataset-specific labels into canonical labels
- building a sliding-window index per patient
- calling optional hooks for patient filtering, target preparation, and final sample preparation

Tests confirm that target filtering can happen before signal loading, which is an important performance and design detail.

Dataset subclasses appear to fall into two groups:

- reusable adapters for named datasets such as `CAP`, `ISRUC`, `SHHS`, `SleepEDFx`, `Ruhrlandklinik`
- utility datasets such as `NumpyDataset`, `MultiDataset`, `UnlabelledDataset`, and possibly `ZarrDataset`

## Model Layer

The repository uses PyTorch models. `sleepwalker.models.Basemodel.BaseModel` defines the shared preprocessor-aware model interface. Concrete models include `SleepTransformer`, `AttnSleep`, `SeqSleepNet`, `TinySleepNet`, `USleep`, `MRASleepNet`, and `UTime`.

`MetaModel` and `MultiModel` support multi-branch or multi-task setups. Current evidence suggests that `MultiLabelTrainer` expects a `MetaModel`.

## Trainer Layer

`BaseTrainer` owns the common training loop, checkpoint handling, preprocessor warmup, repeated-window handling, and prediction helpers.

The main concrete trainers documented so far are:

- `MulticlassTrainer`: one categorical target per window
- `MultiLabelTrainer`: multiple categorical tasks, potentially at different temporal resolutions within one window

`sleepwalker.trainer.Run.run(...)` sits one level above individual trainers and coordinates loaders, training, test evaluation, jsonl logging, and optional package export.

## Deployment Layer

Deployment is package-based rather than source-based. A `.swmodel` bundle currently stores:

- model weights
- serialized model object
- serialized trainer object
- serialized unlabelled dataset template
- metadata and optional model-card markdown

Tests confirm package round-tripping and basic prediction-package export behavior.

## Unclear Or Still Evolving

- There is no formal boundary yet between stable library code and experiment scripts.
- Several top-level utilities and generated artifacts appear to be lab-internal or exploratory.
- The repository contains some untracked or actively changing files; this overview documents the observed structure, not a frozen release surface.
