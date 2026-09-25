# Module Overview

This page groups the repository into practical subsystems rather than treating every file as equally central.

## `sleepwalker.core`

Core EDF utilities. Current evidence points to `signal.py` as the main module for EDF metadata reading, signal conversion, and EDF header repair.

## `sleepwalker.datasets`

Dataset ingestion and sample construction.

Important parts:

- `Basedataset.py`: core dataset lifecycle, patient preparation, window indexing, lazy item loading
- dataset adapters such as `CAP.py`, `ISRUC.py`, `SHHS.py`, `SleepEDFx.py`, `Ruhrlandklinik.py`
- `NumpyDataset.py` and `UnlabelledDataset.py`: cache/export/inference-oriented dataset forms
- `utils.py`: EDF discovery, splits, export helpers, XML readers
- `normalizer/`: channel-level normalizers
- `augmentation/`: optional training-time signal transforms

## `sleepwalker.models`

PyTorch model definitions.

Important parts:

- `Basemodel.py`: shared preprocessor-aware base class
- architecture files such as `SleepTransformer.py`, `USleep.py`, `AttnSleep.py`
- `MetaModel.py` and `MultiModel.py`: fused or task-specific multi-head models
- `preprocessors/`: tensor preprocessors with optional warmup behavior

## `sleepwalker.trainer`

Training, evaluation, and prediction helpers.

Important parts:

- `BaseTrainer.py`: common fit/test/predict mechanics
- `Run.py`: shared experiment runner used by training scripts
- `MulticlassTrainer.py`: single-head categorical tasks
- `MultiLabelTrainer.py`: multi-task categorical prediction
- `losses.py`: class weighting and multitask masking helpers
- `utils/`: targets, metrics, filtering, splits, disk helpers, confusion-table rendering

## `sleepwalker.deployment`

Prediction-package export and loading. This is the bridge between training-time code and inference-time package consumption.

## Top-Level Scripts

The top level currently mixes maintained scripts and more experimental utilities.

Most important observed scripts:

- `train_sleep.py`
- `train_multilabel.py`
- other `train_*.py` task scripts
- `predict.py`
- `compare_predictions_ruhrland.py`

These scripts are useful documentation sources because they show how the internal modules are actually composed. They should not yet be treated as a polished external CLI layer.
