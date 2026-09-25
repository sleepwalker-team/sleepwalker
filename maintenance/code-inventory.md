# Code cleanup inventory

Date: 2026-09-11

This is a review queue, not a removal list. The structural refactor deliberately does not combine or delete candidates until their behavior and research ownership are known.

## Status labels

- **extract**: belongs to a paper or lab workflow rather than the reusable package.
- **review**: plausible duplication or obsolete surface; compare behavior and callers first.
- **remove after verification**: no current framework caller was found and the code already refers to removed APIs.

## Extraction candidates

| Candidate | Evidence | Next decision |
| --- | --- | --- |
| `iclr2026/` | Contains paper source, experiment queues, paper configs, generated tables, models, and paper-specific tests. | Move to its own repository and pin Sleepwalker. |
| `train_embedding_downstream.py` | Root-level experiment assembly with local environment assumptions. | Identify owning paper; extract or replace with a small reusable example. |
| `train_maskedembedding_hsp.py` | Root-level HSP experiment with duplicated dataset/loader construction. | Extract to the owning experiment repository. |
| `train_maskedembedding_ruhrland.py` | Root-level Ruhrland experiment with local paths and custom orchestration. | Extract to the owning experiment repository. |
| `tools/fix_headers.py` | Executes at import time and contains a hard-coded `/raid/sleepwalker/ruhrlandklinik/raw` path. | Move to a lab repository or replace with a parameterized API/CLI only if still used. |

## Stale or unused candidates

| Candidate | Evidence | Proposed disposition |
| --- | --- | --- |
| `test_ruhrland.py` | Imports `load_models`, `predict_dataset`, and `sleepwalker.trainer.utils.metrics`, which are absent from the current package. | Remove after confirming that current evaluation commands cover the workflow. |
| `tools/predict.py` | Contains an older table-normalization API and is only imported by `tests/test_predict.py`; inference now uses `PackagedModel.predict_patient()`. | Decide whether its output conversion is still needed, then move the useful transformation into the package or remove it with its old test. |
| `configs/old/` | 26 configurations are explicitly stored as old and are excluded from the public documentation. | Extract paper-owned configs; delete configs with no reproducibility owner. |
| `maintenance/legacy-docs/` | 24 pre-refactor pages include obsolete `.swmodel`, trainer-pickle, `MetaModel`, `MultiModel`, and root `predict.py` descriptions. | Migrate verified reasoning page by page; never publish directly. |
| Root compatibility wrappers in `tools/` | Six wrappers now delegate to `sleepwalker.cli` after installation. | Remove after internal usage has moved to the `sleepwalker` entry point. |

## Duplication candidates

The scan compared top-level public symbol names; matching names are leads, not proof of equivalent behavior.

| Area | Candidates | Observation |
| --- | --- | --- |
| Transformer utilities | `models/utils.py` and `models/SleepTransformer.py` both define `AttentionPooling` and `SinusoidalPositionalEncoding`. | Implementations are substantially duplicated; the generic copy supports a configurable pooling dimension. |
| Download/checksum helpers | `datasets/DCSM.py` and `datasets/SleepEDFx.py` both define `validate_sha256`, `download_and_validate`, and `download_dataset`; ISRUC has another `download_dataset`. | Likely reusable download mechanics mixed into adapters. Keep separate until retry, checksum, and archive behavior are compared. |
| XML parsing | ABC, HCHS, MNC, MROS, Numom2b, and SHHS each define `read_xml`. | Some share NSRR/Profusion structure, but dataset-specific differences may be scientifically meaningful. |
| Datetime conversion | Apples, CAP, and SVUH_UCD each define `convert_to_datetime`. | Compare accepted formats and timezone assumptions before combining. |
| EDF filtering | `training/files.py` and `cli/split.py` both define `filter_edf_files`. | Names overlap, but the split variant groups logical channel alternatives; clarify whether one strict primitive can serve both. |
| Channel/normalizer catalogs | HSP and Ruhrland each define `get_channels` and `resolve_normalizer`. | Retain dataset ownership unless a shared contract emerges from documentation. |
| Experiment scripts | The three root training scripts repeat patient preparation, collate setup, dataset construction, and loader construction. | Extraction is safer than creating a new framework abstraction around old experiments. |

## Boundary violations to address deliberately

- Several modules still import optional dependencies inside functions, contrary to current repository conventions. Most occur in the large legacy logger and one `BaseDataset` demonstration block.
- Existing tests mix reusable API behavior, paper configuration validation, integration data requirements, and historical regression checks.
- Some tests import files through repository paths instead of the installed package.
- Several generated artifacts and caches are colocated with paper sources under `iclr2026`.

The next test redesign should start from public contracts and failure modes after extraction, not mechanically preserve the current test-file layout.
