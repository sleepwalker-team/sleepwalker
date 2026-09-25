# Dataset adapters

Every adapter turns one public sleep database into a [`BaseDataset`](api.md#base-dataset): it locates the EDF files, maps the annotation format into the standardized `Starttime`/`Endtime`/`Label` table, and declares which physical channels the database provides. The shared windowing, unit, resampling, and callback machinery is documented once in [Load some data](../how-to/data.md); this page is the index of concrete adapters and what is special about each one.

Each adapter has its own generated reference page below. The pages render the class signature, parameters, and docstrings via `mkdocstrings`; where an adapter has extra helpers (download functions, alternate annotation sources), they are shown there.

## EDF-based clinical adapters

| Adapter | Reference | Notes |
| --- | --- | --- |
| `SleepEDFx` | [Sleep-EDFx](datasets/SleepEDFx.md) | Sleep-EDF expanded (SC + ST). Includes download helpers and a `*-Hypnogram.edf` label source. The reference dataset for the guides. |
| `SHHS` | [SHHS](datasets/SHHS.md) | NSF-SHHS via NSRR; Profusion XML sidecars for labels. |
| `HSP` | [HSP](datasets/HSP.md) | Used by the breathing-event configs; many alias channel names for airflow/RIP. |
| `Ruhrlandklinik` | [Ruhrlandklinik](datasets/Ruhrlandklinik.md) | Ruhrlandklinik polysomnography; `return_nox=True` exposes a second (NOX) annotation timeline as `target_extra`. |
| `ISRUC` | [ISRUC](datasets/ISRUC.md) | ISRUC with download helpers; second scorer available as `target_extra` when `merge=False`. |
| `ABC` | [ABC](datasets/ABC.md) | NSRR ABC dataset with Profusion event sidecars; alternate annotation source. |
| `MROS` | [MROS](datasets/MROS.md) | NSRR MROS with Profusion XML sidecars; alternate annotation source. |
| `MNC` | [MNC](datasets/MNC.md) | MNC database adapter. |
| `Stages` | [Stages](datasets/Stages.md) | Stages (MESA/CHS/SHHS/CFS) adapter. |
| `SVUH_UCD` | [SVUH_UCD](datasets/SVUH_UCD.md) | SVUH-UCD neonatal/adult PSG adapter. |
| `WSC` | [WSC](datasets/WSC.md) | Wisconsin Sleep Cohort adapter. |
| `CAP` | [CAP](datasets/CAP.md) | CAP sleep database adapter. |
| `Apples` | [Apples](datasets/Apples.md) | APPLES dataset adapter. |
| `HCHS` | [HCHS](datasets/HCHS.md) | HCHS/SOL adapter. |
| `NCHSDB` | [NCHSDB](datasets/NCHSDB.md) | NCHS database adapter. |
| `Numom2b` | [Numom2b](datasets/Numom2b.md) | Numom2b adapter. |

!!! note "Label casing is a convention, not a guarantee"
    Several adapters (`SleepEDFx`, `HSP`, `MNC`, `Ruhrlandklinik`, `Stages`, `SVUH-UCD`, `WSC`) lowercase annotation labels while reading them, so `event_mapping` keys must be lowercase for those. The convention is not enforced across adapters — check the adapter page or the raw labels when a mapping silently drops events. See [Loading data with labels](../how-to/data.md#loading-data-with-labels).

## Composition and utility datasets

| Dataset | Reference | Purpose |
| --- | --- | --- |
| `MultiDataset` | [MultiDataset](datasets/MultiDataset.md) | Combines several configured datasets into one index; patient ranges and targets are concatenated. Used for multi-dataset training. |
| `NumpyDataset` | [NumpyDataset](datasets/NumpyDataset.md) | Serves arrays exported by `export_dataloader_to_numpy_dir` from RAM or memmap. See [Performance](../how-to/performance.md#numpydataset-full-in-memory-data). |
| `UnlabelledDataset` | [UnlabelledDataset](datasets/UnlabelledDataset.md) | Wraps a configured dataset and strips label construction; the standard form for inference templates inside packages. |
| `SyntheticDataset` | [SyntheticDataset](datasets/SyntheticDataset.md) | Generates synthetic windows; useful for tests and smoke runs without real recordings. |

## Not (yet) an adapter

[`DCSM`](datasets/DCSM.md) currently ships only download helpers and checksums for the DCSM-Demo files. Its signals live in HDF5 and its hypnogram in a `.ids` file, which do not fit the EDF-based reader, so no dataset can be constructed from it yet. See the [roadmap](../roadmap.md#implement-the-dcsm-dataset-adapter).

## Shared building blocks

The pieces every adapter shares — `ChannelConfig`, `BaseDataset.initialize`, `batch_collate`, `prepare_tensor_sample` — are documented in the [Python API](api.md#base-dataset) and rendered in full in [reference/loading.md](loading.md).
