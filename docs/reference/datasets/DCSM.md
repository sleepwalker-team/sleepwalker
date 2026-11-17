# DCSM

`sleepwalker.datasets.DCSM` currently contains only the download helpers for the DCSM files and their published SHA256 digests. It runs as a module command:

```bash
python -m sleepwalker.datasets.DCSM --out data/dcsmdemo
```

<!-- TODO: There is no `DCSM` dataset adapter class in this module, i.e. no subclass of `BaseDataset` and no `get_event_df()` implementation. DCSM stores its signals as HDF5 (`psg.h5`) plus a `hypnogram.ids` file rather than as EDF, so the EDF-based loading path in `BaseDataset` does not apply. The adapter and a documented loading example are missing; see the [roadmap](../../roadmap.md#implement-the-dcsm-dataset-adapter). -->

## Download helpers

::: sleepwalker.datasets.DCSM.download_dataset

::: sleepwalker.datasets.DCSM.download_and_validate

::: sleepwalker.datasets.DCSM.validate_sha256
