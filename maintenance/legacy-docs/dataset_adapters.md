# Dataset Adapters

This page summarizes the dataset-specific adapters that sit on top of
`sleepwalker.datasets.Basedataset.BaseDataset`.

Confirmed pattern:

- each adapter discovers EDF files in a dataset-specific directory layout
- each adapter translates sidecar annotations into a dataframe with at least
  `Label`, `Starttime`, and `Endtime`
- the shared dataset lifecycle in
  [dataset_lifecycle.md](/root/projects/sleepwalker/docs/dataset_lifecycle.md)
  then handles windowing, filtering, resampling, and target construction

The adapters are internal and lab-facing. They should be read as repository
implementations for specific storage layouts, not as stable public APIs.

## XML-backed adapters

Several adapters share the same broad structure:

- `ABC`
- `SHHS`
- `MROS`

Confirmed behavior:

- they locate XML files under parallel `annotations-events-nsrr` or
  `annotations-events-profusion` folders
- they can expose one annotation source as the main target and the other as
  `target_extra`
- they can optionally reject patients whose observed labels do not cover every
  mapped label

What remains local to each adapter is the exact path layout used to locate the
XML files.

## Tabular or custom sidecars

- `CAP` reads tab-delimited text annotations next to EDF files
- `ISRUC` reads annotator-specific Excel files and can either merge two
  annotators or expose one as `target_extra`
- `Apples` reads `.annot` sidecars
- `SleepEDFx` reads annotation events from a second EDF file

These adapters contain more dataset-specific parsing rules and small date
heuristics. Those heuristics are documented in code where visible, but their
historical rationale is not always present in the repository.

## Utility cache adapters

- `NumpyDataset` reads numpy-based frozen window caches
- `ZarrDataset` reads one-patient-per-store Zarr caches
- `UnlabelledDataset` acts as an inference-time EDF template
- `MultiDataset` concatenates already initialized datasets
- `SyntheticDataset` supports controlled tests and local experiments

These are repository utilities rather than external dataset integrations.

## Current caveats

- many dataset adapters assume a specific on-disk folder structure
- several adapters include hard-coded assumptions about time-of-day rollover
- large summary blocks in adapter docstrings are descriptive snapshots, not
  guaranteed contracts
- tests cover adapter initialization and sampling for several datasets, but
  many checks require local access to the underlying raw data
