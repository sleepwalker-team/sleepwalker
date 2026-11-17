# Sleep Datasets

This repository currently mixes multiple sleep-study datasets behind a common
dataset interface.

Confirmed from `train_sleep.py` and the adapter modules:

- `SleepEDFx`
- `CAP`
- `ISRUC`
- `SHHS`
- `ABC`
- `Apples`
- `MROS`
- `Ruhrlandklinik`

The project does not currently present these as a polished public data-loading
library. They are the datasets the lab's training scripts know how to read.

## Common model-facing interface

Each dataset adapter eventually provides windows with:

- time-indexed signal data
- a target dataframe derived from annotation intervals
- optional `target_extra` annotations in a few adapters

This makes it possible for the same training and evaluation code to operate on
very different source datasets.

## Why the adapters differ

The datasets use different annotation formats:

- EDF annotations in `SleepEDFx`
- text sidecars in `CAP`
- Excel sidecars in `ISRUC`
- XML sidecars in `ABC`, `SHHS`, and `MROS`
- `.annot` files in `Apples`
- spreadsheet and CSV combinations in `Ruhrlandklinik`

As a result, the adapter layer is partly about normalization of file layout and
partly about normalization of labels and interval timing.

## Confirmed limitations

- some adapters include dataset-specific date rollover heuristics
- some adapters assume parallel directory trees rather than colocated files
- some adapters can reject patients based on label coverage rather than only on
  file readability
- the repository contains both general adapters and lab-specific adapters, so
  “supported dataset” should be read as “currently used by local scripts”

## Related pages

- [dataset_lifecycle.md](/root/projects/sleepwalker/docs/dataset_lifecycle.md)
- [target_construction.md](/root/projects/sleepwalker/docs/target_construction.md)
- [ruhrland_dataset_notes.md](/root/projects/sleepwalker/docs/ruhrland_dataset_notes.md)
