# Troubleshoot EDF and package errors

## `ModuleNotFoundError: No module named 'sleepwalker'`

Activate the environment where the current checkout is installed:

```bash
source .venv/bin/activate
python -m pip install -e .
python -c "import sleepwalker; print(sleepwalker.__file__)"
```

The printed path should point into the current checkout. Reinstall after moving the repository or switching to a branch that changes the package layout.

## No Sleep-EDFx recordings are found

Point the code at the `SC` or `ST` directory, not its parent:

```python
from sleepwalker.datasets.utils import get_edf_files_in_repo


files = get_edf_files_in_repo("data/sleep-edfx/SC")
patients = [path for path in files if path.endswith("-PSG.edf")]
print(len(patients))
```

An empty list usually means the path is wrong or the download is incomplete.

## A Sleep-EDFx hypnogram is missing or ambiguous

Keep each `*-PSG.edf` beside its matching `*-Hypnogram.edf` and retain the original filenames. Sleep-EDFx requires exactly one matching hypnogram.

## An EDF channel cannot be resolved

Inspect the names accepted by the package:

```python
for channel in package.dataset.channels:
    print(channel.logical_name, channel.physical_names)
```

Compare them with the EDF header. Add a physical alternative while configuring a dataset for a new training run. Do not silently substitute a different signal for an already trained model unless that substitution is scientifically justified.

## A physical unit is missing or incompatible

A `ConvertUnit` step returns `None` when the source unit is unknown or incompatible. Add the documented dataset correction function before `ConvertUnit` in the processor list. If calibration is unknown, use a processor that supports relative signals or digital loading. A unit label alone cannot recover an unknown gain.

## Package payload hash mismatch

`load_packaged_model()` checks `model.pt` against the SHA256 value in `manifest.json`. A mismatch means one of the two files changed or the copy is incomplete. Copy or export the complete package directory again; do not edit the checksum to suppress the error.

## Unsupported package format or missing Python class

Use a Sleepwalker checkout compatible with the package's `format_version`, `model_class`, and `dataset_class`. Read those fields from `manifest.json` without loading `model.pt`. The optional `git_commit` field identifies the exporting source revision when it was available.

## `predict_patient()` reports no classification output

Check:

```python
print(package.capabilities)
print(package.classification_contract)
```

An embedding-only package cannot return class predictions. Use [Extract embeddings](extract-embeddings.md) instead.
