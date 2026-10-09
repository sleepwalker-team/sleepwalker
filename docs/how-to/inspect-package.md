# Inspect a model package

Inspect `manifest.json` before loading a package. This file records the model class, accepted inputs, outputs, source commit, and payload checksum.

```python
import json
from pathlib import Path


package_path = Path("artifacts/model")
manifest = json.loads((package_path / "manifest.json").read_text())

print(manifest["model_class"])
print(manifest["dataset_class"])
print(manifest["input_channels"])
print(manifest["capabilities"])
print(manifest["classification"])
print(manifest["git_commit"])
```

The checksum detects changes to `model.pt`; it does not prove who created the package. Do not load an untrusted payload.

## Check the EDF settings

Load a trusted package and inspect its dataset:

```python
from sleepwalker.deployment import load_packaged_model


package = load_packaged_model("artifacts/model", map_location="cpu")
dataset = package.dataset

print("channels:", dataset.get_input_channels())
print("sampling frequency:", dataset.sample_frequency)
print("window duration:", dataset.total_input)
print("window stride:", dataset.stride)
print("resampling:", dataset.resample_type)
print("channel processing:", [(channel.logical_name, channel.preprocessors) for channel in dataset.channels])

for channel in dataset.channels:
    print("logical name:", channel.logical_name)
    print("accepted EDF names:", channel.physical_names)
```

`logical_name` is the channel name passed to the model. `physical_names` lists EDF header names that can supply it, in lookup order. Normalizers and preparation callbacks are also stored on the dataset object.

For a graph model, `package.dataset.datasets` contains one dataset per graph node. Inspect each entry separately because their channel and window settings may differ.

## Check model input and output shapes

```python
shape, metadata = package.model.input_spec()
print(shape)
print(metadata)
print(package.capabilities)
print(package.classification_contract)
```

Most models receive a tensor with layout `[batch, time, channel]`. The leading `1` shown by `input_spec()` is an example batch size, not a limit. Models that receive named inputs return a mapping of names to shapes.

For a single-head classifier, `classification_contract` records:

- `classes`: output labels in logit order;
- `sequence_len`: predictions produced for each input window;
- `target_resolution`: duration represented by each prediction;
- `target_offset`: first prediction time relative to the window start.

Do not infer class names or timestamps from tensor dimensions.

To check an independently created dataset before prediction, call:

```python
package.assert_compatible(candidate_dataset)
predictions = package.predict_dataset(candidate_dataset, device="cpu")
```

For one raw EDF, use `package.predict_patient(path)`. It clones and initializes the saved dataset settings automatically.
