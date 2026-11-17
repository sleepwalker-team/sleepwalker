# Model packages

A model checkpoint is not enough to interpret a raw EDF. Prediction also depends on the expected channel names and units, sampling frequency, normalization, window size, class order, and output timing. Sleepwalker saves these values with the trained model in a model package.

Each package is a directory:

```text
my-model/
├── manifest.json
└── model.pt
```

`manifest.json` is JSON and can be inspected without loading the model. `model.pt` contains the model and a copy of the dataset configured for prediction.

## How an EDF becomes a prediction

When you call `package.predict_patient("night.edf")`, Sleepwalker:

1. clones the saved dataset;
2. resolves the required signals from the EDF header;
3. prepares the recording processors and fits their optional state;
4. resamples and normalizes the signals;
5. creates windows with the saved duration and stride;
6. runs the model in batches; and
7. labels each output with its class and timestamp.

A fresh dataset clone is used for every call, so initialized patient state is not shared between recordings.

The saved channel configuration distinguishes a logical channel from its accepted EDF names. For example, a model may consume a logical channel named `eeg` while accepting `EEG Fpz-Cz` or `EEG Pz-Oz` from an EDF header. The accepted names are tried in order. The chosen signal passes through the ordered processor list, including any explicit unit conversion.

## Output labels and times

Classifiers store `classification_contract` on the loaded package. For single-head classification it contains the ordered class names, the number of predictions per input window, the duration represented by each prediction, and the offset of the first prediction from the input window start.

These values are needed even when the output tensor shape is known. A five-element tensor does not say which element represents REM, and a sequence of outputs does not say which EDF timestamps they describe.

## Checkpoints and packages serve different purposes

The training checkpoint contains trainer state needed to resume optimization. A package contains what `predict_patient()` needs to process EDF recordings. Use a checkpoint to continue training and a package for inference or distribution.

## Compatibility checks

`predict_dataset()` accepts a separately initialized dataset after `package.assert_compatible(dataset)` checks its logical channels, sampling, and window geometry.

Compatibility checks cover input geometry and timing. They do not infer units or compare arbitrary processor functions. Record preprocessing changes as part of the experiment. Channel swaps, unit mistakes, and shifted windows can produce plausible numbers while invalidating a comparison.

## Loading packages executes serialized code

The SHA256 value in `manifest.json` detects whether `model.pt` has changed since the manifest was written. It does not establish that the package is safe. The payload uses Python/PyTorch serialization and can execute code during loading. Load packages only from a trusted source and use a compatible Sleepwalker environment.
