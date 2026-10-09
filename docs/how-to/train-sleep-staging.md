# Train and export a sleep-staging model

This tutorial trains [AttnSleep](../reference/models.md) on Sleep-EDFx **directly in Python** and exports a model package that can read another EDF recording. It is the shortest complete path through the API: build a dataset, a model, and a trainer, wrap them in a [`RunCfg`](../reference/api.md#run-cfg), and call [`run()`](../reference/api.md#run).

Sleepwalker is API first and tooling second: everything the CLI can do, you can do by constructing the same objects yourself. This page is that Python path for a simple model. For the in-depth treatment of targets, rejection, and class balance, see [Train a multiclass model](train-multiclass.md). To run the same example from a YAML file instead, see [Train from the CLI](train-from-cli.md).

The example is a short functional run, not a benchmark configuration. Increase the number of epochs and training samples when running an experiment.

## Build the objects

Download Sleep-EDFx first (see [Download Sleep-EDFx](sleep-edfx.md)), then construct the dataset, model, and trainer in Python:

```python
from functools import partial

import torch

from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models.AttnSleep import AttnSleep
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.targets import prepare_multiclass_target


classes = ["n1", "n2", "n3", "rem", "wake"]

def make_dataset():
    return SleepEDFx(
        channels=[ChannelConfig("eeg", ["EEG Fpz-Cz"], preprocessors=[EEGFilterNormalizer(fs=100)], unit="uV")],
        sample_frequency=100,
        event_mapping={
            "sleep stage w": "wake",
            "sleep stage 1": "n1",
            "sleep stage 2": "n2",
            "sleep stage 3": "n3",
            "sleep stage 4": "n3",
            "sleep stage r": "rem",
        },
        total_input="30s",
        stride="30s",
        prepare_target=partial(prepare_multiclass_target, target_classes=classes, target_resolution="30s"),
    )

files = get_edf_files_in_repo("data/sleep-edfx/SC")
patients = [path for path in files if path.endswith("-PSG.edf")]
train_patients, held_out = random_split(patients, test_frac=0.3, seed=17)
validation_patients, test_patients = random_split(held_out, test_frac=0.67, seed=17)

train_dataset = make_dataset()
validation_dataset = make_dataset()
test_dataset = make_dataset()
train_dataset.initialize(train_patients, num_workers=2, strict=True)
validation_dataset.initialize(validation_patients, num_workers=2, strict=True)
test_dataset.initialize(test_patients, num_workers=2, strict=True)

model = AttnSleep(n_channels=1, ts_len=3000, classes=classes)
trainer = MulticlassTrainer(
    epochs=5,
    optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    classes=classes,
    loss_function=torch.nn.functional.cross_entropy,
    target_resolution="30s",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    early_stopping=3,
)
```

Use the same `classes` list for target preparation, the model, and the trainer. Its order determines both the target columns and the predicted class indices. The `optimizer` argument is a factory that receives the model, not an optimizer instance; the trainer calls it with the model on the first `fit()`, so the parameter groups must be extracted inside the factory (the natural-looking `partial(torch.optim.Adam, lr=1e-3)` fails — see [Accept optimizer factories that receive the model](../roadmap.md#accept-optimizer-factories-that-receive-the-model)).

## Assemble the run and train

Wrap the objects in a `RunCfg` and call `run()`:

```python
configuration = RunCfg(
    experiment_name="sleep-edfx-attnsleep-python",
    model_name="attnsleep",
    model=model,
    trainer=trainer,
    train_datasets=[train_dataset],
    val_datasets=[validation_dataset],
    test_datasets=[("sleep-edfx", test_dataset)],
    batch_size=32,
    n_samples=4096,
    n_samples_test=1024,
    num_workers_dataloader=2,
    collate_fn=batch_collate,
    log_path="results/tutorial",
    expert_task="sleep",
    package_path="artifacts/sleep-edfx-attnsleep-python",
)

result = run(configuration)
```

`run()` trains the model, restores the selected weights when the trainer returns a best checkpoint, evaluates the test dataset, writes a resumable checkpoint, and exports the model package to `package_path`. It raises `FileExistsError` rather than overwriting an existing result, so change `experiment_name` for a variant. Call `sleepwalker.trainer.Run.seed_everything(seed)` before building the objects if you need a reproducible run.

The exported package under `artifacts/sleep-edfx-attnsleep-python/` holds the trained weights together with the EDF settings used during training — channel names, units, sampling frequency, normalization, window length, and class order — so prediction can repeat the exact preparation:

```text
artifacts/sleep-edfx-attnsleep-python/
├── manifest.json
└── model.pt
```

## Predict another recording

```python
from sleepwalker.deployment import load_packaged_model


package = load_packaged_model("artifacts/sleep-edfx-attnsleep-python", map_location="cpu")
predictions = package.predict_patient("recordings/night.edf", device="cpu", progress=True)

print(predictions[["time", "prediction", "valid"]].head())
```

`predictions` is a pandas `DataFrame` with one row per output time step. Probability columns are named `prob__<class>`, for example `prob__wake` and `prob__rem`. See [Predict sleep stages](predict-patient.md) for batching, GPU execution, and output details.

## Export from your own training loop

If you train the model without `run()`, export the selected inference weights directly with [`save_packaged_model()`](../reference/api.md#save-packaged-model):

```python
from sleepwalker.deployment import save_packaged_model


package = save_packaged_model(
    "artifacts/my-model",
    name="my-model",
    task="sleep",
    model=model,
    dataset=train_dataset,
    classification_contract=trainer.classification_contract(),
    config={"experiment": "custom-loop"},
)
```

Export after loading the weights you want to use for prediction. The saved dataset supplies the EDF settings used by `predict_patient()`.

## Where to go next

- [Train a multiclass model](train-multiclass.md) — the in-depth Python tutorial: targets, rejection, class balance, and the full `RunCfg`.
- [Train from the CLI](train-from-cli.md) — turn this example into a YAML config and run it with `sleepwalker train`.
- [RunCfg and the CLI](runcfg-cli.md) — every run field and CLI command.
