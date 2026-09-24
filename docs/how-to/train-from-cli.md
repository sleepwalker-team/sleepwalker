# Train from the CLI

The two Python tutorials — [Train and export a sleep-staging model](train-sleep-staging.md) and [Train a multiclass model](train-multiclass.md) — build the dataset, model, trainer, and [`RunCfg`](../reference/api.md#run-cfg) by hand. The CLI does the same thing from a YAML file: it reads fully qualified component specs, constructs the objects, and calls [`run()`](../reference/api.md#run). This page shows how to turn the Python you already wrote into a config and invoke it. For the exhaustive field list and the exact execution order, see [RunCfg and the CLI](runcfg-cli.md).

## From Python to YAML

The Python tutorial ends with a `RunCfg` you constructed yourself:

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
)
```

The equivalent YAML replaces every constructed object with a mapping whose `name` is the fully qualified import path and whose remaining keys are the constructor arguments. The Python `make_dataset()` factory becomes the `data:` section, the `AttnSleep(...)` call becomes `model:`, and the `MulticlassTrainer(...)` call becomes `trainer:`:

```yaml
seed: 17

data:
  name: sleepwalker.datasets.SleepEDFx.SleepEDFx
  channels:
    - logical_name: eeg
      physical_names: [EEG Fpz-Cz]
      unit: uV
      normalizer:
        name: sleepwalker.datasets.normalizer.EEGFilterNormalizer.EEGFilterNormalizer
        fs: 100
  sample_frequency: 100
  event_mapping:
    sleep stage w: wake
    sleep stage 1: n1
    sleep stage 2: n2
    sleep stage 3: n3
    sleep stage 4: n3
    sleep stage r: rem
  total_input: 30s
  stride: 30s
  output_classes: &classes [wake, n1, n2, n3, rem]
  prepare_target:
    name: sleepwalker.trainer.utils.targets.prepare_multiclass_target
    target_resolution: 30s
    target_classes: *classes
  patient_filter:
    name: sleepwalker.training.files.filter_edf_files
  files:
    train:
      name: sleepwalker.datasets.utils.get_edf_files_in_repo
      root: data/sleep-edfx/SC
    validation_fraction: 0.1
    test_fraction: 0.2
  label: sleep-edfx
  num_workers: 2

model:
  name: sleepwalker.models.AttnSleep.AttnSleep

trainer:
  name: sleepwalker.trainer.MulticlassTrainer.MulticlassTrainer
  target_resolution: 30s
  epochs: 5
  optimizer:
    name: torch.optim.Adam
    lr: 0.001
  loss_function: torch.nn.functional.cross_entropy
  early_stopping: 3
  device: cpu

run:
  experiment_name: sleep-edfx-attnsleep
  model_name: attnsleep
  batch_size: 32
  n_samples: 4096
  n_samples_test: 1024
  num_workers_dataloader: 2
  test_repeats: [1]
  use_mlflow: false
  log_path: results/tutorial
  expert_task: sleep
```

Three things the CLI does for you that the Python version did by hand:

- **Context injection.** Before building the model, the CLI derives `classes`, `sequence_len`, `input_channels`, `n_channels`, `ts_len`, and `sampling_frequency` from the dataset and injects any the model's constructor accepts but the YAML omitted. That is why `model:` above needs no arguments — it cannot disagree with the data.
- **Optimizer factory.** The Python `lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)` becomes a plain `optimizer:` mapping; the CLI wraps it as the factory the trainer expects.
- **File selection and splits.** `files:` with `validation_fraction` / `test_fraction` reproduces the `random_split` logic from the tutorial, drawn with the config `seed`.

`collate_fn` is the one field the CLI sets for you (to `batch_collate`); in Python you must pass it explicitly or `run()` raises `ValueError`.

## Invoke the CLI

Save the config (for example `configs/examples/sleep_edfx_tutorial.yml`) and run it. Always smoke-test first:

```bash
sleepwalker dry configs/examples/sleep_edfx_tutorial.yml    # one epoch, capped patients and samples
sleepwalker train configs/examples/sleep_edfx_tutorial.yml  # the real run
```

`dry` caps patients (≤30/10/10), forces one epoch, disables MLflow and package export, and logs to a temporary directory — it exercises dataset loading, model input checks, fitting, evaluation, and export without touching your real `artifacts/`. Fix missing recordings, channel errors, and incompatible shapes before the real run.

The real run writes the same artifacts the Python tutorial produced — a resumable `checkpoint.pt` and an inference `package/` — under `<log_path>/<experiment_name>/`:

```text
results/tutorial/sleep-edfx-attnsleep/
├── hparams.yml            # the resolved configuration of this run
├── results.json           # train/test losses, confusion matrices, classes, paths
├── figures/               # confusion matrices
└── final/
    ├── checkpoint.pt      # resumable trainer state plus the selected weights
    └── package/           # inference package for predict_patient()
```

To continue a run that died before finishing:

```bash
sleepwalker resume results/tutorial/sleep-edfx-attnsleep/final/checkpoint.pt
```

For cross-validation, write the subject-level split once and train one fold at a time:

```bash
sleepwalker split data/sleep-edfx/SC splits/sleep-edfx.yml --folds 3 --seed 42
sleepwalker train configs/examples/sleep_edfx_tutorial.yml --fold fold_0
```

!!! warning "Trainers default to cuda:0"
    `MulticlassTrainer` and friends default to `device="cuda:0"`, and `sleepwalker train` does not override this. On a CPU-only machine, set `device: cpu` in the `trainer:` section or the run fails inside CUDA initialization. See the [roadmap](../roadmap.md#detect-the-training-device).

## Where to go next

- [RunCfg and the CLI](runcfg-cli.md) — every `RunCfg` field, the full YAML schema, and all CLI subcommands.
- [Train and export a sleep-staging model](train-sleep-staging.md) — the same example as Python.
- [Train a multiclass model](train-multiclass.md) — the in-depth Python tutorial.
