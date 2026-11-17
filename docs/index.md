# Sleepwalker

<div class="sw-hero">
  <h1 class="sw-hero__title">Walk across the sleep of thousands of patients</h1>
  <p class="sw-hero__subtitle">
    Sleepwalker is a scientific library for training, packaging, and running machine-learning
    models for sleep analysis — from a single EDF recording to terabyte-scale cohorts on
    commodity hardware.
  </p>
  <div class="sw-hero__cta">
    <a class="md-button md-button--primary" href="getting-started/">Get started</a>
    <a class="md-button" href="how-to/">How-to guides</a>
  </div>
</div>

<div class="sw-features">
  <a class="sw-card" href="how-to/predict-patient.md">
    <div class="sw-card__icon">🩺</div>
    <h3>Score your patients</h3>
    <p>Run a pretrained package on any EDF recording and get per-window sleep stages with per-class probabilities — no resampling or channel wrangling required.</p>
  </a>
  <a class="sw-card" href="how-to/train-sleep-staging.md">
    <div class="sw-card__icon">🧠</div>
    <h3>Train new models</h3>
    <p>Assemble a dataset, model, and trainer in Python and run a reproducible experiment, or drive the same run from a YAML config on the CLI.</p>
  </a>
  <a class="sw-card" href="how-to/extract-embeddings.md">
    <div class="sw-card__icon">📉</div>
    <h3>Extract embeddings</h3>
    <p>Obtain low-dimensional per-window embeddings from embedding models and inspect patients in a compact latent space.</p>
  </a>
</div>

## Why Sleepwalker

- **EDF-native.** Read polysomnography EDF files directly. Channel mapping, unit conversion, resampling, normalization, and windowing are handled by the dataset layer, so a model sees a clean, consistent tensor.
- **API first, tooling second.** Every object — datasets, models, trainers, deployment — is usable directly from Python. The CLI and YAML are a thin, repeatable wrapper over the same objects, never a more powerful path.
- **Built for scale.** Lazy, window-at-a-time loading with worker processes and patient-balanced sampling keeps terabytes of recordings streaming to the GPU without holding them in memory. See [Tune data loading performance](how-to/performance.md).
- **Reproducible packages.** A model package freezes the weights together with the exact EDF preparation used in training, so prediction repeats the pipeline that produced the result.

## A taste of the API

Score a recording with a trained package:

```python
from sleepwalker.deployment import load_packaged_model

package = load_packaged_model("artifacts/sleep-staging", map_location="cpu")
predictions = package.predict_patient("recordings/night.edf", device="cpu")
print(predictions[["time", "prediction"]].head())
```

Or assemble and run a training experiment in Python:

```python
from sleepwalker.trainer.Run import RunCfg, run

result = run(
    RunCfg(
        model=model,
        trainer=trainer,
        train_datasets=[train_dataset],
        val_datasets=[val_dataset],
        test_datasets=[("sleep-edfx", test_dataset)],
        collate_fn=batch_collate,
        experiment_name="my-run",
    )
)
```

`predictions` is a pandas `DataFrame` with one row per output time step, including the predicted label and one probability column per class.

## Where to go next

- New here? Start with [Getting started](getting-started/index.md) — install, download the example data, and train your first model.
- Solving a specific task? Browse the [how-to guides](how-to/index.md).
- Looking something up? Use the [reference](reference/api.md).
- Want to contribute? Read the [contributing guide](about/contributing.md).
