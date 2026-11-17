# Predict sleep stages of a patient

!!! warning "No published package yet"
    Replace `/path/to/sleep-staging-package` with a local package directory. Sleepwalker does not have a public package download location yet; a concrete downloadable package will be added once one is published, tracked on the [roadmap](../roadmap.md#publish-pretrained-model-packages).

Load the package with [`load_packaged_model()`](../reference/api.md#load-packaged-model), then call [`PackagedModel.predict_patient()`](../reference/api.md#predict-patient):

```python
from sleepwalker.deployment import load_packaged_model

# Using a CPU
package = load_packaged_model(
    "/path/to/sleep-staging-package",
    map_location="cpu",
)
predictions = package.predict_patient(
    "/path/to/patient.edf",
    batch_size=64,
    num_workers_dataset=0,
    num_workers_loader=0,
    device="cpu",
    progress=True,
)

# Using a GPU/CUDA device
# package = load_packaged_model(
#     "/path/to/sleep-staging-package",
#     map_location="cuda:0",
# )
# predictions = package.predict_patient(
#     "/path/to/patient.edf",
#     batch_size=128,
#     num_workers_dataset=4,
#     num_workers_loader=2,
#     device="cuda:0",
# )

print(predictions.head().to_string(index=False))

# Store predictions on disk
predictions.to_csv("patient-predictions.csv", index=False)
```

Pass the original EDF. There is no need for resampling, channel mapping etc. `predict_patient()` applies the settings stored in the package. `predict_patient()` returns a pandas `DataFrame` with one row per output time step. For a five-stage sleep model, the first rows have this form:

```text
              patient  task                time  prediction_idx prediction  prob__wake  prob__n1  prob__n2  prob__n3  prob__rem  valid
/path/to/patient.edf  sleep 2026-01-01 22:00:00               0       wake        0.82      0.04      0.09      0.02       0.03   True
/path/to/patient.edf  sleep 2026-01-01 22:00:30               0       wake        0.74      0.08      0.12      0.02       0.04   True
/path/to/patient.edf  sleep 2026-01-01 22:01:00               1         n1        0.18      0.51      0.22      0.03       0.06   True
```

In general, you can expect this output format:

| Column | Meaning |
| --- | --- |
| `patient` | Path of the source EDF. |
| `task` | Task name saved when the package was exported. |
| `time` | Timestamp assigned to the output using the package's output offset and resolution. |
| `prediction_idx` | Zero-based index of the largest model output in the saved class order. |
| `prediction` | Class name at `prediction_idx`. |
| `prob__<class>` | Softmax probability for one class. These columns sum to approximately one per row. |
| `valid` | Validity of a target when a labelled dataset supplies a target mask; otherwise `True`. |

For `predict_patient()`, `valid` is normally `True`. The flag marks target positions that should be excluded from the loss during training; prediction computes no loss, so every output row is reported.

## Average repeated views

Sometimes models are trained to consume one channel (e.g. one EEG channel `C4-M1`), but a patient recording supplies multiple EEG recordings. In this case, you can execute the same model multiple times using different channel from the same channel group and average the predictions. Using `n_repeat > 1` will execute such a sampling dependent on the packages configuration. Internally,  sleepwalker prepares $R$ views of the same input window, runs the model on every view, and averages the logits:

$$
\bar{\mathbf{z}}(x) = \frac{1}{R}\sum_{r=1}^{R} f\!\left(T_r(x)\right)
$$

The probabilities returned in the DataFrame are computed after averaging:

$$
p(y \mid x) = \operatorname{softmax}\!\left(\bar{\mathbf{z}}(x)\right)
$$

Here, $T_r$ is the random preparation applied to view $r$, $f$ is the model, and $\bar{\mathbf{z}}$ is the mean logit vector. Sleepwalker averages logits, not probabilities, and still returns one row per output time step.

```python
predictions = package.predict_patient(
    "/path/to/patient.edf",
    n_repeat=5,
    seed=42,
    device="cpu",
)
```

Leave `n_repeat=1` for deterministic preparation. The current `seed` argument initializes the data loader's PyTorch generator but does not guarantee that every custom callback RNG is reset. Reproducible repeated-view inference is tracked on the [roadmap](../roadmap.md#make-repeated-view-inference-reproducible).

See [Inspect the EDF settings](inspect-package.md#check-the-edf-settings) when an EDF uses unexpected channel names or units.


!!! warning "No command-line prediction yet"
    Patient prediction is currently documented through `PackagedModel.predict_patient()`. There is no integrated `sleepwalker predict` command. Stabilizing command-line prediction and evaluation is tracked on the [roadmap](../roadmap.md#stabilize-command-line-prediction-and-evaluation).
