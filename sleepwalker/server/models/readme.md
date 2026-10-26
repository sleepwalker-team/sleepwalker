# Trained Sleepwalker models

All experts were trained on HSP.

## Performance

**Macro-F1, pooled across scored time steps (all declared classes), rounded to
three decimals.** HSP is held-out; the other five cohorts are cross-cohort tests.

| Package / task | HSP held-out | SHHS | ABC | Ruhrlandklinik | APPLES | MrOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sw-sleep` | 0.781 | 0.729 | 0.713 | 0.679 | 0.647 | 0.729 |
| `sw-arousal` | 0.618 | 0.630 | 0.560 | 0.585 | 0.651 | 0.601 |
| `sw-breathing` | 0.705 | 0.591 | 0.640 | 0.690 | 0.555 | 0.579 |
| `sw-desaturation` | 0.735 | 0.559 | 0.721 | 0.750 | 0.749 | 0.523 |
| `joint` / sleep | 0.768 | 0.728 | 0.690 | 0.659 | 0.645 | 0.703 |
| `joint` / arousal | 0.630 | 0.598 | 0.584 | 0.595 | 0.660 | 0.547 |
| `joint` / breathing | 0.713 | 0.556 | 0.594 | 0.692 | 0.493 | 0.543 |
| `joint` / desaturation | 0.662 | 0.524 | 0.625 | 0.681 | 0.619 | 0.458 |

[model_performance.csv](model_performance.csv) contains all 48 model/task/cohort
results at full precision: accuracy, macro/micro-F1, Cohen's kappa, precision,
recall, per-class scores, confusion matrices, patient means, evaluation counts
and source/package checksums. Expert scores use the saved `resolved` aggregates. Joint scores pool the saved
per-recording confusion matrices after annotation filtering and dependency
gating (sleep → arousal/breathing; breathing → desaturation). Eligible recordings
and scoring differ, so expert and joint scores are not directly comparable.
`predict_patient()` alone does not apply this evaluation pipeline.

Source paths in the CSV are relative to
`/raid/sleepwalker/results/recording-normalized/` (`test/` and `system/`).

## Load and use

Run from the directory containing `models/`, in your Python environment
([installation docs](https://sleepwalker-team.github.io/sleepwalker/getting-started/)):

```bash
git clone --branch main https://github.com/sleepwalker-team/sleepwalker.git sleepwalker
git -C sleepwalker pull --ff-only origin main
git -C sleepwalker checkout --detach 2c77bf61671a2d3690d6a29d73fd5eece7e1c544
python -m pip install -e ./sleepwalker
```

For an existing clean checkout, omit the clone command. The checkout pins the
compatible framework revision after pulling. The manifests' `git_commit`
(`8db9e8dd19f60ef40abce45103a26b803ea69179`) identifies the **models repository**,
not the Sleepwalker framework. The framework revision above was checked against
all five packages.

Save as `predict.py` and run `python predict.py`:

```python
from sleepwalker.deployment import load_packaged_model


def main():
    package = load_packaged_model("models/sw-sleep", map_location="cpu")
    predictions = package.predict_patient("/path/to/night.edf", device="cpu")
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
```

Choose any package directory from the table; `joint` returns all four tasks.
Pass the original EDF: the package supplies preprocessing. For another cohort,
check its channel names and units against the saved input contract.
See the [prediction guide](https://sleepwalker-team.github.io/sleepwalker/how-to/predict-patient/)
for output columns and GPU use, [package inspection](https://sleepwalker-team.github.io/sleepwalker/how-to/inspect-package/)
for input requirements, and the [API reference](https://sleepwalker-team.github.io/sleepwalker/reference/api/).
[Documentation at the pinned revision](https://github.com/sleepwalker-team/sleepwalker/tree/2c77bf61671a2d3690d6a29d73fd5eece7e1c544/docs)
is available alongside the source.
