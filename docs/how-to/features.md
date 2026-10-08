# Train a head on rocket features

Sleepwalker can turn the rocket feature extractors (random convolutional kernels) into a frozen encoder package and train a small head on top, the same way it reuses a [pretrained package as a backbone](models.md#reusing-pretrained-packages-as-backbones). The extractors are fitted once, saved as an embedding-only package, and `PackagedClassifierModel` trains a linear or MLP head on their output.

## Available extractors

| Name | Features | Fitted | Notes |
|---|---|---|---|
| `minirocket` | 9,996 proportions of positive values (PPV) in [0, 1] | yes (kernel biases are data quantiles) | Fast; works without scaling |
| `multirocket` | 49,728 features (PPV, mean/longest stretch of positive values, …) up to ~3,000 | yes | Not scaled yet: trains poorly (see below) |
| `rocket` | 20,000 (PPV and max per kernel) | yes | Slowest of the three |

The rocket extractors convolve every channel and ignore amplitude.

## 1. Describe the data in a training config

Start from `configs/examples/rocket_head_sleep_edfx.yml` (or `rocket_head_ruhrland.yml`). Use one physical name per channel, so every recording feeds the same signal:

```yaml
channels:
  - {logical_name: eeg_frontal, physical_names: [EEG Fpz-Cz], unit: uV}
  - {logical_name: eeg_occipital, physical_names: [EEG Pz-Oz], unit: uV}
  - {logical_name: eog_left, physical_names: [EOG horizontal], unit: uV}
```

## 2. Build the feature package

```bash
sleepwalker package-features configs/examples/rocket_head_sleep_edfx.yml -f minirocket --n-jobs 16 -o /raid/packages/minirocket-sleep-edfx
```

The command builds the dataset from the config's first `data` entry with one-epoch windows, makes the same patient split as `sleepwalker train`, and fits the extractors on 1,024 random windows from the **training** patients only. Add `--dry` to fit on two patients for a quick check. Combine extractors by listing several names (`-f minirocket multirocket`).

`--n-jobs` is how many CPUs featurize each batch wherever the package runs: in training, evaluation and inference. It is stored in the package and changes speed, not the features. Without it, the package featurizes on one CPU, with a warning.

!!! warning
    MultiRocket's features are not scaled yet: values up to ~3,000 sit next to features in [0, 1], and a linear head on them barely trains. Use `minirocket` until feature scaling is added.

## 3. Train a head

Point `model.package` at the package and run the config:

```yaml
model:
  name: sleepwalker.models.PackagedClassifierModel.PackagedClassifierModel
  package: /raid/packages/minirocket-sleep-edfx
  head: mlp
```

```bash
sleepwalker dry configs/examples/rocket_head_sleep_edfx.yml
sleepwalker train configs/examples/rocket_head_sleep_edfx.yml
```

!!! note
    On a machine with fewer CPUs than the package's `--n-jobs`, featurization stops with an error. This includes the package that `sleepwalker train` exports. Lower `n_jobs` on the loaded package and save it as a new package:

    ```python
    from sleepwalker.deployment import load_packaged_model
    from sleepwalker.models.preprocessors.Featurize import Featurize

    package = load_packaged_model("/raid/packages/minirocket-sleep-edfx")
    for stage in package.model.modules():
        if isinstance(stage, Featurize):
            stage.n_jobs = 8
    package.save("/raid/packages/minirocket-sleep-edfx-8cpu")
    ```

For several epochs of context, set `total_input` to N × 30 s with a 30 s `stride`, and set `prepare_target.target_resolution` to the whole window with `sequence_len: N` (and the trainer's `sequence_len` to N). `PackagedClassifierModel` then featurizes each epoch, concatenates the N feature vectors and predicts all N labels.

!!! note
    Features are recomputed in every forward pass; there is no feature cache yet. On CPU, a full epoch over a cohort of ~200,000 windows takes minutes to tens of minutes, so check the timing with `sleepwalker dry` first.
