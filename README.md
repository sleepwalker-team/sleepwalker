# Sleepwalker

Sleepwalker is a research-oriented Python repository for training and evaluating models on EDF-backed sleep-study data. The current codebase is aimed at developers and ML engineers in the same lab, not at an external public API. That is visible in the top-level training scripts, the hard-coded data-root assumptions in several files, and the fact that most workflows are assembled directly in Python rather than exposed through stable package entry points.

## Confirmed Scope

Confirmed from code and tests:

- EDF files are loaded through utilities in `sleepwalker/core/signal.py`.
- Dataset adapters under `sleepwalker/datasets/` turn patient recordings into sliding-window items.
- Models live under `sleepwalker/models/`.
- Training logic lives under `sleepwalker/trainer/`.
- Trained models can be exported as manifest-backed `PackagedModel` directories and later loaded for inference through `sleepwalker/deployment/`.
- `tools/predict.py` contains shared prediction-table normalization helpers.

Probable but not fully stabilized:

- The repository is used for sleep staging plus several related event-detection tasks such as arousal, desaturation, noisy windows, body position, and Ruhrland-specific multitask experiments.
- Some top-level scripts and utilities appear to be actively developed or experimental.

## Repository Layout

- `sleepwalker/core/`: EDF I/O and signal conversion helpers.
- `sleepwalker/datasets/`: dataset base class, dataset adapters, normalizers, augmentation, cache/export helpers.
- `sleepwalker/models/`: base models, architectures, meta/fusion models, preprocessors.
- `sleepwalker/trainer/`: base trainer, task-specific trainers, loss/metric/split/target utilities.
- `sleepwalker/deployment/`: prediction-package export and loading.
- `tests/`: unit tests and small EDF/text fixtures.
- top-level `train_*.py`: task-specific experiment scripts.
- `tools/predict.py`: prediction-table normalization helpers used by analysis scripts.

## Main Workflow

At a high level, the repository follows this flow:

1. A task-specific `train_*.py` script selects records and configures its dataset, including channels, units, normalizers, and patient/target/sample callbacks.
2. The configured dataset prepares EDF-backed recordings into sliding windows.
3. A model and trainer are configured in the same task-specific script.
4. `sleepwalker.trainer.Run.run(...)` executes shared training and test mechanics.
5. Trainers write serialized model/training-state checkpoints. The run stores its resolved configuration separately in `hparams.yml`; metrics are written to the run's JSONL results.
6. At the end of a run, `run(...)` writes a `PackagedModel` coupling the model, unlabelled inference dataset, and classification contract. Trainers are not stored in packages.
7. `package.predict_edf(...)` uses the stored unlabelled inference dataset, so inference reuses channel resolution, unit conversion, resampling, normalizers, and callbacks from training.

The usual inference path is deliberately small:

```python
from sleepwalker.deployment import load_packaged_model

package = load_packaged_model("results/.../final/package")
prediction = package.predict_edf("night.edf")
```

Inference always uses the dataset contract stored in the package. Missing
channels, incompatible units, and changed model geometry fail before model
execution. Dataset-specific corrections for known bad header labels, such as
HSP saturation channels labelled `uV`, are stored explicitly.

`StackedClassifierModel` composes packaged task experts. It preserves each
expert's native input contract, aligns their prediction timelines, and adds
task-specific probability correction heads. `PairedDataset` coalesces the
aligned native input windows without moving signal loading into the model.

## Current Constraints

- There is no stable public API yet.
- The `train_*` scripts are currently active development surfaces.
- Packaged models are trusted, repo-local research artifacts. Loading their PyTorch/cloudpickle payload can execute code and requires a compatible checkout.
- Raw-EDF inference is exposed through `PackagedModel.predict_edf(...)`; the artifact
  format and deployment behavior remain lab-internal and evolving.

Dataset manifests are created explicitly with `python tools/split.py ...`; use `--folds N` for cross-validation. A trained classifier can be evaluated with `python tools/evaluate.py CONFIG`, where `test.package` names its package directory. Complete systems are evaluated with `python tools/evaluate_system.py CONFIG`; these configs define packaged classifiers together with a dependency policy that conditions downstream predictions on upstream predicted classes. The reproducible paper pipeline is defined in `iclr2026/scripts/run.sh`, the current stacking work queue is in `iclr2026/scripts/run_next.sh`, and SHHS/Ruhrland hand-off evaluations are kept in `iclr2026/scripts/run_other.sh`.

Pinned SleepFM, SleepGPT, and OSF sources can be downloaded and converted into embedding-only packages with `tools/foundation_models.py`. A regular `tools/train.py train CONFIG` run can then fit `PackagedClassifierModel`; see `configs/examples/foundation_sleep_head.yml`.
- The repository contains generated artifacts, notebooks, logs, and likely experimental files alongside maintained source.

# Discord Update Workflow

This repository supports manual Discord update posts via GitHub Actions and Discord webhooks.

## Overview

Updates are written into:

```text
.discord_update.md
```

and can then be posted to the Sleepwalker Discord channel using a manually triggered GitHub Action.

This avoids noisy commit spam while still allowing curated project updates.

## Usage

### 1. Write an update

Edit:

```text
.discord_update.md
```

### 2. Commit and push

```bash
git add .discord_update.md
git commit -m "Add Discord update"
git push
```

---

### 3. Trigger the GitHub Action

From the terminal:

```bash
gh workflow run discord-update.yml --ref "$(git branch --show-current)"
```

or manually from GitHub:

```text
Actions
→ Post Discord update
→ Run workflow
```

---

## Behavior

The workflow automatically:

* reads `.discord_update.md`
* appends:
  * commit SHA
  * repository URL
  * branch name
* splits long messages into multiple Discord posts
* posts everything to the configured Discord channel

---

## Notes

* The workflow is intentionally manual (`workflow_dispatch`) to avoid channel spam.
* Long updates are automatically chunked to satisfy Discord message limits.
* The workflow file is located at:
