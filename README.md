# Sleepwalker

Sleepwalker is a research-oriented Python repository for training and evaluating models on EDF-backed sleep-study data. The current codebase is aimed at developers and ML engineers in the same lab, not at an external public API. That is visible in the top-level training scripts, the hard-coded data-root assumptions in several files, and the fact that most workflows are assembled directly in Python rather than exposed through stable package entry points.

## Confirmed Scope

Confirmed from code and tests:

- EDF files are loaded through utilities in `sleepwalker/core/signal.py`.
- Dataset adapters under `sleepwalker/datasets/` turn patient recordings into sliding-window items.
- Models live under `sleepwalker/models/`.
- Training logic lives under `sleepwalker/trainer/`.
- Trained models can be exported as `.swmodel` bundles and later loaded for inference through `sleepwalker/deployment/`.
- `predict.py` can load one or more exported packages and write per-EDF prediction tables.

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
- `predict.py`: inference CLI for exported prediction packages.

## Main Workflow

At a high level, the repository follows this flow:

1. A dataset adapter prepares EDF-backed patient recordings into sliding windows.
2. A model and trainer are configured in a task-specific script.
3. `sleepwalker.trainer.Run.run(...)` executes shared training and test mechanics.
4. A trained model can optionally be exported as a `.swmodel` prediction package.
5. `predict.py` loads exported packages and scores EDF files.

## Current Constraints

- There is no stable public API yet.
- The `train_*` scripts are currently active development surfaces.
- `predict.py` exists and is tested in parts, but actual deployment behavior is still best treated as lab-internal and evolving.
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
