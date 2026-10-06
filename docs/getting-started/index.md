# Getting started

Go from an empty environment to a trained, exported sleep-staging model. This page installs Sleepwalker and points you at the example data and your first run.

## Install Sleepwalker

Sleepwalker requires Python 3.10 or newer. Clone the repository, create a virtual environment, and install the package:

```bash
git clone https://github.com/sleepwalker-team/sleepwalker.git
cd sleepwalker
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Verify the installation:

```bash
python -c "import sleepwalker; print(sleepwalker.__file__)"
sleepwalker --help
```

Install additional dependencies only when you need them:

```bash
python -m pip install -e ".[datasets]"     # spreadsheet and archive-backed datasets
python -m pip install -e ".[tracking]"     # MLflow and Weights & Biases
python -m pip install -e ".[dev,docs]"     # tests and documentation
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

[SleepFM and OSF](../how-to/foundation-models.md) use the default dependencies. Their weights require explicit download consent or a local released checkpoint.

## Get the example data

The introductory examples use Sleep-EDFx because it can be downloaded directly and contains paired PSG and hypnogram EDF files. Follow [Download Sleep-EDFx](../how-to/sleep-edfx.md) to fetch it into `data/sleep-edfx/`.

## Train your first model

Work through the canonical Python tutorial, which trains AttnSleep on Sleep-EDFx and exports a model package you can run on another recording:

→ [Train and export a sleep-staging model](../how-to/train-sleep-staging.md)

Prefer the command line? The same example as a YAML config is in [Train from the CLI](../how-to/train-from-cli.md).

## Where to go next

- [How-to guides](../how-to/index.md) — predict, extract embeddings, package models, and tune performance.
- [Reference](../reference/api.md) — the full Python API and dataset/model/trainer listings.
- [Contributing](../about/contributing.md) — design principles and how to open a pull request.
