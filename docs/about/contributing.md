# Contributing

This page is for anyone changing Sleepwalker itself: adding a dataset, implementing a model, changing training behavior, improving the docs, or preparing a pull request. It collects the design principles, the code architecture, the repository boundary, and the documentation rules in one place.

## Coding agents are welcome

Sleepwalker is developed in the open with the help of LLM-based coding agents, and **you are welcome to use them too**. Draft a change with an agent, an IDE, or by hand — the tool is not what matters.

What matters is that **every pull request is held to the same bar, regardless of who or what wrote it.** An agent-authored PR is reviewed exactly like a human-authored one: it must be correct, tested, minimal, and consistent with the code around it. In practice that means:

- **You own the change.** If you open a PR, you understand every line in it and can explain and defend it. Do not open a PR you have not read.
- **Verify, do not assume.** Run the code, run the tests, and check the docs build before opening. A plausible-looking patch that was never executed is not ready.
- **Keep it small and focused.** One change per PR. Do not let an agent sprawl across the tree or reformat files it was not asked to touch.
- **No invented APIs.** Do not add behavior, arguments, or examples that the implementation does not actually support. If something cannot be verified, say so rather than papering over it.

The principles below are the standard a change must meet. They are the same rules the core team follows and the same rules encoded in the repository's `AGENTS.md`.

## Design principles

Sleepwalker is research code. The goals, in rough priority order:

- **Flexible and easy to understand** over clever or compact.
- **Fewer abstractions**, unless one is directly called for.
- **APIs may break.** There is no obligation to preserve backwards compatibility unless a page says so.
- **Fail loud and early.** Bad input should raise a clear error at the point of the mistake, not silently produce wrong numbers three layers down.
- **API first, tooling second.** Dataset classes, models, trainers, loaders, and deployment functions must stay directly usable from Python. Commands and YAML construct those objects for repeatable runs; they must not contain behavior a Python caller cannot reach.

A few concrete conventions:

- Imports go at the top of the file; no in-function imports.
- No leading underscores in function names.
- Do not force a line wrap for overly long lines.
- Do not use `getattr` inside a class for its own members. Class members always exist and are set to a correct value; in the worst case they are set to `None` in the constructor.

If you want to plan or discuss an implementation before writing it, open an issue or a draft — a brief summary of the intended approach is welcome before code.

## Code architecture

Sleepwalker is API first and tooling second. Dataset objects turn a patient path and a window index into tensors plus metadata. Models consume those tensors. Trainers optimize models and describe classification outputs. `RunCfg` accepts objects that have already been constructed and initialized. Deployment saves a model together with a dataset clone so inference can repeat the EDF preparation used for training.

```text
src/sleepwalker/
├── core/          # EDF signal reading and metadata
├── datasets/      # dataset classes, channel definitions, normalization
├── models/        # PyTorch models and preprocessors
├── trainer/       # optimization and task-specific evaluation
├── training/      # loaders, samplers, callbacks, file selection
├── deployment/    # model packages and prediction formatting
└── cli/           # command implementations and YAML construction
```

The command-line code may import all of these modules. Core modules must not depend on the command-line configuration reader.

**Change the smallest relevant component:**

- Dataset-specific file and annotation rules go in a dataset class.
- Signal normalization goes in `datasets/normalizer`.
- Tensor preprocessing that is part of a model goes in `models/preprocessors`.
- Sampling policy goes in `training/loader.py` or `training/samplers.py`.
- Task losses, metrics, and output descriptions go in a trainer.
- Long-format prediction conversion goes in `deployment`.

Prefer plain callables and PyTorch objects when a new building block is needed; a function or an `nn.Module` that a caller can compose is better than a new configuration layer.

## Repository organization

Sleepwalker contains code that is useful across sleep-analysis studies. A paper repository should depend on a pinned Sleepwalker revision and contain the study-specific material.

Keep these in Sleepwalker:

- EDF reading and dataset implementations;
- generally useful models, trainers, metrics, and preprocessing;
- model export and prediction code;
- focused examples and public documentation.

Keep these in the study repository:

- experiment queues and exact paper configurations;
- study-specific data splits and cohort rules;
- result tables, plotting scripts, and manuscripts;
- intermediate artifacts used only to reproduce the paper.

This boundary lets a paper preserve its exact environment without freezing framework development, and keeps the installed Python package free of experiment-specific entry points.

## Writing documentation

Write public pages for a researcher trying to complete a task. Start with the object, command, or file they need, and use executable examples built from the current API.

A tutorial should produce a working result from one realistic example. A how-to guide should solve one named problem without turning into a tour of the source tree. State required inputs, expected outputs, shapes, units, and important failure cases. Do not invent pseudocode that looks executable. If behavior cannot be verified from the implementation or a test, mark it as unknown.

Docstrings and comments:

- Public docstrings state accepted values, shapes, units, returned values, mutation, and failure modes.
- Module docstrings identify the code in the file without narrating the repository history.
- Inline comments explain non-obvious reasons, numerical constraints, or scientific assumptions.
- Remove comments that only restate the next line of code.

Update reference text when a signature or return value changes; update a task page when the command a user should run changes.

## Preview your changes

Build and serve the documentation locally before opening a PR:

```bash
python -m pip install -e ".[docs]"
zensical build --clean
zensical serve
```

The build reports broken internal links and reference errors; fix them before opening. Run the test suite the same way you would for any code change.
