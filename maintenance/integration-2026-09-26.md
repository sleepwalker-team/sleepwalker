# Framework integration checkpoint — 2026-09-26

The paper experiment is preserved in the sibling Git repository at `../iclr2026`. Its local commits `262fb45` and `7c4a548` preserve the uncommitted experiment sources, paper-specific tests, manuscript sources, and external evaluation configs. The paper's `models/` and `archive/` moved intact and are ignored by its Git repository. Its existing run outputs remain in `sleepwalker/results/iclr2026` and are ignored there. The separate paper repository is a preservation checkpoint; its scripts still assume the old shared checkout and have not been validated as an independent installation.

Sleepwalker integration is on `integration/framework-docs`. The branch contains the framework refactor, the fetched `origin/docs` packaging and documentation migration, and the fetched `origin/main` changes. For the HSP masked-embedding script's scientific conflict, the user chose to retain the current branch's six channel groups. Main's newer logging and trainer behavior were reconciled with those groups.

The merged package uses the `src/` layout and exposes the installed `sleepwalker` CLI. The 14 active `configs/train` examples now pass target resolution to their target callbacks and trainers, with centered target offsets in the trainer contract. The wheel declares `pandas>=2.2,<3`, matching the tested pandas 2.3.3 baseline. Pandas 3 support requires a separate timestamp-unit audit.

Verification:

- Full CPU test suite after the branch merges and supported pandas installation: **619 passed, 16 skipped** (`/tmp/sleepwalker-integration/tests.log`).
- After the masked-autoencoder checkpoint repair: targeted trainer and train-config tests **22 passed** (`focused-tests.log`). The new trainer test exercises training, scheduler steps, and both intermediate and best checkpoint paths.
- `zensical build --clean --strict`: passed with no build issues (`docs-build.log`). Griffe still reports non-fatal docstring parsing warnings in legacy dataset adapters.
- Built a regular wheel with declared runtime dependencies. An isolated `python -I` import with the editable finder disabled loaded the wheel's trainer, augmentation, cache, stacked model, and CLI modules; `sleepwalker --help` dispatch returned 0 (`wheel-build.log`).
- The prior synthetic paper packaging smoke passed before extraction; it has **not** been rerun from the separate paper repository.

No push or main merge has been performed. The paper repository has no remote configured. Before publishing either repository, review the local commits and explicit staging list, then perform the paper integration against a pinned framework revision in a later step.
