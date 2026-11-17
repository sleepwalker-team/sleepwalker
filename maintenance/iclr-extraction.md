# ICLR extraction checklist

The current directory remains in this checkout until a destination repository is chosen. Packaging discovers the top-level `sleepwalker` package and does not include `iclr2026`.

## Before moving files

- Move every generally useful helper into Sleepwalker. `PairedDataset` and stacking alignment now live under `sleepwalker`; `iclr2026.pipeline` only provides the paper-local import surface.
- Ensure no test under the future Sleepwalker test suite imports `iclr2026`.
- Replace imports from root `tools` with installed `sleepwalker.cli` or public framework modules.
- Decide where model payloads and large result artifacts will be stored.
- Record dataset access instructions without copying restricted data.

## New paper repository contents

- Manuscript, bibliography, figures, and table sources.
- Experiment-specific Python and shell orchestration.
- All ICLR configurations.
- Paper-specific tests.
- An environment lock and a pinned Sleepwalker tag or commit.
- A reproduction README mapping each paper result to its command and expected artifact.

## Final removal from Sleepwalker

After the new repository reproduces its configuration validation and imports Sleepwalker only as a dependency, remove `iclr2026`, ICLR-only tests, README references, and `.gitignore` exceptions from this repository.
