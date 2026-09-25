# Development Notes

This repository appears to be developed primarily for internal lab use. The guidance below is based on observed code, tests, and repository contents rather than a formal contributor guide.

## Audience

Current code organization strongly suggests the main audience is developer and ML-engineer users working in the same environment as the repository authors.

Evidence:

- several training scripts embed task-specific configuration directly in Python
- some scripts assume local dataset roots such as `/raid/sleepwalker/...`
- the package surface is incomplete compared with the number of internal modules in use

## Working Style

The common pattern is:

1. choose or build a dataset adapter
2. configure channels, mappings, and preparation hooks
3. choose a model and trainer
4. call `sleepwalker.trainer.Run.run(...)`
5. optionally export a prediction package

## Repository Hygiene

The repository currently contains source code together with notebooks, logs, jsonl outputs, sqlite artifacts, model bundles, and temporary files. Documentation and maintenance work should keep distinguishing:

- core source modules
- test fixtures
- generated artifacts
- exploratory notebooks and utilities

## API Stability

There is no stable public API yet.

Practical implications:

- internal imports are common
- package `__init__` files expose only part of what scripts use
- task scripts and prediction tooling should be documented as current workflows, not long-term compatibility guarantees

## Documentation Approach

For this repository, good internal documentation should:

- explain why a module exists
- describe actual data flow and lifecycle hooks
- avoid pretending that all scripts are stable interfaces
- mark uncertain or evolving areas explicitly
