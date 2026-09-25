# Testing Notes

The test suite mixes self-contained unit tests with environment-dependent dataset checks.

## Well-Covered Areas

Observed tests provide meaningful coverage for:

- dataset window construction and lazy loading behavior
- target preparation helpers
- multitask target construction and conditioning logic
- model preprocessors
- selected model constructors and forward passes
- prediction-package export and loading
- prediction table formatting
- selected Ruhrland-specific helpers

## Environment-Dependent Areas

Several dataset adapter tests skip unless the corresponding dataset path is available through the local environment. Those tests are useful as integration checks, but they are not self-contained repository tests.

## Key Test Files

- `tests/test_datasets.py`
- `tests/test_deployment.py`
- `tests/test_predict.py`
- `tests/test_multilabel_trainer.py`
- `tests/test_targets.py`
- `tests/test_preprocessors.py`
- `tests/test_models.py`

## Current Testing Caveats

- Presence of tests does not imply all scripts are production-ready.
- `predict.py` is tested in helper-level behavior, but actual deployment workflows should still be treated as evolving.
- Many dataset adapters are only lightly documented through integration-style checks or naming conventions, so code reading is still necessary when changing those modules.
