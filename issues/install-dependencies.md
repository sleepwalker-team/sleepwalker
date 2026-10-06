# Simplify installation dependencies

## Goal

Make the default installation support the standard framework functions, including all supplied dataset adapters.

## Requirements

- Review dependencies in `pyproject.toml` against package imports and public functions.
- Move required dataset dependencies from the `datasets` extra to `project.dependencies`.
- Include dependencies required for SleepFM and OSF in `project.dependencies`.
- Remove unused dependencies and redundant extras.
- Keep an extra only when its feature is optional. State what that feature provides.
- Coordinate foundation-model dependencies with [the model factory specification](foundation-model-factory.md). SleepGPT support is optional for this work.
- Update installation commands in the README and documentation.

## Completion criteria

- In a clean environment, `pip install .` supports dataset imports and a small local data load.
- The default installation supports the SleepFM and OSF factory when that feature is complete.
- Each retained extra has one documented purpose and a working installation command.
- Missing optional dependencies cause a clear error when the user selects the related feature.
- tracking and dev are the only optional dependencies 
