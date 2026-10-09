# Add a foundation-models

Status: Ready to be implemented

## Goal

Create foundation models through the Python API without a separate download or conversion command.

## Requirements

- Provide model definitions in `sleepwalker.models` that returns framework model objects.
- Support SleepFM and OSF. Do not make completion depend on SleepGPT support. SUpport it, if easily possible. Otherwise skip
- Support embedding extraction and use with the existing classifier wrappers.
- Download missing weights on model creation, if user allows. Reuse valid cached weights on later calls.
- Accept one checkpoint file path for loading or download. Disable downloads by default.
- Fix source revisions and verify weight checksums. Reject unknown model names and invalid weights with clear errors.
- Describe each model's channels, units, sample frequency, input shape, and signal preparation.
- Use the framework package export and load functions. 
- Replace the current CLI / tool
- Include required dependencies in the default installation. Coordinate this change with [the installation issue](install-dependencies.md).

## Completion criteria

- SleepFM and OSF load from downloads, cache, and a local checkpoint.
- A saved package loads without network access and produces equivalent embeddings within a stated numerical tolerance.
- Add proepr documentation to the model classes and documenation 
