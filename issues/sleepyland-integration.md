# Add released models to SLEEPYLAND

Status: Backlog research idea. Start this work after the model release.

## Goal

Make released Sleepwalker models available in [SLEEPYLAND](https://github.com/biomedical-signal-processing/sleepyland).

## Requirements

- Select the released sleep-stage models to include.
- Record the Sleepwalker version, model weights, download locations, licenses, and citations.
- Inspect the current SLEEPYLAND model interface and contribution rules.
- Add model adapters that preserve channel selection, units, sample frequency, signal preparation, and stage order.
- Use the released weights. Document installation, model selection, and prediction.
- Prepare and submit a pull request to the SLEEPYLAND repository after release.

## Completion criteria

- Each selected model runs through SLEEPYLAND on a small recording.
- Predictions agree with direct Sleepwalker predictions within a stated numerical tolerance.
- The pull request contains the adapters, checks, and instructions. Record its URL; acceptance depends on SLEEPYLAND maintainers.

## Input needed

Specify the model release and the models to include.
