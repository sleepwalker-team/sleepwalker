# Use Aeon inside Sleepwalker

Status: Draft for review.

## Goal

Use [Aeon](https://www.aeon-toolkit.org/en/stable/) classifiers with Sleepwalker datasets and evaluation functions.

## Requirements

- Provide a Python interface to train and evaluate Aeon classifiers inside Sleepwalker.
- Adapt Sleepwalker data to Aeon inputs. Use the selected classifier's training method.
- Follow the [Aeon classifier interface](https://www.aeon-toolkit.org/en/stable/api_reference/classification.html).
- Define input shapes, channel order, class order, and probability output shapes.
- Preserve signal preparation and patient identifiers. Keep each patient in one data split.
- Define supported multiclass and multilabel targets. Reject unsupported targets before training. Do not change label meaning.
- State the supported Aeon versions and installation requirements.

## Completion criteria

- A small multivariate example trains and evaluates an Aeon classifier through Sleepwalker.
- Checks confirm data shapes, class order, probability values, and errors for unsupported targets.

## Input needed

Select the first Aeon classifiers and confirm whether multilabel classification is required.

Track the reverse direction in [the Aeon contribution research issue](sleepwalker-in-aeon.md).
