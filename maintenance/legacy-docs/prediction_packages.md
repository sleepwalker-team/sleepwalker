# Prediction Packages

This page summarizes the current `.swmodel` export and load path.

## What A Package Contains

An exported prediction package currently stores:

- metadata in `meta.json`
- model-card markdown in `model_card.md`
- model weights in `model.pt`
- serialized model object in `model.pkl`
- serialized trainer object in `trainer.pkl`
- serialized unlabelled dataset template in `unlabelled_dataset.pkl`

This format is implemented in `sleepwalker/deployment/package.py`.

## Purpose

The package exists so inference can be run later without reconstructing the original training script by hand.

In practice, the package bundles:

- enough model state to restore weights
- enough trainer state to format predictions
- enough dataset configuration to initialize EDF files at prediction time

## Current Status

What is confirmed:

- export and load round-trips are covered by tests
- package metadata preserves key source information
- the embedded unlabelled dataset template is used during prediction

What is still unclear or unstable:

- long-term backward compatibility
- deployment guarantees across different environments
- whether the current file format should be treated as durable outside the lab workflow

## Relationship To `predict.py`

`predict.py` is the main consumer of exported packages in the current tree.

The flow is:

1. load one or more `.swmodel` bundles
2. initialize the embedded unlabelled dataset template for an EDF file
3. run trainer prediction
4. normalize prediction frames
5. write one output table per EDF file
