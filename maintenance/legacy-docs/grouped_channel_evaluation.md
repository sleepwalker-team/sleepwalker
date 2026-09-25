# Grouped-Channel Evaluation

This page summarizes the current grouped-channel and gradient-reversal path.

## Background

The repository has two related ideas:

- grouped channel selection at the dataset layer
- repeated-view or multi-dataset evaluation at the trainer layer

## Gradient Reversal Trainer

`sleepwalker.trainer.NegativeGroupedChanelMulticlassTrainer.GradReverseTrainer` extends `MulticlassTrainer` with:

- an auxiliary domain head
- gradient reversal on the shared features
- a batch field named `dataset` used as the domain label

This is currently an experiment-oriented trainer, used in the sleep-staging script through the `--gradrev` path.

## What It Optimizes

The trainer combines:

- the main task loss
- an auxiliary domain-classification loss during training

At evaluation time, the domain loss is disabled.

## Current Status

This path is documented because it is explicit in code and referenced by active scripts, but it should still be treated as a research workflow rather than a stable training mode.
