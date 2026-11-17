# Trainer Lifecycle

This page documents the current shared trainer flow used in the repository.

## Layering

The trainer stack currently has three important layers:

- `BaseTrainer`: shared fit/test/predict mechanics
- concrete trainers such as `MulticlassTrainer` and `MultiLabelTrainer`
- `Run.py`: one level above trainers, coordinating loaders, logging, evaluation, and optional package export

## Shared Flow

The common training flow is:

1. build train and optional validation loaders
2. run trainer-specific warmup
3. warm model preprocessors
4. fit across epochs
5. evaluate test datasets
6. optionally export a prediction package

This split lets task scripts focus on experiment configuration while reusing one training skeleton.

## Trainer Warmup

Trainer warmup is currently used for:

- estimating class counts
- configuring class-weighted losses
- preparing conditioning masks or balancing behavior

Preprocessor warmup is separate and is handled through `BaseTrainer`.

## Repeated Windows

`BaseTrainer` supports repeated-window evaluation through `RepeatSampler`.

Observed behavior:

- a logical sample may appear several times in one batch
- model outputs are averaged back together before prediction or evaluation

This is used in grouped-channel or repeated-view evaluation paths.

## Current Caveats

- return structures are convention-based rather than defined by a formal protocol
- some trainer behavior mutates dataset callbacks for balancing
- the training scripts remain the clearest examples of intended usage
