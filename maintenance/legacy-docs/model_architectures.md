# Model Architectures

This page summarizes the main architecture modules documented so far. These notes describe the current repository implementations, not exact claims of paper-faithful reproduction.

## SleepTransformer

`SleepTransformer` uses:

- a spectrogram front-end
- normalization
- a frame-level transformer with attention pooling
- a sequence-level transformer
- a configurable temporal output strategy

It is used directly in sleep-staging scripts and also as a submodel in current multitask Ruhrland experiments.

## AttnSleep

`AttnSleep` combines:

- a multi-resolution CNN front-end
- squeeze-and-excitation residual blocks
- a temporal-context encoder with self-attention

The implementation follows the broad paper structure but should be documented as the repository's current implementation rather than a certified reproduction.

## MRASleepNet

`MRASleepNet` combines:

- a convolutional front-end
- a multi-resolution attention block
- a gMLP-style block
- a concatenated global feature vector

It uses `NormalizeAlongDim` as a preprocessor.

## SeqSleepNet

`SeqSleepNet` combines:

- a spectrogram front-end
- learned triangular filterbanks
- a frame-level recurrent encoder with attention
- an epoch-level recurrent encoder

It supports several output strategies, including sequence outputs.

## USleep

The current `USleep` module is a flexible U-shaped convolutional architecture with:

- a `RobustScaler` preprocessor
- encoder/decoder blocks
- epoch-based temporal segmentation
- several output strategies

The code itself explicitly notes that this implementation is more flexible than the exact original paper architecture.

## Caveat

The model files include paper references and expected performance notes, but those should be read as implementation context, not as guarantees for the current repository configuration.
