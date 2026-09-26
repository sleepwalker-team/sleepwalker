# Models

All Sleepwalker models follow the same input convention: a `[B, T, C]` tensor (batch, time, channel — the `"BTC"` layout), where \(T = \operatorname{round}(\texttt{total\_input} \cdot f_s)\) and \(C\) is the number of logical channels. What a model does with that tensor is declared through two capability markers: [`ClassifierModel`](#classifier-model) means "`forward()` returns classification logits", and [`EmbeddingModel`](#embedding-model) means the model also exposes `features()` for embedding extraction. A model can be both, which is the common case: the same backbone supports a classification head and downstream embedding use.

Models are selected in YAML by their **fully qualified import path** (`sleepwalker.models.UTime.UTime`); there is no short-name registry. The user-facing guide for these concepts is [Models: classification and embeddings](../how-to/models.md); this page is the generated reference.

## Base classes

### `BaseModel` {#base-model}

::: sleepwalker.models.BaseModel.BaseModel
    options:
      show_root_heading: false
      members:
        - apply_preprocessors
        - compute
        - input_spec

### `EmbeddingModel` {#embedding-model}

::: sleepwalker.models.BaseModel.EmbeddingModel
    options:
      show_root_heading: false

### `ClassifierModel` {#classifier-model}

::: sleepwalker.models.BaseModel.ClassifierModel
    options:
      show_root_heading: false

## Classification models

Every supervised classifier below is sequence-capable: `sequence_len = 1` produces one label per input window, `sequence_len = S` produces \(S\) center-aligned step logits. All return `[B, S, K]` logits from `compute()`.

| Model | Capabilities | Built-in preprocessors | Notes |
| --- | --- | --- | --- |
| [`TinySleepNet`](#tinysleepnet) | classification + embeddings | none | CNN per sub-chunk + optional LSTM. |
| [`AttnSleep`](#attnsleep) | classification + embeddings | none | MRCNN + temporal attention; inverse-log class weights match the paper. |
| [`SleepTransformer`](#sleeptransformer) | classification + embeddings | `Spectrogram`, `Normalize` | Two-level (frame + epoch) transformer; expects `ts_len = epoch_seq_len × epoch_len`. |
| [`SeqSleepNet`](#seqsleepnet) | classification + embeddings | `Spectrogram`, `Normalize` | Recurrent sleep staging model. |
| [`MRASleepNet`](#mrasleepnet) | classification + embeddings | `NormalizeAlongDim` | Fixed `feature_dim = 192`. |
| [`USleep`](#usleep) | classification + embeddings | `RobustScaler` | U-net style temporal model. |
| [`UTime`](#utime) | classification + embeddings | none (configurable) | Pooled or per-epoch output via `epoch_len`; used for breathing/desaturation tasks. |

### `TinySleepNet` {#tinysleepnet}

::: sleepwalker.models.TinySleepNet.TinySleepNet
    options:
      show_root_heading: false

### `AttnSleep` {#attnsleep}

::: sleepwalker.models.AttnSleep.AttnSleep
    options:
      show_root_heading: false

### `SleepTransformer` {#sleeptransformer}

::: sleepwalker.models.SleepTransformer.SleepTransformer
    options:
      show_root_heading: false

### `SeqSleepNet` {#seqsleepnet}

::: sleepwalker.models.SeqSleepNet.SeqSleepNet
    options:
      show_root_heading: false

### `MRASleepNet` {#mrasleepnet}

::: sleepwalker.models.MRASleepNet.MRASleepNet
    options:
      show_root_heading: false

### `USleep` {#usleep}

::: sleepwalker.models.USleep.USleep
    options:
      show_root_heading: false

### `UTime` {#utime}

::: sleepwalker.models.UTime.UTime
    options:
      show_root_heading: false

## Self-supervised models

### `MaskedAutoencoder` {#maskedautoencoder}

A plain `nn.Module` (not a `BaseModel`): preprocessors are a `ModuleDict` keyed by **modality**, and every call takes an explicit modality name — `forward(x, modality)` returns `(patches, mask)` for reconstruction training, `embed(x, modality)` returns latents `[B, N, enc_dim, D]`.

::: sleepwalker.models.MaskedAutoencoder.MaskedAutoencoder
    options:
      show_root_heading: false

### `CLSMaskedAutoencoder` {#clsmaskedautoencoder}

::: sleepwalker.models.CLSMaskedAutoencoder.CLSMaskedAutoencoder
    options:
      show_root_heading: false

## Packaged backbone models

These wrap an existing [`PackagedModel`](api.md#packaged-model) encoder so a pretrained package can serve as the backbone of a new trainable head.

| Model | Head placement | Use case |
| --- | --- | --- |
| [`PackagedEmbeddingModel`](#packaged-embedding-model) | none — the encoder *is* the model | Ship external/foundation encoders as packages. |
| [`PackagedClassifierModel`](#packaged-classifier-model) | one head over concatenated per-window features | Frozen encoder + trained classifier over sliding native windows. |
| [`PackagedSequenceClassifierModel`](#packaged-sequence-classifier-model) | position-wise head over encoder tokens | Per-step predictions aligned to encoder tokens. |

### `PackagedEmbeddingModel` {#packaged-embedding-model}

::: sleepwalker.models.PackagedEmbeddingModel.PackagedEmbeddingModel
    options:
      show_root_heading: false

### `PackagedClassifierModel` {#packaged-classifier-model}

::: sleepwalker.models.PackagedClassifierModel.PackagedClassifierModel
    options:
      show_root_heading: false

### `PackagedSequenceClassifierModel` {#packaged-sequence-classifier-model}

::: sleepwalker.models.PackagedSequenceClassifierModel.PackagedSequenceClassifierModel
    options:
      show_root_heading: false

## Multitask and composition models

`MultiTaskClassifierModel` gives one shared embedding encoder a separate head for each task. It returns task-keyed logits for `MultiLabelTrainer`.

### `MultiTaskClassifierModel` {#multitask-classifier-model}

::: sleepwalker.models.MultiTaskClassifierModel.MultiTaskClassifierModel
    options:
      show_root_heading: false

`StackedClassifierModel` combines existing packaged task experts on an aligned timeline. It adds trainable correction heads over the other experts' class probabilities. `PairedDataset` loads the native input windows each expert needs.

### `StackedClassifierModel` {#stacked-classifier-model}

::: sleepwalker.models.StackedClassifierModel.StackedClassifierModel
    options:
      show_root_heading: false

### `PairedDataset` {#paired-dataset}

::: sleepwalker.datasets.PairedDataset.PairedDataset
    options:
      show_root_heading: false

## Preprocessors

Preprocessors are `nn.Module`s stored on a `BaseModel` and applied inside `forward()` before `compute()`. Each optionally restricts itself to a subset of channel indices on the last tensor dimension; statistics-based steps implement `requires_warmup()`/`update(x)` and are fitted by the trainer's preprocessor warmup pass.

### `Preprocessor` {#preprocessor}

::: sleepwalker.models.preprocessors.Preprocessor.Preprocessor
    options:
      show_root_heading: false

### `Spectrogram` {#spectrogram}

::: sleepwalker.models.preprocessors.Spectrogram.Spectrogram
    options:
      show_root_heading: false

### `WindowedSpectrogram` {#windowed-spectrogram}

::: sleepwalker.models.preprocessors.WindowedSpectrogram.WindowedSpectrogram
    options:
      show_root_heading: false

### `Normalize` {#normalize}

::: sleepwalker.models.preprocessors.Normalize.Normalize
    options:
      show_root_heading: false

### `NormalizeAlongDim` {#normalize-along-dim}

::: sleepwalker.models.preprocessors.NormalizeAlongDim.NormalizeAlongDim
    options:
      show_root_heading: false

### `RobustScaler` {#robust-scaler}

::: sleepwalker.models.preprocessors.RobustScaler.RobustScaler
    options:
      show_root_heading: false

### `EmpiricalClipScaler` {#empirical-clip-scaler}

::: sleepwalker.models.preprocessors.EmpiricalClipScaler.EmpiricalClipScaler
    options:
      show_root_heading: false

### `FixedChannelStandardizer` {#fixed-channel-standardizer}

::: sleepwalker.models.preprocessors.FixedChannelStandardizer.FixedChannelStandardizer
    options:
      show_root_heading: false

### `Crop` {#crop}

::: sleepwalker.models.preprocessors.Crop.Crop
    options:
      show_root_heading: false
