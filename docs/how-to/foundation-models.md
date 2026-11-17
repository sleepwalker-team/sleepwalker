# Use foundation models

A foundation model learns features from a large collection of recordings before
it is trained for a specific task. For sleep data, self-supervised pretraining
uses the signals themselves to provide a learning target. This lets the encoder
learn from recordings without sleep-stage or disease labels. You can then reuse
its features to train a prediction head with labelled data, or fine-tune the
encoder and head together.

Sleepwalker provides `SleepFM` and `OSF` as ordinary embedding models. They return
feature vectors through `model.features(x)` and use the same dataset, training,
and package interfaces as other models. Their dependencies are included in the
default installation. The released checkpoints contain encoders; a prediction
task still needs its own head, labels, and evaluation.

## Choose an encoder

[SleepFM: A multimodal sleep foundation model for disease prediction](https://www.nature.com/articles/s41591-025-04133-4)
uses contrastive pretraining across brain activity, respiratory signals, ECG,
and EMG. It learns from agreement between signal modalities recorded at the
same time. The paper evaluates the resulting representations for sleep staging,
apnea classification, and disease prediction. Sleepwalker uses the clinical
base encoder from [sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical).
It returns contextual features for each modality and each five-second interval.

[OSF: On Pre-training and Scaling of Sleep Foundation Models](https://arxiv.org/abs/2603.00190)
studies how pretraining objectives, data sources, and model size affect transfer
to sleep and disease tasks. Sleepwalker uses the released
[OSF-Base DINO encoder](https://huggingface.co/yang-ai-lab/OSF-Base), a vision
transformer that treats signal segments as patches. It returns one feature
vector for each 30-second window.

| Model | Sample frequency | Default input | Embedding output |
| --- | --- | --- | --- |
| SleepFM | 128 Hz | `[B, 38400, 13]`, 300 seconds | `[B, 30720]` |
| OSF | 64 Hz | `[B, 1920, 12]`, 30 seconds | `[B, 768]` |

Inputs follow Sleepwalker's batch, time, channel order. Use
`model.input_channels` for the channel names and order. Both encoders use ECG,
chin and leg EMG, respiratory signals, two EOG channels, and two EEG channels.
SleepFM also uses SpO2; OSF has no SpO2 input. The [model reference](../reference/models.md)
lists the full input contracts.

SleepFM flattens its output in brain activity, respiratory, ECG, EMG order,
then time, then feature. Each five-second interval has 128 features per modality.
Its `ts_len` argument accepts positive multiples of 640, up to 81920 samples.
OSF returns the 768-dimensional CLS token at its fixed window length. Individual
feature dimensions have no assigned physical meaning or class label.

## Load released weights

Choose a checkpoint file. Permit a first download with `allow_download=True`;
later calls can reuse that file without network access:

```python
from sleepwalker.models import OSF, SleepFM

model = OSF(checkpoint="artifacts/weights/osf.pt", allow_download=True)
model = OSF(checkpoint="artifacts/weights/osf.pt")
local_model = SleepFM(checkpoint="weights/sleepfm-best.pt")
```

`checkpoint` is both the file to load and the download destination.
`allow_download` defaults to `False`, so a missing file causes an error. With
`allow_download=True` and no path, the model uses
`~/.cache/sleepwalker/foundation/MODEL/SHA256.pt`. Existing files are verified
and reused. Each file must match the pinned release checksum; an invalid file
causes an error and is not replaced automatically.

Calling `OSF()` or `SleepFM()` without weight options initializes random weights
in training mode. Loading released weights selects evaluation mode and leaves
parameters trainable. To load your own trained weights, use the usual
`model.load_state_dict()` method.

## Prepare an EDF and extract features

The released encoders expect dimensionless signals with recording-wide channel
z-normalization. Configure this in the dataset, together with channel mapping
and resampling. OSF clips normalized values to `[-6, 6]` inside the model.
This example extracts the first OSF window from an EDF:

```python
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.models import OSF

model = OSF(checkpoint="artifacts/weights/osf.pt")
channels = [ChannelConfig(name, [name]) for name in model.input_channels]
dataset = UnlabelledDataset(
    channels=channels,
    sample_frequency=model.sampling_frequency,
    total_input="30s",
    stride="30s",
    resample_type="nearest",
    z_normalize=True,
)
dataset.initialize(["recordings/night.edf"], num_workers=0, strict=True)

with torch.inference_mode():
    embeddings = model.features(dataset[0]["data"].unsqueeze(0))
```

Replace the EDF aliases in `ChannelConfig` with the names in your recordings.
Prepare EEG and EOG montages before this step if the required leads are absent.
For SleepFM's default window, use `SleepFM`, `total_input="300s"`,
`stride="300s"`, and `resample_type="polyphase"`.

These examples use Sleepwalker's EDF pipeline. For comparisons with published
results, match the upstream filtering, normalization, montages, and resampling,
then pass the prepared tensors to `features()`. Pretraining does not remove the
need to evaluate transfer to your recordings and task.

## Save the encoder and train a head

Save the encoder together with the dataset template so later use has the same
input settings. A package includes the encoder parameters and can load without
the original checkpoint or cache:

```python
from sleepwalker.deployment import load_packaged_model, save_packaged_model
from sleepwalker.models.PackagedClassifierModel import PackagedClassifierModel

save_packaged_model(
    "artifacts/osf-encoder",
    name="osf",
    model=model,
    dataset=dataset,
    config={"source": model.source},
)
package = load_packaged_model("artifacts/osf-encoder")
classifier = PackagedClassifierModel(
    package=package,
    classes=["wake", "n1", "n2", "n3", "rem"],
    ts_len=package.model.ts_len,
    freeze_encoder=True,
)
```

The new classifier head needs training on labelled windows. With
`freeze_encoder=True`, only the head learns. Set it to `False` to fine-tune the
encoder too. Use [Train a multiclass model](train-multiclass.md) to configure the
labelled dataset, loss, optimizer, and evaluation. SleepFM can also use
`PackagedSequenceClassifierModel` to predict within a window; its default
geometry is `token_count=60`, `token_dim=128`, and `modality_count=4`.

For features from a full recording, follow [Extract embeddings from one EDF](extract-embeddings.md).
`model.source` records the source revision, checkpoint URL, checksum, and license.

## Licenses

Sleepwalker's own code remains MIT. The adapted SleepFM module retains its
upstream [CC BY-NC 4.0 license](https://github.com/zou-group/sleepfm-clinical/blob/2bcbae04c3592f61352addb7ac3d4193f0a3ca25/LICENSE).
The adapted OSF module is [MIT](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM/blob/d7e4edbc77f2b72713402234036c72b98b9b83ca/LICENSE).
The distribution includes both upstream notices in `LICENSES/`; each notice
applies to its respective model file.
