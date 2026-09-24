# Extract embeddings from one EDF

`PackagedModel` does not currently provide an `embed_patient()` method. For a package with an embedding model, clone its saved dataset settings, build an ordered loader, and call `model.features()`.

```python
import pandas as pd
import torch

from sleepwalker.datasets import batch_collate
from sleepwalker.deployment import load_packaged_model
from sleepwalker.models.BaseModel import EmbeddingModel
from sleepwalker.training.loader import build_loader


def embed_patient(package_path, edf_path, device="cpu"):
    package = load_packaged_model(package_path, map_location=device)
    if not isinstance(package.model, EmbeddingModel):
        raise TypeError(f"{package.model.__class__.__name__} is not an EmbeddingModel")

    dataset = package.dataset.clone()
    dataset.initialize([edf_path], num_workers=0, strict=True)
    package.assert_compatible(dataset)

    loader = build_loader(
        dataset,
        batch_size=64,
        num_workers=0,
        n_samples=None,
        collate_fn=batch_collate,
        sampling="sequential",
        seed=0,
        rejection_strategy="none",
    )

    package.model.to(device)
    package.model.eval()
    frames = []
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            values = package.model.features(batch["data"].to(device)).detach().cpu()
            frame = pd.DataFrame(values.numpy(), columns=[f"embedding_{i}" for i in range(values.shape[1])])
            frame.insert(0, "time", pd.to_datetime(batch["time"]))
            frame.insert(0, "patient", [str(patient) for patient in batch["patient"]])
            frames.append(frame)

    if not frames:
        raise ValueError(f"No valid windows found in {edf_path}")
    return pd.concat(frames, ignore_index=True)


embeddings = embed_patient("artifacts/embedding-model", "recordings/night.edf", device="cpu")
embeddings.to_parquet("night-embeddings.parquet", index=False)
```

Each row describes one input window. `time` is the start time of that window. The `embedding_*` columns follow the order returned by the model; the package does not attach names or physical meanings to individual dimensions.

`model.features()` also applies preprocessors stored on the model. Pass the dataset's `data` tensor directly, as shown above, instead of calling a preprocessor separately.

This method applies to classes that implement `EmbeddingModel`. A package advertising `classification` alone may not expose embeddings. Check `package.capabilities` or [inspect the package](inspect-package.md) first.
