# Package manifest

`manifest.json` can be read without loading the serialized model.

| Field | Value |
| --- | --- |
| `format_version` | Package format accepted by the loader. |
| `name` | Name supplied during export. |
| `task` | Single-head task name, or `null`. |
| `payload` | Serialized payload filename, currently `model.pt`. |
| `sha256` | SHA256 digest of the payload. |
| `model_class` | Fully qualified Python class of the model. |
| `dataset_class` | Fully qualified Python class of the saved dataset. |
| `capabilities` | Available outputs: `classification`, `embeddings`, or both. |
| `input_channels` | Ordered logical inputs, or inputs grouped by graph node. |
| `classification` | Class order and output timing, or `null`. |
| `config` | JSON-compatible metadata supplied to the exporter. |
| `git_commit` | Source revision detected during export, when available. |

Example:

```python
import json
from pathlib import Path


manifest = json.loads(Path("artifacts/model/manifest.json").read_text())
print(manifest["input_channels"])
print(manifest["classification"])
```

The digest detects file changes; it does not make `model.pt` safe. Only load packages from a trusted source.
