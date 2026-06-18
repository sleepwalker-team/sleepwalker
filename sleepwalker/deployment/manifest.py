"""Schema types for the Sleepwalker expert package manifest.

The deployment package is intentionally split into two conceptual layers:

1. The package directory in ``package.py``:
   a folder containing tensors, optional optimizer/scheduler state, and one
   manifest file.
2. The manifest schema in this module:
   a JSON-serializable description of what the package contains and how it can
   be reconstructed.

This separation is deliberate. The manifest types are pure data structures plus
small serialization helpers, while ``package.py`` owns filesystem I/O and torch
state persistence. Keeping the schema light makes it easier to reason about the
contract independently of storage mechanics.

What users should expect from the manifest:

- It describes *compatibility expectations* for reload and inference, such as
  required channels, time resolution, callable hooks, and output structure.
- It records enough builder information to reconstruct code-defined objects
  inside the same repository revision family.
- It is a repo-local handoff format, not a stable cross-project interchange
  standard.

What users should not expect:

- No promise of forward compatibility across arbitrary future schema changes.
- No embedding of all Python logic needed to run the model without the source
  repository. The builder points back into repository code.
- No claim that the manifest alone is sufficient to verify semantic equivalence
  of preprocessing implementations. It records callable references and key
  shape/config contracts, not full source snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import functools
import json
from typing import Any, Optional

import numpy as np
import pandas as pd


def _callable_ref(func: Any) -> str:
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if module is None or qualname is None:
        raise ValueError(f"Cannot serialize callable reference for {func!r}")
    return f"{module}:{qualname}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if callable(value):
        return _callable_ref(value)
    return str(value)


def serialize_callable(func: Any) -> Any:
    if func is None:
        return None
    if isinstance(func, functools.partial):
        return {
            "type": "partial",
            "function": _callable_ref(func.func),
            "args": [_jsonable(arg) for arg in func.args],
            "keywords": _jsonable(func.keywords or {}),
        }
    return {
        "type": "callable",
        "function": _callable_ref(func),
    }


@dataclass
class BuilderSpec:
    """Describe how to reconstruct an expert's Python objects.

    The builder is the code-aware part of the deployment story. Instead of
    trying to serialize arbitrary Python objects, the package stores a module,
    a function name, and a small config payload. Loading a package imports that
    function and calls it to rebuild the model/trainer/template objects before
    applying tensor state.

    Field details:

    - ``module``: Import path containing the builder function.
    - ``function``: Attribute name resolved from ``module``.
    - ``config``: JSON-serializable configuration passed back into the builder
      at load time.

    This means packages are intentionally coupled to repository code. That is a
    conscious tradeoff: it keeps snapshots small and transparent, but it also
    means that users should reload them inside a compatible codebase.
    """

    module: str
    function: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "function": self.function,
            "config": _jsonable(self.config),
        }


@dataclass
class ExpertManifest:
    """Structured description of one saved expert package.

    The manifest is the authoritative index for the package directory. The
    payload tells loaders which files exist, how model inputs/outputs are
    shaped, which builder should be called, and what assumptions an inference
    caller must satisfy.

    Layout reasoning:

    - ``model`` stores model-class identity plus the state-dict file and input
      spec emitted by the model object itself.
    - ``input_contract`` and ``preprocessing_contract`` capture the practical
      compatibility checks a caller cares about: channel order, sampling rate,
      window size, and preprocessing hooks.
    - ``output_contract`` documents what the model/trainer pair emits, which is
      especially important for multitask and non-standard heads.
    - ``training_state`` points to optional optimizer/scheduler state and names
      the trainer/dataset classes used when the snapshot was created.
    - ``metadata`` is intentionally open-ended and reserved for experiment
      provenance, human-readable context, and deployment annotations.

    Field details:

    - ``format_version``: Schema identifier for the manifest format itself.
    - ``expert_name``: User-facing name of the packaged expert.
    - ``task``: Short task label such as ``sleep`` or ``arousal``.
    - ``source_git_commit``: Best-effort source revision for traceability.
    - ``builder``: Optional :class:`BuilderSpec` used to reconstruct live
      Python objects at load time.
    - ``model``: Dictionary describing the model class, state file, and model
      input specification.
    - ``input_contract``: Required input layout and dataset-facing dimensions
      expected by the reloaded model.
    - ``preprocessing_contract``: Dataset-side and model-side preprocessing
      expectations, including callable references where applicable.
    - ``output_contract``: Description of the logits/tasks/labels produced by
      the model-trainer combination.
    - ``training_state``: References to optional optimizer/scheduler state plus
      trainer and dataset class provenance.
    - ``metadata``: Free-form JSON-compatible annotations preserved verbatim.

    The manifest is descriptive, not magical. It helps detect mismatches and
    explains expectations, but it does not sandbox code or make arbitrary model
    snapshots portable outside a compatible Sleepwalker checkout.
    """

    format_version: str
    expert_name: str
    task: str
    source_git_commit: Optional[str]
    builder: Optional[BuilderSpec]
    model: dict[str, Any]
    input_contract: dict[str, Any]
    preprocessing_contract: dict[str, Any]
    output_contract: dict[str, Any]
    training_state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["builder"] = None if self.builder is None else self.builder.to_dict()
        return _jsonable(payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, sort_keys=True)
