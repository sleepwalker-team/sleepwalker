"""Helpers for jsonl artifacts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Encode the scalar/container types commonly written by experiments."""

    def default(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timedelta):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        return super().default(value)


def json_ready(value: Any) -> Any:
    """Return ``value`` normalized with the repository JSON encoder."""
    return json.loads(json.dumps(value, cls=NumpyEncoder))


def read_jsonl(filename: str) -> pd.DataFrame:
    """Read newline-delimited JSON records into a DataFrame."""
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(data)


def append_to_jsonl(filename: str, record: dict):
    """Append one JSON record to a `.jsonl` artifact file.

    Args:
        filename: Artifact stem without the `.jsonl` suffix.
        record: Mapping to serialize. Common NumPy and pandas scalar types are
            normalized through a custom encoder.
    """
    with open(f"{filename}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n")
