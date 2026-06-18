"""Helpers for jsonl artifacts."""

import json
import numpy as np
import pandas as pd


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
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.Timedelta):
                return str(o)
            if isinstance(o, pd.Timestamp):
                return o.isoformat()
            return super().default(o)

    with open(f"{filename}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n")
