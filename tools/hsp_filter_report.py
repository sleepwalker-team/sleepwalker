#!/usr/bin/env python3
"""Report why HSP EDF files pass or fail channel/annotation preflight checks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from os.path import basename, dirname, exists, join
from pathlib import Path
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.HSP import get_channels
from sleepwalker.datasets.utils import get_edf_files_in_repo


DEFAULT_ROOT = "/raid/sleepwalker/hsp"


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    return cfg


def expected_annotation_path(edf_path: str) -> str | None:
    bn = basename(edf_path)
    candidates = [
        join(dirname(edf_path), bn.replace("eeg", "annotations").replace(".edf", ".csv")),
        join(dirname(edf_path), bn.replace("-psg_eeg.edf", "_Xltek.csv")),
    ]
    for candidate in candidates:
        if exists(candidate):
            return candidate
    return None


def requested_channel_groups(cfg: dict[str, Any]) -> dict[str, list[str]]:
    requested = get_channels(
        cfg.get("channels", ["eeg", "eog", "chin_emg", "ECG"]),
        grouped=bool(cfg.get("grouped", False)),
        normalize=False,
        sample_frequency=float(cfg.get("sample_frequency", 100)),
        override_normalize=None,
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    for channel_cfg in requested:
        key = channel_cfg.group or channel_cfg.name
        grouped[key].append(channel_cfg.name)
    return dict(grouped)


def diagnose_file(edf_path: str, required: dict[str, list[str]]) -> dict[str, Any]:
    annotation_path = expected_annotation_path(edf_path)
    if annotation_path is None:
        return {"path": edf_path, "usable": False, "reason": "missing_annotation"}

    try:
        meta = read_edf_meta(edf_path)
    except Exception as exc:
        return {"path": edf_path, "usable": False, "reason": "meta_error", "error": repr(exc)}

    available = set(meta["signals"])
    missing_groups = {
        group: names
        for group, names in required.items()
        if not any(name in available for name in names)
    }
    if missing_groups:
        return {
            "path": edf_path,
            "usable": False,
            "reason": "missing_required_channel_group",
            "missing_groups": missing_groups,
            "available_channels": sorted(available),
        }

    return {
        "path": edf_path,
        "usable": True,
        "reason": "usable",
        "annotation_path": annotation_path,
        "available_channels": sorted(available),
    }


def summarize(records: list[dict[str, Any]], top_channels: int) -> dict[str, Any]:
    reasons = Counter(record["reason"] for record in records)
    missing_groups = Counter()
    channel_counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for group in record.get("missing_groups", {}):
            missing_groups[group] += 1
        for channel in record.get("available_channels", []):
            channel_counter[channel] += 1
        reason = record["reason"]
        if len(examples[reason]) < 5:
            examples[reason].append(record["path"])

    return {
        "total": len(records),
        "usable": reasons.get("usable", 0),
        "reasons": dict(reasons),
        "missing_required_groups": dict(missing_groups),
        "top_channels": channel_counter.most_common(top_channels),
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config with HSP root/channels/grouped settings.")
    parser.add_argument("--root", type=str, default=None, help="Override HSP root.")
    parser.add_argument("--channels", nargs="+", default=None, help="Override required HSP channel groups/names.")
    parser.add_argument("--grouped", action="store_true", help="Treat channel groups as grouped model inputs.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional EDF cap for quick diagnostics.")
    parser.add_argument("--top-channels", type=int, default=40)
    parser.add_argument("--json-out", type=str, default=None, help="Optional path for detailed JSON output.")
    args = parser.parse_args()

    cfg = read_config(args.config)
    if args.root is not None:
        cfg["root"] = args.root
    if args.channels is not None:
        cfg["channels"] = args.channels
    if args.grouped:
        cfg["grouped"] = True
    cfg.setdefault("root", DEFAULT_ROOT)
    cfg.setdefault("channels", ["eeg", "eog", "chin_emg", "pulse"])
    cfg.setdefault("grouped", False)
    cfg.setdefault("sample_frequency", 100)

    edf_files = get_edf_files_in_repo(cfg["root"], recursive=True)
    if args.max_files is not None:
        edf_files = edf_files[: args.max_files]

    required = requested_channel_groups(cfg)
    records = [diagnose_file(edf_path, required) for edf_path in edf_files]
    payload = {
        "config": {
            "root": cfg["root"],
            "channels": cfg["channels"],
            "grouped": cfg["grouped"],
            "sample_frequency": cfg["sample_frequency"],
        },
        "required_channel_groups": required,
        "summary": summarize(records, args.top_channels),
        "records": records,
    }

    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2, sort_keys=True))
    if args.json_out is not None:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
