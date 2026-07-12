#!/usr/bin/env python3
"""Report why HSP EDF files pass or fail channel/annotation preflight checks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.HSP import (
    get_hsp_annotation_label_counts,
    get_annotated_hsp_edf_files,
    get_channels,
    get_hsp_annotation_path,
)
from sleepwalker.datasets.utils import get_edf_files_in_repo


DEFAULT_ROOT = "/raid/sleepwalker/hsp"
SLEEP_LABELS = ["n1", "n2", "n3", "rem"]
BREATHING_LABELS = ["apnea", "obstructive-apnea", "central-apnea", "mixed-apnea", "hypopnea"]
TASK_DEFAULTS = {
    "sleep": {
        "channels": ["eeg"],
        "grouped": True,
        "positive_labels": ["wake", "n1", "n2", "n3", "rem"],
        "required_any_labels": ["wake", "n1", "n2", "n3", "rem"],
    },
    "arousal": {
        "channels": ["eeg", "eog", "chin_emg"],
        "grouped": True,
        "positive_labels": ["arousal"],
        "required_any_labels": SLEEP_LABELS,
    },
    "breathing": {
        "channels": ["abdomen", "chest", "airflow", "spo2"],
        "grouped": True,
        "positive_labels": BREATHING_LABELS,
        "required_any_labels": SLEEP_LABELS,
    },
    "desaturation": {
        "channels": ["spo2"],
        "grouped": True,
        "positive_labels": ["desaturation"],
        "required_any_labels": SLEEP_LABELS,
    },
}


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    return cfg


def expected_annotation_path(edf_path: str) -> str | None:
    return get_hsp_annotation_path(edf_path)


def apply_task_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    task = cfg.get("task")
    if task is None:
        return cfg
    if task not in TASK_DEFAULTS:
        raise ValueError(f"Unknown HSP task '{task}'. Known tasks: {sorted(TASK_DEFAULTS)}")

    for key, value in TASK_DEFAULTS[task].items():
        cfg.setdefault(key, value)
    return cfg


def requested_channel_groups(cfg: dict[str, Any]) -> dict[str, list[str]]:
    requested = get_channels(
        cfg.get("channels", ["eeg", "eog", "chin_emg"]),
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


def annotation_label_counts(annotation_path: str) -> dict[str, int]:
    return get_hsp_annotation_label_counts(annotation_path)


def diagnose_file(edf_path: str, required: dict[str, list[str]], cfg: dict[str, Any]) -> dict[str, Any]:
    annotation_path = expected_annotation_path(edf_path)
    if annotation_path is None:
        return {"path": edf_path, "usable": False, "reason": "missing_annotation"}

    try:
        label_counts = annotation_label_counts(annotation_path)
    except Exception as exc:
        return {"path": edf_path, "usable": False, "reason": "label_error", "error": repr(exc)}

    labels = set(label_counts)
    required_any_labels = set(cfg.get("required_any_labels") or [])
    if required_any_labels and labels.isdisjoint(required_any_labels):
        return {
            "path": edf_path,
            "usable": False,
            "reason": "missing_required_labels",
            "annotation_path": annotation_path,
            "required_any_labels": sorted(required_any_labels),
            "label_counts": label_counts,
        }
    try:
        meta = read_edf_meta(edf_path)
    except Exception as exc:
        return {
            "path": edf_path,
            "usable": False,
            "reason": "meta_error",
            "error": repr(exc),
            "annotation_path": annotation_path,
            "label_counts": label_counts,
        }

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
            "annotation_path": annotation_path,
            "label_counts": label_counts,
        }

    return {
        "path": edf_path,
        "usable": True,
        "reason": "usable",
        "annotation_path": annotation_path,
        "available_channels": sorted(available),
        "label_counts": label_counts,
    }


def summarize(edf_file_reports: list[dict[str, Any]], top_channels: int, positive_labels: list[str]) -> dict[str, Any]:
    reasons = Counter(record["reason"] for record in edf_file_reports)
    missing_groups = Counter()
    channel_counter = Counter()
    label_counter = Counter()
    positive_edf_file_counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for record in edf_file_reports:
        for group in record.get("missing_groups", {}):
            missing_groups[group] += 1
        for channel in record.get("available_channels", []):
            channel_counter[channel] += 1
        for label, count in record.get("label_counts", {}).items():
            label_counter[label] += count
            if label in positive_labels and count > 0:
                positive_edf_file_counter[label] += 1
        reason = record["reason"]
        if len(examples[reason]) < 5:
            examples[reason].append(record["path"])

    positive_edf_files_any = 0
    positive_label_set = set(positive_labels)
    if positive_label_set:
        positive_edf_files_any = sum(
            any(label in positive_label_set and count > 0 for label, count in record.get("label_counts", {}).items())
            for record in edf_file_reports
            if record["reason"] == "usable"
        )

    return {
        "total_edf_files": len(edf_file_reports),
        "usable_edf_files": reasons.get("usable", 0),
        "usable_edf_files_with_positive_mapped_label": positive_edf_files_any,
        "reasons": dict(reasons),
        "missing_required_groups": dict(missing_groups),
        "positive_edf_file_counts": dict(positive_edf_file_counter),
        "top_labels": label_counter.most_common(40),
        "top_channels": channel_counter.most_common(top_channels),
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config with HSP root/channels/grouped settings.")
    parser.add_argument("--task", type=str, choices=sorted(TASK_DEFAULTS), default=None, help="Apply default labels/channels for an HSP task.")
    parser.add_argument("--root", type=str, default=None, help="Override HSP root.")
    parser.add_argument("--channels", nargs="+", default=None, help="Override required HSP channel groups/names.")
    parser.add_argument("--grouped", action="store_true", help="Treat channel groups as grouped model inputs.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional EDF cap for quick diagnostics.")
    parser.add_argument(
        "--all-edfs",
        action="store_true",
        help="Inspect all EDFs instead of only same-record annotation-paired EDFs.",
    )
    parser.add_argument("--top-channels", type=int, default=40)
    parser.add_argument("--json-out", type=str, default=None, help="Optional path for detailed JSON output.")
    args = parser.parse_args()

    cfg = read_config(args.config)
    if args.task is not None:
        cfg["task"] = args.task
    if args.root is not None:
        cfg["root"] = args.root
    if args.channels is not None:
        cfg["channels"] = args.channels
    if args.grouped:
        cfg["grouped"] = True
    cfg.setdefault("root", DEFAULT_ROOT)
    cfg.setdefault("sample_frequency", 100)
    cfg.setdefault("annotated_only", True)
    cfg = apply_task_defaults(cfg)
    cfg.setdefault("grouped", False)
    cfg.setdefault("channels", ["eeg", "eog", "chin_emg"])

    annotated_only = bool(cfg.get("annotated_only", True)) and not args.all_edfs
    if annotated_only:
        edf_files = get_annotated_hsp_edf_files(cfg["root"], recursive=True)
    else:
        edf_files = get_edf_files_in_repo(cfg["root"], recursive=True)
    if args.max_files is not None:
        edf_files = edf_files[: args.max_files]

    required = requested_channel_groups(cfg)
    positive_labels = list(cfg.get("positive_labels") or [])
    edf_file_reports = [diagnose_file(edf_path, required, cfg) for edf_path in edf_files]
    payload = {
        "config": {
            "root": cfg["root"],
            "task": cfg.get("task"),
            "channels": cfg["channels"],
            "grouped": cfg["grouped"],
            "sample_frequency": cfg["sample_frequency"],
            "annotated_only": annotated_only,
            "required_any_labels": cfg.get("required_any_labels", []),
            "positive_mapped_labels": positive_labels,
        },
        "required_channel_groups": required,
        "summary": summarize(edf_file_reports, args.top_channels, positive_labels),
        "edf_file_reports": edf_file_reports,
    }

    print(json.dumps({k: v for k, v in payload.items() if k != "edf_file_reports"}, indent=2, sort_keys=True))
    if args.json_out is not None:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
