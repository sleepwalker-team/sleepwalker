#!/usr/bin/env python3
"""Build a deterministic subject-level HSP split shared by all expert scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.HSP import (
    get_annotated_hsp_edf_files,
    get_hsp_annotation_label_counts,
    get_hsp_annotation_path,
)
from hsp_filter_report import TASK_DEFAULTS, apply_task_defaults, requested_channel_groups


TASKS = ("sleep", "arousal", "breathing", "desaturation")


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    return config


def subject_id(edf_path: str) -> str:
    match = re.search(r"/(sub-[^/]+)/", edf_path)
    return match.group(1) if match else Path(edf_path).stem


def split_name(subject: str, seed: str, fractions: dict[str, float]) -> str:
    value = int(
        hashlib.sha1(f"{seed}:{subject}".encode("utf-8")).hexdigest()[:12],
        16,
    ) / float(16**12)
    if value < float(fractions["train"]):
        return "train"
    if value < float(fractions["train"]) + float(fractions["val"]):
        return "val"
    return "test"


def task_requirements(root: str) -> dict[str, dict[str, Any]]:
    requirements = {}
    for task in TASKS:
        config = apply_task_defaults(
            {
                "root": root,
                "task": task,
                "sample_frequency": 100,
                "annotated_only": True,
            }
        )
        requirements[task] = {
            "required_channel_groups": requested_channel_groups(config),
            "required_any_labels": set(config.get("required_any_labels") or []),
            "min_duration_s": config.get("min_duration_s"),
        }
    return requirements


def usable_tasks(
    item: tuple[str, dict[str, dict[str, Any]]],
) -> tuple[str, list[str]]:
    edf_path, requirements = item
    annotation_path = get_hsp_annotation_path(edf_path)
    if annotation_path is None:
        return edf_path, []
    try:
        label_counts = get_hsp_annotation_label_counts(annotation_path)
        labels = set(label_counts)
        metadata = read_edf_meta(edf_path)
        available_channels = set(metadata["signals"])
    except Exception:
        return edf_path, []

    usable = []
    for task, requirement in requirements.items():
        minimum_duration = requirement["min_duration_s"]
        if minimum_duration is not None and float(metadata["duration_s"]) < float(minimum_duration):
            continue
        if requirement["required_any_labels"] and labels.isdisjoint(
            requirement["required_any_labels"]
        ):
            continue
        if any(
            not any(channel in available_channels for channel in alternatives)
            for alternatives in requirement["required_channel_groups"].values()
        ):
            continue
        usable.append(task)
    return edf_path, usable


def build_split(config: dict[str, Any]) -> dict[str, Any]:
    root = str(config["root"])
    seed = str(config.get("seed", "sleepwalker-hsp-v1"))
    fractions = dict(config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1}))
    if abs(sum(float(fractions[key]) for key in ["train", "val", "test"]) - 1.0) > 1e-8:
        raise ValueError("HSP split fractions must sum to one.")

    paths_by_task = {task: [] for task in TASKS}
    edf_files = get_annotated_hsp_edf_files(root, recursive=True)
    requirements = task_requirements(root)
    workers = int(config.get("num_workers_split", 2))
    work = [(path, requirements) for path in edf_files]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.imap_unordered(usable_tasks, work)
            for edf_path, tasks in results:
                for task in tasks:
                    paths_by_task[task].append(edf_path)
    else:
        for item in work:
            edf_path, tasks = usable_tasks(item)
            for task in tasks:
                paths_by_task[task].append(edf_path)

    payload: dict[str, Any] = {
        "root": root,
        "seed": seed,
        "method": "subject-hash",
        "fractions": fractions,
        "tasks": {},
    }
    for task in TASKS:
        partitions = {"train": [], "val": [], "test": []}
        for path in sorted(paths_by_task[task]):
            partitions[split_name(subject_id(path), seed, fractions)].append(path)
        payload["tasks"][task] = {
            "edf_files": partitions,
            "counts": {name: len(paths) for name, paths in partitions.items()},
            "total_edf_files": len(paths_by_task[task]),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    output = Path(args.output or Path(config["output_dir"]) / "hsp_split.yml")
    split = build_split(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(split, sort_keys=False), encoding="utf-8")
    print(json.dumps({task: split["tasks"][task]["counts"] for task in TASKS}, indent=2))


if __name__ == "__main__":
    main()
