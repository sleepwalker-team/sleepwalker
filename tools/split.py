#!/usr/bin/env python3
"""Build a deterministic EDF holdout or cross-validation manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import random
import sys
from typing import Mapping, Sequence

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleepwalker.training.files import edf_filter_result
from sleepwalker.utils import logger
from tools.train import apply_patient_filter


def subject_id(path: str | Path) -> str:
    path = Path(path)
    return next((part for part in path.parts if part.startswith("sub-")), path.stem)


def find_edf_files(root: str | Path) -> list[str]:
    return sorted(str(path) for path in Path(root).resolve().rglob("*") if path.is_file() and path.suffix.lower() == ".edf")


def filter_edf_files(
    patients: Sequence[str | Path],
    *,
    channels: Mapping[str, Sequence[str]],
    min_duration: str | None = None,
    num_workers: int = 1,
) -> list[str]:
    """Keep EDF files containing every configured channel group."""
    if not isinstance(channels, Mapping) or not channels:
        raise ValueError("channels must be a non-empty mapping of logical names to physical alternatives.")
    channel_groups = []
    for logical_name, physical_names in channels.items():
        if not isinstance(logical_name, str) or not logical_name:
            raise ValueError("Every channel group must have a non-empty logical name.")
        if not isinstance(physical_names, Sequence) or isinstance(physical_names, str) or not physical_names:
            raise ValueError(f"Channel group '{logical_name}' must contain physical alternatives.")
        channel_groups.append(tuple(str(name) for name in physical_names))

    worker_count = int(num_workers)
    if worker_count < 1:
        raise ValueError("num_workers must be at least 1.")
    min_duration_s = None if min_duration is None else pd.to_timedelta(min_duration).total_seconds()
    patient_list = [str(path) for path in patients]
    work = [(path, tuple(channel_groups), min_duration_s, ()) for path in patient_list]
    if worker_count > 1 and work:
        with multiprocessing.Pool(worker_count) as pool:
            results = list(pool.imap(edf_filter_result, work))
    else:
        results = [edf_filter_result(item) for item in work]

    reasons = Counter(reason for _, reason in results)
    selected = [path for path, reason in results if reason == "usable"]
    excluded = {reason: count for reason, count in sorted(reasons.items()) if reason != "usable"}
    logger.info(f"Split EDF filter kept {len(selected)}/{len(patient_list)} files; excluded by reason: {excluded or '{}'}")
    return selected


def read_filter_config(path: str | Path) -> str | dict | list:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or set(config) != {"patient_filter"}:
        raise ValueError(f"Filter config {path} must contain only patient_filter.")
    if not isinstance(config["patient_filter"], (str, dict, list)) or not config["patient_filter"]:
        raise ValueError(f"Filter config {path} must define a non-empty patient_filter.")
    return config["patient_filter"]


def grouped_files(files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in files:
        groups.setdefault(subject_id(path), []).append(path)
    return {subject: sorted(paths) for subject, paths in groups.items()}


def ordered_subjects(groups: dict[str, list[str]], seed: str) -> list[str]:
    subjects = sorted(groups)
    random.Random(seed).shuffle(subjects)
    return subjects


def flatten_subjects(subjects: list[str], groups: dict[str, list[str]]) -> list[str]:
    return sorted(path for subject in subjects for path in groups[subject])


def build_holdout(files: list[str], fractions: tuple[float, float, float], seed: str) -> dict:
    train_fraction, validation_fraction, test_fraction = fractions
    if min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test fractions must be non-negative and sum to 1.")
    groups = grouped_files(files)
    subjects = ordered_subjects(groups, seed)
    train_end = round(len(subjects) * train_fraction)
    validation_end = train_end + round(len(subjects) * validation_fraction)
    split = {
        "train": flatten_subjects(subjects[:train_end], groups),
        "validation": flatten_subjects(subjects[train_end:validation_end], groups),
        "test": flatten_subjects(subjects[validation_end:], groups),
    }
    return {"seed": seed, "folds": {"holdout": split}}


def build_cross_validation(files: list[str], n_folds: int, seed: str) -> dict:
    if n_folds < 3:
        raise ValueError("Cross-validation requires at least three folds so train, validation, and test are disjoint.")
    groups = grouped_files(files)
    subjects = ordered_subjects(groups, seed)
    if len(subjects) < n_folds:
        raise ValueError(f"Cannot create {n_folds} folds from {len(subjects)} subjects.")
    buckets = [subjects[index::n_folds] for index in range(n_folds)]
    folds = {}
    for fold_index in range(n_folds):
        validation_index = (fold_index + 1) % n_folds
        train_subjects = [subject for index, bucket in enumerate(buckets) if index not in {fold_index, validation_index} for subject in bucket]
        folds[f"fold_{fold_index}"] = {
            "train": flatten_subjects(train_subjects, groups),
            "validation": flatten_subjects(buckets[validation_index], groups),
            "test": flatten_subjects(buckets[fold_index], groups),
        }
    return {"seed": seed, "folds": folds}


def manifest_summary(manifest: dict) -> dict:
    return {fold: {partition: len(files) for partition, files in split.items()} for fold, split in manifest["folds"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory traversed recursively for EDF files.")
    parser.add_argument("output_file", type=Path, help="YAML manifest to create.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fractions", nargs=3, type=float, metavar=("TRAIN", "VALIDATION", "TEST"), help="Holdout fractions; defaults to 0.8 0.1 0.1.")
    mode.add_argument("--folds", type=int, help="Number of cross-validation folds.")
    parser.add_argument("--seed", default="sleepwalker-split-v1", help="Deterministic split seed.")
    parser.add_argument("--filter-config", type=Path, help="YAML file containing a patient_filter pipeline applied before splitting.")
    parser.add_argument("--workers", type=int, default=1, help="Workers passed to patient filters.")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    files = find_edf_files(args.input_dir)
    if not files:
        raise ValueError(f"No EDF files found below {args.input_dir}.")
    if args.filter_config is not None:
        files = apply_patient_filter(read_filter_config(args.filter_config), files, dataset=None, num_workers=args.workers)
        if not files:
            raise ValueError(f"Patient filters from {args.filter_config} removed every EDF file.")
    if args.folds is not None:
        manifest = build_cross_validation(files, args.folds, args.seed)
    else:
        fractions = tuple(args.fractions or (0.8, 0.1, 0.1))
        manifest = build_holdout(files, fractions, args.seed)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(json.dumps(manifest_summary(manifest), indent=2))


if __name__ == "__main__":
    main()
