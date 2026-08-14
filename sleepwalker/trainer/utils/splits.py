"""Dataset and fold-manifest helpers shared by training and evaluation."""

from pathlib import Path
from typing import Any

import yaml

from sleepwalker.datasets.MultiDataset import combine_datasets


PARTITIONS = ("train", "validation", "test")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load one file manifest."""
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a top-level mapping in manifest {path}.")
    return manifest


def fold_names(path: str | Path) -> list[str]:
    manifest = load_manifest(path)
    if "files" in manifest:
        return []
    if not isinstance(manifest.get("folds"), dict) or not manifest["folds"]:
        raise ValueError(f"Manifest {path} has no folds.")
    return [str(name) for name in manifest["folds"]]


def load_split(path: str | Path, fold: str | None = None) -> dict[str, list[str]]:
    """Load one train/validation/test fold from a manifest."""
    manifest = load_manifest(path)
    if not isinstance(manifest.get("folds"), dict) or not manifest["folds"]:
        raise ValueError(f"Manifest {path} does not define folds.")
    folds = manifest["folds"]
    if fold is None:
        if len(folds) != 1:
            raise ValueError(f"Manifest {path} contains folds {list(folds)}; select one explicitly.")
        fold = str(next(iter(folds)))
    if fold not in folds:
        raise ValueError(f"Unknown fold '{fold}' in {path}; available folds are {list(folds)}.")
    split = folds[fold]
    if not isinstance(split, dict):
        raise ValueError(f"Fold '{fold}' in {path} must be a mapping.")
    missing = [partition for partition in PARTITIONS if partition not in split]
    if missing:
        raise ValueError(f"Fold '{fold}' in {path} is missing partitions {missing}.")
    for partition in PARTITIONS:
        if not isinstance(split[partition], list):
            raise ValueError(f"Fold '{fold}' partition '{partition}' in {path} must be a list.")
    resolved = {partition: [str(item) for item in split[partition]] for partition in PARTITIONS}
    for partition, files in resolved.items():
        if len(files) != len(set(files)):
            raise ValueError(f"Fold '{fold}' partition '{partition}' contains duplicate files.")
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1:]:
            overlap = set(resolved[left]).intersection(resolved[right])
            if overlap:
                raise ValueError(f"Fold '{fold}' partitions '{left}' and '{right}' overlap: {sorted(overlap)[:5]}.")
    return resolved


def load_files(path: str | Path, fold: str | None = None, partition: str | None = None) -> list[str]:
    """Resolve either an unsplit file manifest or one fold partition."""
    manifest = load_manifest(path)
    if "files" in manifest:
        if fold is not None or partition not in (None, "test"):
            raise ValueError(f"Unsplit manifest {path} does not accept a fold or non-test partition.")
        if not isinstance(manifest["files"], list):
            raise ValueError(f"Unsplit manifest {path} must contain a files list.")
        files = [str(item) for item in manifest["files"]]
    else:
        if partition is None:
            raise ValueError(f"Fold manifest {path} requires a partition.")
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition '{partition}'.")
        files = load_split(path, fold)[partition]
    if not files:
        raise ValueError(f"Manifest selection from {path} is empty.")
    if len(files) != len(set(files)):
        raise ValueError(f"Manifest selection from {path} contains duplicate files.")
    return files


__all__ = ["combine_datasets", "fold_names", "load_files", "load_manifest", "load_split"]
