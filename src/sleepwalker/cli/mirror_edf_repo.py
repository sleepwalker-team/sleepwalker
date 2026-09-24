#!/usr/bin/env python3
"""Mirror an EDF repository with selected, rate-limited signal channels.

The channel list and default target rate come from a normal Sleepwalker
training YAML. Normalizers, filters, and dataset callbacks are ignored.
Non-EDF files are copied unchanged and the directory structure is preserved.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import time
import uuid
import warnings
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyedflib
import yaml

from sleepwalker.core.signal import edf_to_df
from sleepwalker.datasets.Basedataset import ChannelConfig, unit_conversion_factor
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.utils import logger


def edf_physical_bound(value: float, lower: bool) -> int | float:
    if not np.isfinite(value):
        raise ValueError(f"EDF physical bounds must be finite, got {value}")
    decimal_value = Decimal(str(value))
    rounding = ROUND_FLOOR if lower else ROUND_CEILING
    for decimal_places in range(15, -1, -1):
        quantized = decimal_value.quantize(Decimal(1).scaleb(-decimal_places), rounding=rounding)
        candidate = int(quantized) if quantized == quantized.to_integral_value() else float(quantized)
        if len(str(candidate)) <= 8 and (candidate <= value if lower else candidate >= value):
            return candidate
    raise ValueError(f"Cannot represent physical bound {value} in the eight-character EDF header field")


def read_edf(source: Path, channels: Mapping[str, str | None], unit_overrides: Mapping[str, str], max_sample_rate: float, exact_sampling: bool, resample_method: str, convert_units: bool, assume_units_if_missing: bool) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any], int]:
    requested = set(channels)
    with pyedflib.EdfReader(str(source)) as reader:
        labels = list(reader.getSignalLabels())
        selected = [(index, label) for index, label in enumerate(labels) if label in requested]
        if not selected:
            raise ValueError(f"{source}: none of the configured channels are present")
        if len({label for _, label in selected}) != len(selected):
            raise ValueError(f"{source}: selected EDF channel labels are not unique")

        all_headers = reader.getSignalHeaders()
        signal_headers = [dict(all_headers[index]) for index, _ in selected]
        file_header = dict(reader.getHeader())
        annotations = zip(*reader.readAnnotations())
        file_header["annotations"] = [[float(onset), float(duration), str(description)] for onset, duration, description in annotations]
        file_type = int(reader.filetype)

        signals: dict[str, np.ndarray] = {}
        groups: dict[float, list[str]] = {}
        output_rates: dict[str, float] = {}
        for index, label in selected:
            native_rate = float(all_headers[index]["sample_frequency"])
            output_rate = max_sample_rate if exact_sampling else min(native_rate, max_sample_rate)
            output_rates[label] = output_rate
            if np.isclose(native_rate, output_rate):
                signals[label] = reader.readSignal(index, digital=False)
            else:
                groups.setdefault(output_rate, []).append(label)

        for output_rate, group in groups.items():
            frame = edf_to_df(reader, group, start=None, end=None, frequency=output_rate, how=resample_method)
            if any(label not in frame for label in group):
                raise ValueError(f"failed to resample channels {group}")
            for label in group:
                signals[label] = frame[label].to_numpy(dtype=np.float64)

        output_signals: list[np.ndarray] = []
        for header, (_, label) in zip(signal_headers, selected):
            header["sample_frequency"] = output_rates[label]
            signal = np.asarray(signals[label], dtype=np.float64)

            target_unit = channels[label]
            if convert_units and target_unit is not None:
                factor = unit_conversion_factor(unit_overrides.get(label, str(header.get("dimension", ""))), target_unit, assume_if_missing=assume_units_if_missing)
                # Unit definitions are decimal; discard binary division noise such as 1000.0000000000001 before storing EDF header bounds.
                factor = float(f"{factor:.12g}")
                signal = signal * factor
                header["physical_min"] = float(header["physical_min"]) * factor
                header["physical_max"] = float(header["physical_max"]) * factor
                header["dimension"] = target_unit
            output_signals.append(signal)

    return output_signals, signal_headers, file_header, file_type


def process_edf(file_paths: tuple[str, str], read_selected_edf: partial, overwrite: bool) -> tuple[str, int, int, str, str | None]:
    source_name, target_name = file_paths
    source, target = Path(source_name), Path(target_name)
    source_bytes = 0
    temporary: Path | None = None
    status = "failed"
    target_bytes = 0
    error_message: str | None = None
    try:
        source_bytes = source.stat().st_size
        if os.path.lexists(target) and not overwrite:
            return "skipped", source_bytes, target.stat().st_size, source_name, None
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        signals, signal_headers, file_header, file_type = read_selected_edf(source)
        if len(signals) != len(signal_headers):
            raise ValueError(f"Read {len(signals)} signals but {len(signal_headers)} signal headers")
        for signal, header in zip(signals, signal_headers):
            physical_min = min(float(header["physical_min"]), float(np.min(signal)))
            physical_max = max(float(header["physical_max"]), float(np.max(signal)))
            header["physical_min"] = edf_physical_bound(physical_min, lower=True)
            header["physical_max"] = edf_physical_bound(physical_max, lower=False)
            if header["physical_min"] == header["physical_max"]:
                raise ValueError(f"Physical minimum and maximum are equal for channel {header['label']}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"phys_(?:min|max) is .*?, but signal_(?:min|max) is .*? for channel .*", category=UserWarning)
            pyedflib.highlevel.write_edf(str(temporary), signals, signal_headers, header=file_header, file_type=file_type)
        with pyedflib.EdfReader(str(temporary)) as reader:
            expected = [str(header["label"]) for header in signal_headers]
            if list(reader.getSignalLabels()) != expected:
                raise ValueError("output EDF channel verification failed")
        shutil.copystat(source, temporary)
        os.replace(temporary, target)
        status = "processed"
        target_bytes = target.stat().st_size
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
    try:
        if temporary is not None and os.path.lexists(temporary):
            temporary.unlink()
    except Exception as error:
        status = "failed"
        target_bytes = 0
        cleanup_error = f"temporary-file cleanup failed: {type(error).__name__}: {error}"
        error_message = f"{error_message}; {cleanup_error}" if error_message is not None else cleanup_error
    return status, source_bytes, target_bytes, source_name, error_message


def format_size(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024 or unit == "PiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} PiB"


def mirror_repository(source: str | Path, destination: str | Path, channels: Sequence[ChannelConfig], max_sample_rate: float, *, exact_sampling: bool = False, resample_method: str = "nearest", convert_units: bool = False, unit_overrides: Mapping[str, str] | None = None, assume_units_if_missing: bool = False, overwrite: bool = False, workers: int = 4) -> dict[str, int]:
    """Mirror a repository with Pool.imap_unordered and progress logging."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if max_sample_rate <= 0:
        raise ValueError("max_sample_rate must be positive")
    if resample_method not in {"nearest", "mean", "max"}:
        raise ValueError(f"Unsupported resample method: {resample_method}")

    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve(strict=False)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must not overlap")

    channels_to_copy: dict[str, str | None] = {}
    for channel in channels:
        for physical_name in channel.physical_names:
            if physical_name not in channels_to_copy:
                channels_to_copy[physical_name] = channel.unit
            if channel.unit is not None:
                if channels_to_copy[physical_name] not in {None, channel.unit}:
                    raise ValueError(f"Conflicting target units for {physical_name}")
                channels_to_copy[physical_name] = channel.unit
            quality_name = channel.quality_name_for(physical_name)
            if quality_name is not None:
                channels_to_copy.setdefault(quality_name, None)
    if not channels_to_copy:
        raise ValueError("At least one channel must be configured")

    logger.info(f"Scanning {source} folder")
    edf_sources = {Path(path) for path in get_edf_files_in_repo(source, recursive=True)}
    logger.info(f"Found {len(edf_sources)} EDF files. Scanning the directory tree and copying other files.")
    edf_files: list[tuple[str, str]] = []
    summary = {
        "processed": 0,
        "copied": 0,
        "skipped": 0,
        "failed": 0,
        "source_bytes": 0,
        "target_bytes": 0,
    }
    scanned_files = 0
    last_scan_log = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path in edf_sources:
            scanned_files += 1
            edf_files.append((str(path), str(target)))
        elif path.is_file():
            scanned_files += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            summary["copied"] += 1
            summary["source_bytes"] += path.stat().st_size
            summary["target_bytes"] += target.stat().st_size
        if time.monotonic() - last_scan_log >= 30:
            logger.info(f"Repository scan: {scanned_files} files found, {len(edf_files)} EDF files queued, {summary['copied']} other files copied.")
            last_scan_log = time.monotonic()

    logger.info(f"Discovered {len(edf_files)} EDF files and {summary['copied']} other files.")
    read_selected_edf = partial(read_edf, channels=channels_to_copy, unit_overrides=dict(unit_overrides or {}), max_sample_rate=max_sample_rate, exact_sampling=exact_sampling, resample_method=resample_method, convert_units=convert_units, assume_units_if_missing=assume_units_if_missing)
    process = partial(process_edf, read_selected_edf=read_selected_edf, overwrite=overwrite)
    logger.progress_start(len(edf_files), desc="Converting EDF files", leave=True)
    try:
        if edf_files:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                results = pool.imap_unordered(process, edf_files, chunksize=1)
                for status, source_bytes, target_bytes, source_name, error in results:
                    summary[status] += 1
                    summary["source_bytes"] += source_bytes
                    summary["target_bytes"] += target_bytes
                    if error is not None:
                        logger.warning(f"Skipping EDF {source_name}: {error}")
                    logger.progress_advance()
    finally:
        logger.progress_close()

    logger.info(f"Mirror complete: {summary['processed']} EDF processed, {summary['copied']} other files copied, {summary['skipped']} existing EDF files skipped, {summary['failed']} EDF files failed; {format_size(summary['source_bytes'])} -> {format_size(summary['target_bytes'])}.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("config", type=Path, help="Sleepwalker training YAML.")
    parser.add_argument("--max-sample-rate", type=float, default=None)
    parser.add_argument("--exact-sampling", action="store_true", help="Resample lower-rate channels up to the exact target rate too.")
    parser.add_argument("--resample-method", choices=["nearest", "mean", "max"], default="nearest")
    parser.add_argument("--convert-units", action="store_true")
    parser.add_argument("--assume-units-if-missing", action="store_true", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Expected a top-level mapping in {args.config}")
    data = config.get("data", config)
    data_entries = data if isinstance(data, list) else [data]
    if not data_entries or any(not isinstance(entry, Mapping) for entry in data_entries):
        raise ValueError("Expected data to be a mapping or list of mappings")

    channels: list[ChannelConfig] = []
    configured_rates: set[float] = set()
    configured_assumptions: set[bool] = set()
    unit_overrides: dict[str, str] = {}
    for entry in data_entries:
        channel_specs = entry.get("channels") or []
        if not isinstance(channel_specs, list) or any(not isinstance(spec, Mapping) for spec in channel_specs):
            raise ValueError("Expected data.channels to be a list of mappings")
        for spec in channel_specs:
            if "logical_name" not in spec or "physical_names" not in spec:
                raise ValueError("Every channel needs logical_name and physical_names")
            channels.append(ChannelConfig(logical_name=str(spec["logical_name"]), physical_names=spec["physical_names"], quality_name=spec.get("quality_name"), unit=spec.get("unit")))
        if entry.get("sample_frequency") is not None:
            configured_rates.add(float(entry["sample_frequency"]))
        configured_assumptions.add(bool(entry.get("assume_units_if_missing", False)))
        configured_overrides = entry.get("edf_unit_overrides") or {}
        if not isinstance(configured_overrides, Mapping):
            raise ValueError("Expected data.edf_unit_overrides to be a mapping")
        for channel, unit in configured_overrides.items():
            channel, unit = str(channel), str(unit)
            if channel in unit_overrides and unit_overrides[channel] != unit:
                raise ValueError(f"Conflicting unit overrides for {channel}")
            unit_overrides[channel] = unit

    max_sample_rate = args.max_sample_rate
    if max_sample_rate is None:
        if len(configured_rates) != 1:
            raise ValueError("Expected one data.sample_frequency or --max-sample-rate")
        max_sample_rate = next(iter(configured_rates))
    assume_units_if_missing = args.assume_units_if_missing
    if assume_units_if_missing is None:
        if len(configured_assumptions) > 1:
            raise ValueError("Conflicting assume_units_if_missing values")
        assume_units_if_missing = next(iter(configured_assumptions), False)

    mirror_repository(args.source, args.destination, channels, max_sample_rate, exact_sampling=args.exact_sampling, resample_method=args.resample_method, convert_units=args.convert_units, unit_overrides=unit_overrides, assume_units_if_missing=assume_units_if_missing, overwrite=args.overwrite, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
