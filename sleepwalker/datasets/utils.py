"""Dataset discovery, splitting, XML parsing, and cache-export helpers.

This module contains a mix of reusable utilities used across dataset adapters,
trainers, tests, and training scripts. The more stable pieces, based on current
call sites and tests, are EDF discovery, dataset splitting, repeat sampling,
and numpy-cache export helpers.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import partial
from pathlib import Path
import os
import multiprocessing
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xmltodict as xtd
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from sleepwalker.datasets.Basedataset import BaseDataset, ChannelConfig, batch_collate
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.utils import logger


def summarize_dataset(
    dataset_clazz,
    edf_files,
    batch_size=128,
    estimate_class_frequencies=False,
):
    """Summarize a dataset: durations, signal inventory, and optionally class frequencies.

    Parameters
    ----------
    dataset_clazz : type
        Dataset class implementing `get_event_df()` and optionally `get_extra_event_df()`.
    edf_files : list[str]
        List of EDF file paths.
    batch_size : int, default=128
        Batch size for iteration when estimating class frequencies.
    estimate_class_frequencies : bool, default=False
        Whether to iterate over the dataset and compute event class distributions.

    Returns
    -------
    dict
        {
            "meta": DataFrame with file-level info,
            "summary": {
                "duration_stats": {...},
                "duration_histogram": {...},
                "signal_coverage": DataFrame,
                "signal_summary": DataFrame,
                "dataset_classes": list[str] | None,
                "class_distribution": dict[str, int] | None,
                "extra_class_distribution": dict[str, int] | None,
            }
        }
    """
    meta_data = []
    signal_counter = Counter()
    signal_sample_rates: dict[str, set[float]] = {}
    first_channel_name = None
    files_read = 0

    logger.info(f"Found a total of {len(edf_files)} EDF files")

    for e in edf_files:
        try:
            meta = read_edf_meta(e)
            meta_data.append({
                "file": e,
                "start": meta["start"],
                "end": meta["end"],
                "duration[s]": meta["duration_s"],
            })
            for signal in meta["signals"]:
                if first_channel_name is None:
                    first_channel_name = signal
                signal_counter[signal] += 1
                signal_sample_rates.setdefault(signal, set()).add(float(meta["fs"][signal]))
            files_read += 1
        except Exception as ex:
            logger.warning(f"Skipping {e}: {ex}")
            continue

    logger.info(f"Successfully read {files_read} files")
    df_meta = pd.DataFrame(meta_data)

    # --- (1) Duration statistics & histogram ---
    duration_s = df_meta["duration[s]"].dropna()

    def _format_hms(seconds: float) -> str:
        return str(pd.to_timedelta(seconds, unit="s"))

    if len(duration_s) > 0:
        duration_stats = {
            "min": _format_hms(duration_s.min()),
            "max": _format_hms(duration_s.max()),
            "mean": _format_hms(duration_s.mean()),
            "median": _format_hms(duration_s.median()),
            "q25": _format_hms(duration_s.quantile(0.25)),
            "q75": _format_hms(duration_s.quantile(0.75)),
        }
        bins = np.linspace(duration_s.min(), duration_s.max(), num=11)
        hist, edges = np.histogram(duration_s, bins=bins)
        duration_hist = {
            f"[{_format_hms(edges[i])} - {_format_hms(edges[i+1])})": int(hist[i])
            for i in range(len(hist))
        }
    else:
        duration_stats, duration_hist = {}, {}

    # --- (2) Channel coverage ---
    if files_read > 0:
        dff = pd.DataFrame([signal_counter]).transpose()
        dff.columns = ["count"]
        dff["coverage[%]"] = dff["count"] / files_read * 100.0
        signal_coverage = dff.sort_values("coverage[%]", ascending=False)

        signal_summary = signal_coverage.copy()
        signal_summary["sample_rate"] = [
            sample_rates[0] if len(sample_rates) == 1 else sample_rates
            for sample_rates in (
                sorted(signal_sample_rates[signal]) for signal in signal_summary.index
            )
        ]
    else:
        signal_coverage = pd.DataFrame(columns=["count", "coverage[%]"])
        signal_summary = pd.DataFrame(columns=["count", "coverage[%]", "sample_rate"])

    if first_channel_name is None:
        raise ValueError("Could not determine a channel name from the provided EDF files.")

    # --- (3) Dataset setup ---
    dataset = dataset_clazz(
        channels=[
            ChannelConfig(
                logical_name=first_channel_name,
                physical_names=[first_channel_name],
            )
        ],
        sample_frequency=100,
        event_mapping={},
        remove_unmapped_events=False,
    )
    dataset.initialize(edf_files, 8)

    # always include known class list
    dataset_classes = getattr(dataset, "classes", None)

    # if class frequency estimation is disabled, stop here
    if not estimate_class_frequencies:
        return {
            "duration_stats": duration_stats,
            "duration_histogram": duration_hist,
            "signal_coverage": signal_coverage,
            "signal_summary": signal_summary,
            "dataset_classes": dataset_classes,
            "class_distribution": None,
            "extra_class_distribution": None,
        }

    # --- (4) Iterate and compute class frequencies ---
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: batch_collate(
            x, ignore_list=["time", "patient", "target", "target_extra"]
        ),
        num_workers=8
    )

    label_counter = Counter()
    extra_label_counter = Counter()

    logger.progress_start(len(loader)*batch_size, desc=f"Estimating class frequencies", leave=True)
    for batch in loader:
        if batch is None:
            continue
        try:
            edf_path = batch["patient"][0]
            start_dt = pd.Timestamp(batch["time"][0])
            df = dataset.get_event_df(edf_path, start_dt)
            label_counter.update(df["Label"].astype(str).str.lower())

            if dataset.has_extra_target():
                extra_df = dataset.get_extra_event_df(edf_path, start_dt)
                extra_label_counter.update(extra_df["Label"].astype(str).str.lower())
            logger.progress_advance(batch_size)

        except Exception as ex:
            logger.warning(f"Failed to process {edf_path}: {ex}")
            continue

    logger.info("Finished summarizing dataset")

    return {
        "duration_stats": duration_stats,
        "duration_histogram": duration_hist,
        "signal_coverage": signal_coverage,
        "signal_summary": signal_summary,
        "dataset_classes": dataset_classes,
        "class_distribution": dict(label_counter),
        "extra_class_distribution": (
            dict(extra_label_counter) if extra_label_counter else None
        ),
    }


def read_profusion(xml_path):
    """Read Profusion XML annotations into a normalized event table.

    Notes:
        The mapping logic is dataset-specific and currently tailored to the XML
        structures used by repository dataset adapters such as ``ABC`` and
        ``SHHS``. The function returns a concatenated event table but does not
        yet document a stronger schema guarantee beyond observed call sites.
    """
    with open(xml_path, "r") as f:
        doc = xtd.parse(f.read())
        apnea_df = pd.DataFrame(doc["CMPStudyConfig"]["ScoredEvents"]["ScoredEvent"][1:])
        apnea_df["Name"] = apnea_df.apply(lambda row: row["Name"].lower().strip(), axis=1)
        apnea_map = {
            'arousal ()': "arousal|arousal ()",
            'arousal (aro res)': "arousal resulting from respiratory effort|arousal (aro res)",
            'arousal (asda)': "asda arousal|arousal (asda)",
            'arousal (cheshire)': "arousal resulting from chin emg|arousal (cheshire)",
            'arousal (external arousal)': "external arousal|arousal (external arousal)",
            'arousal (standard)': "arousal|arousal (standard)",
            'central apnea': "central apnea|central apnea",
            'hypopnea': "hypopnea|hypopnea",
            'mixed apnea': "mixed apnea|mixed apnea",
            'obstructive apnea': "obstructive apnea|obstructive apnea",
            'periodic breathing': "periodic breathing|periodic breathing",
            'respiratory artifact': "respiratory artifact|respiratory artifact",
            'spo2 artifact': "spo2 artifact|spo2 artifact",
            'spo2 desaturation': "spo2 desaturation|spo2 desaturation",
            'unsure': "unsure|unsure"
        }
        apnea_df["Label"] = apnea_df.apply(lambda row: apnea_map.get(row["Name"], None), axis=1)
        
        # Profusion mainly exports snoring events (see above), but also contains sleep stages.
        # These sleep stages are numbered 0-6/9 internally. We match them back to the nsrr annotations here
        # to make them comparable.
        sleep_df = pd.DataFrame(doc["CMPStudyConfig"]["SleepStages"])
        sleep_df["Start"] = sleep_df.index*30
        sleep_df["Duration"] = 30
        sleep_map = {
            "0": "wake|0",
            "1": "stage 1 sleep|1",
            "2": "stage 2 sleep|2",
            "3": "stage 3 sleep|3",
            "4": "stage 4 sleep|4",
            "5": "rem sleep|5",
            "6": "movement|6",
            "9": "unscored|9"
        }
        sleep_df["Label"] = sleep_df.apply(lambda row: sleep_map.get(row["SleepStage"], None), axis=1)

        return pd.concat([apnea_df, sleep_df], ignore_index=True)

def read_nsrr(xml_path):
    """Read NSRR-style XML annotations into an event table."""
    with open(xml_path, "r") as f:
        doc = xtd.parse(f.read())
        """
        [
            'arousal resulting from chin emg|arousal (cheshire)', 
            'arousal resulting from respiratory effort|arousal (aro res)', 
            'arousal|arousal ()', 
            'arousal|arousal (standard)', 
            'asda arousal|arousal (asda)', 
            'central apnea|central apnea', 
            'external arousal|arousal (external arousal)', 
            'hypopnea|hypopnea', 
            'mixed apnea|mixed apnea', 
            'movement|6', 
            'obstructive apnea|obstructive apnea', 
            'periodic breathing|periodic breathing', 
            'rem sleep|5', 
            'respiratory artifact|respiratory artifact', 
            'spo2 artifact|spo2 artifact', 
            'spo2 desaturation|spo2 desaturation', 
            'stage 1 sleep|1', 
            'stage 2 sleep|2', 
            'stage 3 sleep|3', 
            'stage 4 sleep|4', 
            'unscored|9', 
            'unsure|unsure', 
            'wake|0'
        ]
        """
        xml_df = pd.DataFrame(doc["PSGAnnotation"]["ScoredEvents"]["ScoredEvent"][1:])
        xml_df["Label"] = xml_df.apply(lambda row: row["EventConcept"].lower().strip(),axis=1)

        return xml_df

def get_edf_files_in_repo(root: str|os.PathLike, recursive: bool = True, ending: str = ".edf") -> List[str]:
    """List EDF files in a folder (optionally including subfolders).

    - root: base directory containing EDF files
    - recursive: if True, traverse subdirectories
    - ending: Ending of the file. Typically .edf, but legacy use .rec  
    Returns a list of absolute filepaths ending with ".edf" (or .ending).
    """
    edfs: List[str] = []
    if recursive:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(f"{ending}"):
                    edfs.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            if f.endswith(f"{ending}"):
                edfs.append(os.path.join(root, f))
    
    if len(edfs) == 0:
        logger.warning(f"No EDF files found in {root}. Is this expected?")

    return edfs

def _matches_any(path: str, patterns: Optional[Sequence[str]]) -> bool:
    if patterns is None:
        return True
    if len(patterns) == 0:
        return False
    return any(p in path for p in patterns)

def fixed_split(
    all_patients: Sequence[str],
    train_patterns: Optional[Sequence[str]] = None,
    test_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split patients by substring matching against path patterns."""
    train = [p for p in all_patients if _matches_any(p, train_patterns) and not _matches_any(p, exclude_patterns)]
    test = [p for p in all_patients if _matches_any(p, test_patterns) and not _matches_any(p, exclude_patterns)]
    return train, test

def random_split(
    all_patients: Sequence[str],
    test_frac: float = 0.3,
    seed: Optional[int] = None,
    include_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Randomly split patient paths into train and test partitions."""
    pats = [p for p in all_patients if _matches_any(p, include_patterns)]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pats))
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(pats)))
    test = [pats[i] for i in idx[:n_test]]
    train = [pats[i] for i in idx[n_test:]]
    return train, test

def kfold_split(all_patients: list[str],n_splits: int = 5) -> List[Tuple[str,str]]:
    """Create K-fold train/test patient splits."""
    kf = KFold(n_splits)
    
    folds = []
    for train_idx, test_idx in kf.split(all_patients):
        train_patients = [all_patients[idx] for idx in train_idx]
        test_patients = [all_patients[idx] for idx in test_idx]
        folds.append( (train_patients, test_patients) )
    return folds

# def publish_split(all_patients: Sequence[str], include_patterns: Optional[Sequence[str]] = None) -> Tuple[List[List[str]], List[List[str]]]:
#     pats = [p for p in all_patients if _matches_any(p, include_patterns)]
#     return [pats], [[]]

def estimate_class_cnts(loader: DataLoader):
    """Estimate multiclass target counts from a dataloader.

    Args:
        loader: Dataloader yielding batches with one-hot ``target`` tensors.

    Returns:
        A mapping from class name to observed count over one realized pass.
    """
    dataset = loader.dataset
    total_batches = len(loader)
    batch_size = loader.batch_size or 1
    class_cnts = torch.zeros(len(dataset.get_classes()))
    logger.progress_start(total_batches * batch_size, desc="Estimating class counts", leave=True)
    for batch in loader:
        if batch is None:
            continue
        y = batch["target"]
        if "target_mask" in batch:
            mask = batch["target_mask"].to(dtype=torch.bool)
            if tuple(mask.shape) != tuple(y.shape[:-1]):
                raise ValueError(f"Expected target_mask shaped {tuple(y.shape[:-1])}, got {tuple(mask.shape)}.")
            observed = y[mask]
        else:
            observed = y.reshape(-1, y.shape[-1])
        class_cnts += observed.sum(dim=0).cpu()
        logger.progress_advance(len(y))
    logger.progress_close()

    return {
        cname:c.item() for cname,c in zip(dataset.get_classes(), class_cnts)
    }

def dataloader_to_numpy(loader: DataLoader):
    X = []
    Y = []
    Y_extra = []
    timestamps = []
    patients = []

    total_batches = len(loader)
    batch_size = loader.batch_size or 1
    logger.progress_start(total_batches * batch_size, desc="Converting dataloader to numpy", leave=True)
    for batch in loader:
        if batch is None:
            continue
        X.append(batch["data"].cpu().numpy())
        if batch["data"].shape[0] < batch_size:
            logger.warning("Incomplete batch found")

        if "target" in batch: Y.append(batch["target"].cpu().numpy())
        if "target_extra" in batch: Y_extra.append(batch["target_extra"].cpu().numpy())
        timestamps.extend(batch["time"])
        patients.extend(batch["patient"])
        logger.progress_advance(batch["data"].shape[0])
    logger.progress_close()

    X = np.vstack(X)
    if Y: Y = np.vstack(Y)
    if Y_extra: Y_extra = np.vstack(Y_extra)

    return X, Y, Y_extra, timestamps, patients


def value_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        return value.to_numpy()
    if isinstance(value, pd.Timestamp):
        return np.int64(value.value)
    return np.asarray(value)

def stack_numpy_values(values, key: str):
    arrays = [value_to_numpy(v) for v in values]
    first = arrays[0]
    if np.asarray(first).ndim == 0:
        return np.asarray([np.asarray(v).item() if np.asarray(v).ndim == 0 else v for v in arrays])
    try:
        return np.stack(arrays)
    except ValueError as exc:
        raise ValueError(f"Cannot stack exported values for key '{key}'. Ensure the field has a stable shape across items.") from exc


def export_batch_collate(batch, extra_keys: Optional[Sequence[str]] = None):
    """Collate a batch for cache export without losing selected metadata.

    Args:
        batch: Sequence of dataset items.
        extra_keys: Optional metadata keys that must stay as lists rather than
            being stacked.

    Returns:
        A collated batch suitable for ``export_dataloader_to_numpy_dir``.
    """
    items = [item for item in batch if item is not None]
    keep_stacked = {"data", "target", "target_extra"}
    keep_ignored = {"time", "patient", *(extra_keys or [])}
    ignore_list = list(keep_ignored)

    if items:
        for key in items[0].keys():
            if key not in keep_stacked and key not in keep_ignored:
                ignore_list.append(key)

    return batch_collate(items, ignore_list=ignore_list)


def save_array_chunks(out_path: Path, stem: str, array: np.ndarray, n_samples_per_file: Optional[int]) -> None:
    if n_samples_per_file is None or int(array.shape[0]) <= int(n_samples_per_file):
        np.save(out_path / f"{stem}.npy", array, allow_pickle=False)
        return

    n_samples_per_file = int(n_samples_per_file)
    if n_samples_per_file <= 0:
        raise ValueError("n_samples_per_file must be positive when provided.")

    for chunk_idx, start in enumerate(range(0, int(array.shape[0]), n_samples_per_file)):
        stop = min(start + n_samples_per_file, int(array.shape[0]))
        np.save(out_path / f"{stem}.{chunk_idx:06d}.npy", array[start:stop], allow_pickle=False)


def export_dataloader_to_numpy_dir(
    loader: DataLoader,
    out_dir: str | os.PathLike,
    *,
    extra_keys: Optional[Sequence[str]] = None,
    n_samples_per_file: Optional[int] = None,
) -> Path:
    """Export one realized dataloader pass into a numpy cache directory.

    Args:
        loader: Dataloader producing already-collated dataset items.
        out_dir: Output directory for the cache.
        extra_keys: Optional per-item keys to export in addition to the
            standard fields.
        n_samples_per_file: Optional chunk size used to split arrays across
            multiple ``.npy`` files.

    Returns:
        The output directory path.

    Raises:
        ValueError: If required keys are missing, the dataset is not
            initialized, or exported extra keys are ragged or unsupported.

    Notes:
        The export intentionally captures exactly one realized loader pass,
        including any dataset-side randomization. Tests in
        ``tests/test_datasets.py`` confirm round-tripping, memmap loading, and
        chunked exports.
    """
    dataset = loader.dataset
    if hasattr(dataset, 'initialized') and not dataset.initialized:
        raise ValueError(f"{dataset.__class__.__name__} is not initialized. Call initialize(...) before exporting.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X = []
    Y = []
    Y_extra = []
    timestamps = []
    patients = []
    extras = {key: [] for key in (extra_keys or [])}

    expected = len(loader) * (loader.batch_size or 1)
    logger.progress_start(expected, desc='Exporting dataset to numpy cache', leave=True)
    for batch in loader:
        if batch is None:
            continue
        if 'data' not in batch or 'patient' not in batch or 'time' not in batch:
            raise ValueError('Export requires item keys: data, patient, time')

        X.append(batch['data'].cpu().numpy().astype(np.float32, copy=False))
        if 'target' in batch:
            Y.append(batch['target'].cpu().numpy())
        if 'target_extra' in batch:
            Y_extra.append(batch['target_extra'].cpu().numpy())

        timestamps.append(np.asarray([np.int64(pd.Timestamp(t).value) for t in batch['time']], dtype=np.int64))
        patients.append(np.asarray([str(patient) for patient in batch['patient']]))
 
        batch_size_actual = len(batch['patient'])
        for key in extras:
            if key not in batch:
                raise ValueError(f"Requested extra key '{key}' missing from exported batch.")
            if len(batch[key]) != batch_size_actual:
                raise ValueError(f"Requested extra key '{key}' has inconsistent batch length during export.")
            extras[key].append(stack_numpy_values(batch[key], key))

        logger.progress_advance(batch_size_actual)
    logger.progress_close()

    if len(X) == 0:
        raise ValueError('Export produced no valid items.')

    logger.info(f"Writing files to {out_path}")
    data_arr = np.concatenate(X, axis=0)
    save_array_chunks(out_path, 'data', data_arr, n_samples_per_file)

    if Y:
        save_array_chunks(out_path, 'target', np.concatenate(Y, axis=0), n_samples_per_file)
    if Y_extra:
        save_array_chunks(out_path, 'target_extra', np.concatenate(Y_extra, axis=0), n_samples_per_file)

    save_array_chunks(out_path, 'time', np.concatenate(timestamps, axis=0), n_samples_per_file)
    save_array_chunks(out_path, 'patient', np.concatenate(patients, axis=0), n_samples_per_file)

    for key, values in extras.items():
        save_array_chunks(out_path, f'extra__{key}', np.concatenate(values, axis=0), n_samples_per_file)

    input_channels = dataset.get_input_channels() if hasattr(dataset, 'get_input_channels') else [str(i) for i in range(data_arr.shape[-1])]
    classes = dataset.get_classes() if hasattr(dataset, 'get_classes') else []
    fallback_patients = [str(patient) for patient in np.concatenate(patients, axis=0)]
    all_patients = sorted(set(fallback_patients))

    meta = {
        'sample_frequency': float(getattr(dataset, 'sample_frequency')),
        'resample_type': str(getattr(dataset, 'resample_type', 'cached')),
        'total_input': str(getattr(dataset, 'total_input')),
        'classes': list(classes),
        'input_channels': list(input_channels),
        'all_patients': [str(p) for p in all_patients],
        'extra_keys': list(extras.keys()),
        'n_items': int(data_arr.shape[0]),
        'stride': str(getattr(dataset, 'stride')),
    }
    with (out_path / 'meta.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return out_path
