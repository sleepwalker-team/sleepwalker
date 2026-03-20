from __future__ import annotations

from collections import Counter
from functools import partial
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xmltodict as xtd
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Sampler

from sklearn.model_selection import KFold

from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.utils import logger

import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from sleepwalker.utils import logger


class RepeatSampler(Sampler[int]):
    def __init__(self, sampler: Sampler[int], n_repeat: int = 1):
        if n_repeat <= 0:
            raise ValueError("n_repeat must be positive.")
        self.sampler = sampler
        self.n_repeat = n_repeat

    def __iter__(self):
        for idx in self.sampler:
            for _ in range(self.n_repeat):
                yield idx

    def __len__(self):
        return len(self.sampler) * self.n_repeat


def summarize_dataset(
    dataset_clazz,
    edf_files,
    channel_name=None,
    batch_size=128,
    estimate_class_frequencies=False,
):
    """Summarize a dataset: durations, channel coverage, and optionally class frequencies.

    Parameters
    ----------
    dataset_clazz : type
        Dataset class implementing `get_event_df()` and optionally `get_extra_event_df()`.
    edf_files : list[str]
        List of EDF file paths.
    channel_name : str, optional
        Channel to load for event statistics. If None, only metadata & signals are summarized.
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
                "dataset_classes": list[str] | None,
                "class_distribution": dict[str, int] | None,
                "extra_class_distribution": dict[str, int] | None,
            }
        }
    """
    meta_data = []
    signals = []
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
            signals.extend(meta["signals"])
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
        dff = pd.DataFrame([Counter(signals)]).transpose()
        dff.columns = ["count"]
        dff["coverage[%]"] = dff["count"] / files_read * 100.0
        signal_coverage = dff.sort_values("coverage[%]", ascending=False)
    else:
        signal_coverage = pd.DataFrame(columns=["count", "coverage[%]"])

    # If no channel provided → return only metadata + coverage
    if channel_name is None:
        return {
            "n_patients":dataset.n_patients,
            "duration_stats": duration_stats,
            "duration_histogram": duration_hist,
            "signal_coverage": signal_coverage,
            "dataset_classes": None,
            "class_distribution": None,
            "extra_class_distribution": None,
        }

    # --- (3) Dataset setup ---
    dataset = dataset_clazz(
        patients=edf_files,
        channels=[ChannelConfig(name=channel_name, normalizer=None)],
        sample_frequency=100,
        event_mapping={},
        remove_unmapped_events=False,
        num_workers=8
    )

    # always include known class list
    dataset_classes = getattr(dataset, "classes", None)

    # if class frequency estimation is disabled, stop here
    if not estimate_class_frequencies:
        return {
            "n_patients":dataset.n_patients,
            "duration_stats": duration_stats,
            "duration_histogram": duration_hist,
            "signal_coverage": signal_coverage,
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
    )

    label_counter = Counter()
    extra_label_counter = Counter()

    logger.progress_start(len(loader)*batch_size, desc=f"Estimating class frequencies", leave=True)
    for batch in loader:
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
        "n_patients": dataset.n_patients,
        "duration_stats": duration_stats,
        "duration_histogram": duration_hist,
        "signal_coverage": signal_coverage,
        "dataset_classes": dataset_classes,
        "class_distribution": dict(label_counter),
        "extra_class_distribution": (
            dict(extra_label_counter) if extra_label_counter else None
        ),
    }


def read_profusion(xml_path):
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
    train = [p for p in all_patients if _matches_any(p, train_patterns) and not _matches_any(p, exclude_patterns)]
    test = [p for p in all_patients if _matches_any(p, test_patterns) and not _matches_any(p, exclude_patterns)]
    return train, test

def random_split(
    all_patients: Sequence[str],
    test_frac: float = 0.3,
    seed: Optional[int] = None,
    include_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    pats = [p for p in all_patients if _matches_any(p, include_patterns)]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pats))
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(pats)))
    test = [pats[i] for i in idx[:n_test]]
    train = [pats[i] for i in idx[n_test:]]
    return train, test

def kfold_split(all_patients: list[str],n_splits: int = 5) -> List[Tuple[str,str]]:
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

def estimate_class_cnts(dataset, n_samples:Optional[int] = None, num_workers:int = 8, batch_size:int = 128):
    sampler = RandomSampler(dataset, num_samples = n_samples) if n_samples is not None else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=num_workers, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True) 
    
    total_batches = len(loader)
    class_cnts = torch.zeros(len(dataset.get_classes())) 
    logger.progress_start(total_batches*batch_size, desc=f"Estimating class counts", leave=True)
    for batch in loader:
        y = batch["target"]
        target = y.argmax(dim=1)
        idx, cnt = torch.unique(target, return_counts=True)
        class_cnts[idx] += cnt
        logger.progress_advance(batch_size)
    logger.progress_close()

    return {
        cname:c.item() for cname,c in zip(dataset.get_classes(), class_cnts)
    }

def dataset_to_numpy(dataset, n_samples:Optional[int] = None, num_workers:int = 8, batch_size:int = 128):
    sampler = RandomSampler(dataset, num_samples = n_samples) if n_samples is not None else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=num_workers, collate_fn=partial(batch_collate, ignore_list=["time", "patient"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True) 
    
    X = []
    Y = []
    Y_extra = []
    timestamps = []
    patients = []

    total_batches = len(loader)
    logger.progress_start(total_batches*batch_size, desc=f"Converting dataset to numpy", leave=True)
    for batch in loader:
        X.append(batch["data"].cpu().numpy())
        if batch["data"].shape[0] < batch_size:
            logger.warning("Incomplete batch found")

        if "target" in batch: Y.append(batch["target"].cpu().numpy())
        if "target_extra" in batch: Y_extra.append(batch["target_extra"].cpu().numpy())
        timestamps.extend(batch["time"])
        patients.extend(batch["patient"])

        logger.progress_advance(batch_size)
    logger.progress_close()

    X = np.vstack(X)
    if Y: Y = np.vstack(Y)
    if Y_extra: Y_extra = np.vstack(Y_extra)

    return X, Y, Y_extra, timestamps, patients
