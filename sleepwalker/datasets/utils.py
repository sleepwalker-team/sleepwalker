from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xmltodict as xtd
from sklearn.model_selection import KFold

from sleepwalker.utils import logger

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

def get_edf_files_in_repo(root: str, recursive: bool = True, ending: str = "edf") -> List[str]:
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
                if f.lower().endswith(f".{ending}"):
                    edfs.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith(f".{ending}"):
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