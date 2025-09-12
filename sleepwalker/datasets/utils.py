from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from sleepwalker.utils import logger

def get_edf_files_in_repo(root: str, recursive: bool = True) -> List[str]:
    """List EDF files in a folder (optionally including subfolders).

    - root: base directory containing EDF files
    - recursive: if True, traverse subdirectories
    Returns a list of absolute filepaths ending with ".edf".
    """
    edfs: List[str] = []
    if recursive:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".edf"):
                    edfs.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith(".edf"):
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
) -> Tuple[List[List[str]], List[List[str]]]:
    train = [p for p in all_patients if _matches_any(p, train_patterns) and not _matches_any(p, exclude_patterns)]
    test = [p for p in all_patients if _matches_any(p, test_patterns) and not _matches_any(p, exclude_patterns)]
    return [train], [test]


def random_split(
    all_patients: Sequence[str],
    test_frac: float = 0.3,
    seed: Optional[int] = None,
    include_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    pats = [p for p in all_patients if _matches_any(p, include_patterns)]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pats))
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(pats)))
    test = [pats[i] for i in idx[:n_test]]
    train = [pats[i] for i in idx[n_test:]]
    return [train], [test]


def kfold_split(
    all_patients: Sequence[str],
    n_splits: int = 5,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    pats = [p for p in all_patients if _matches_any(p, include_patterns) and not _matches_any(p, exclude_patterns)]
    folds = np.array_split(np.array(pats, dtype=object), n_splits)
    test_folds = [f.tolist() for f in folds]
    train_folds = [[p for p in pats if p not in test_folds[i]] for i in range(n_splits)]
    return train_folds, test_folds


def publish_split(all_patients: Sequence[str], include_patterns: Optional[Sequence[str]] = None) -> Tuple[List[List[str]], List[List[str]]]:
    pats = [p for p in all_patients if _matches_any(p, include_patterns)]
    return [pats], [[]]