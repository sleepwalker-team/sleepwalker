from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepwalker.trainer.utils.filtering import filter_patients_by_sleep_time
from sleepwalker.trainer.utils.splits import combine_datasets, load_split


class DummyDataset:
    def __init__(self, stats_df):
        self.stats_df = stats_df

    def get_patient_stats(self, patients, summarize_fn, num_workers):
        return self.stats_df[self.stats_df["patient"].isin(patients)].reset_index(drop=True)


def test_filter_patients_by_sleep_time_drops_low_and_high_outliers():
    dataset = DummyDataset(
        pd.DataFrame(
            {
                "patient": ["a", "b", "c", "d", "e"],
                "sleep_seconds": [1.0, 10.0, 12.0, 14.0, 100.0],
            }
        )
    )

    kept = filter_patients_by_sleep_time(
        patients=["a", "b", "c", "d", "e"],
        dataset=dataset,
        sleep_labels=["n1", "n2", "n3", "rem"],
        quantile=0.2,
        num_workers=1,
        label="dummy",
    )

    assert kept == ["b", "c", "d"]


def test_combine_datasets_returns_single_dataset_unchanged():
    dataset = object()

    assert combine_datasets([dataset]) is dataset


def test_load_split_reads_yaml(tmp_path):
    path = tmp_path / "hsp_split.yml"
    path.write_text(
        "folds:\n  holdout:\n    train: [a.edf]\n    validation: [b.edf]\n    test: [c.edf]\n",
        encoding="utf-8",
    )

    split = load_split(path)

    assert split == {
        "train": ["a.edf"],
        "validation": ["b.edf"],
        "test": ["c.edf"],
    }
