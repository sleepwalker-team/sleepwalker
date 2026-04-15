import pandas as pd

from compare_predictions_ruhrland import (
    _exact_match_count,
    _merge_intervals,
    _sweep_overlap,
    compare_prediction_df,
)


def test_merge_and_exact_match_count():
    intervals = [
        (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:00:10")),
        (pd.Timestamp("2024-01-01 00:00:08"), pd.Timestamp("2024-01-01 00:00:12")),
    ]
    assert _merge_intervals(intervals) == [
        (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:00:12"))
    ]
    assert _exact_match_count(intervals, intervals[:1]) == 1


def test_sweep_overlap_handles_longer_covering_interval():
    pred_by_label = {
        "a": [(pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:00:20"))]
    }
    gt_by_label = {
        "a": [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:00:10")),
            (pd.Timestamp("2024-01-01 00:00:11"), pd.Timestamp("2024-01-01 00:00:21")),
        ]
    }
    assert _sweep_overlap(pred_by_label, gt_by_label) == {("a", "a"): 19.0}


def test_compare_prediction_df_uses_inferred_raw_label_mapping():
    prediction_df = pd.DataFrame(
        {"desat_model__desaturation": [1, 1]},
        index=[
            pd.Timestamp("2024-01-01 00:00:05"),
            pd.Timestamp("2024-01-01 00:00:15"),
        ],
    )
    event_df = pd.DataFrame(
        {
            "Label": ["entsättigung", "entsättigung"],
            "Starttime": [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:00:11"),
            ],
            "Endtime": [
                pd.Timestamp("2024-01-01 00:00:10"),
                pd.Timestamp("2024-01-01 00:00:21"),
            ],
        }
    )
    task_entries = [
        {
            "group": "desat_model",
            "task": "desat_model",
            "labels": ["desaturation", "no desaturation"],
            "raw_labels": {"desaturation": ["entsättigung"]},
            "window": pd.to_timedelta("10s"),
        }
    ]
    class_df, task_df, matrix_df = compare_prediction_df(prediction_df, event_df, task_entries, patient="p1.edf")

    assert int(class_df.iloc[0]["exact_matches"]) == 1
    assert float(class_df.iloc[0]["overlap_seconds"]) == 19.0
    assert float(task_df.iloc[0]["overlap_seconds"]) == 19.0
    assert float(matrix_df.iloc[0]["overlap_seconds"]) == 19.0
