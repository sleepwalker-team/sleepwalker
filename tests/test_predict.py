from collections import OrderedDict

import pandas as pd

from predict import prediction_frame_to_probability_records, probability_records_to_onehot


class DummyMulticlassTrainer:
    def __init__(self):
        self.classes = ["clean", "noisy"]


class DummyMultiLabelTrainer:
    def __init__(self):
        self.task_config = OrderedDict(
            {
                "arousal": {
                    "task": "arousal",
                    "labels": ["no_arousal", "arousal"],
                    "n_steps": 1,
                    "target_resolution": pd.to_timedelta("10s"),
                }
            }
        )


def test_multiclass_prediction_frame_converts_to_onehot():
    frame = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:00:05")],
            "prob__clean": [0.8, 0.3],
            "prob__noisy": [0.2, 0.7],
        }
    )

    records = prediction_frame_to_probability_records(frame, model_key="noisy_model", trainer=DummyMulticlassTrainer())
    onehot = probability_records_to_onehot(records)

    assert list(onehot.columns) == ["noisy_model__clean", "noisy_model__noisy"]
    assert int(onehot.loc[pd.Timestamp("2024-01-01 00:00:00"), "noisy_model__clean"]) == 1
    assert int(onehot.loc[pd.Timestamp("2024-01-01 00:00:00"), "noisy_model__noisy"]) == 0
    assert int(onehot.loc[pd.Timestamp("2024-01-01 00:00:05"), "noisy_model__clean"]) == 0
    assert int(onehot.loc[pd.Timestamp("2024-01-01 00:00:05"), "noisy_model__noisy"]) == 1


def test_multilabel_prediction_frame_averages_duplicate_timestamps_before_onehot():
    frame = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-01-01 00:00:05"), pd.Timestamp("2024-01-01 00:00:05")],
            "arousal__time": [pd.Timestamp("2024-01-01 00:00:05"), pd.Timestamp("2024-01-01 00:00:05")],
            "arousal__prob__no_arousal": [0.9, 0.2],
            "arousal__prob__arousal": [0.1, 0.8],
        }
    )

    records = prediction_frame_to_probability_records(frame, model_key="arousal_model", trainer=DummyMultiLabelTrainer())
    onehot = probability_records_to_onehot(records)

    assert list(onehot.columns) == ["arousal_model__arousal__arousal", "arousal_model__arousal__no_arousal"]
    assert int(onehot.loc[pd.Timestamp("2024-01-01 00:00:05"), "arousal_model__arousal__no_arousal"]) == 1
    assert int(onehot.loc[pd.Timestamp("2024-01-01 00:00:05"), "arousal_model__arousal__arousal"]) == 0
