import pandas as pd

from sleepwalker.datasets import Ruhrlandklinik
from sleepwalker.datasets.Ruhrlandklinik import get_channels


def make_dataset() -> Ruhrlandklinik:
    return Ruhrlandklinik(
        channels=[],
        sample_frequency=1,
        event_mapping={},
        remove_unmapped_events=False,
    )


def test_expand_change_based_body_positions_starts_at_first_clear_position_by_default():
    dataset = make_dataset()
    df = pd.DataFrame(
        [
            {"Label": "links", "Starttime": pd.Timestamp("2020-01-01 00:10:00"), "Endtime": pd.NaT, "Duration": pd.NA},
            {"Label": "rechts", "Starttime": pd.Timestamp("2020-01-01 00:20:00"), "Endtime": pd.NaT, "Duration": pd.NA},
            {"Label": "n2", "Starttime": pd.Timestamp("2020-01-01 00:00:00"), "Endtime": pd.Timestamp("2020-01-01 00:30:00"), "Duration": 1800.0},
        ]
    )

    out = dataset._expand_change_based_body_positions(df)
    positions = out[out["Label"].isin(dataset.BODY_POSITION_EVENTS)].reset_index(drop=True)

    assert positions["Label"].tolist() == ["links", "rechts"]
    assert positions["Starttime"].tolist() == [
        pd.Timestamp("2020-01-01 00:10:00"),
        pd.Timestamp("2020-01-01 00:20:00"),
    ]
    assert positions["Endtime"].tolist() == [
        pd.Timestamp("2020-01-01 00:20:00"),
        pd.Timestamp("2020-01-01 00:30:00"),
    ]


def test_expand_change_based_body_positions_can_seed_initial_position():
    dataset = make_dataset()
    dataset.initial_body_position_event = "aufrecht"
    df = pd.DataFrame(
        [
            {"Label": "links", "Starttime": pd.Timestamp("2020-01-01 00:10:00"), "Endtime": pd.NaT, "Duration": pd.NA},
            {"Label": "rechts", "Starttime": pd.Timestamp("2020-01-01 00:20:00"), "Endtime": pd.NaT, "Duration": pd.NA},
            {"Label": "n2", "Starttime": pd.Timestamp("2020-01-01 00:00:00"), "Endtime": pd.Timestamp("2020-01-01 00:30:00"), "Duration": 1800.0},
        ]
    )

    out = dataset._expand_change_based_body_positions(df)
    positions = out[out["Label"].isin(dataset.BODY_POSITION_EVENTS)].reset_index(drop=True)

    assert positions["Label"].tolist() == ["aufrecht", "links", "rechts"]
    assert positions["Starttime"].tolist() == [
        pd.Timestamp("2020-01-01 00:00:00"),
        pd.Timestamp("2020-01-01 00:10:00"),
        pd.Timestamp("2020-01-01 00:20:00"),
    ]
    assert positions["Endtime"].tolist() == [
        pd.Timestamp("2020-01-01 00:10:00"),
        pd.Timestamp("2020-01-01 00:20:00"),
        pd.Timestamp("2020-01-01 00:30:00"),
    ]


def test_expand_change_based_body_positions_ignores_duplicate_changes():
    dataset = make_dataset()
    df = pd.DataFrame(
        [
            {"Label": "links", "Starttime": pd.Timestamp("2020-01-01 00:10:00"), "Endtime": pd.NaT, "Duration": pd.NA},
            {"Label": "links", "Starttime": pd.Timestamp("2020-01-01 00:15:00"), "Endtime": pd.NaT, "Duration": pd.NA},
            {"Label": "rechts", "Starttime": pd.Timestamp("2020-01-01 00:20:00"), "Endtime": pd.NaT, "Duration": pd.NA},
            {"Label": "n2", "Starttime": pd.Timestamp("2020-01-01 00:00:00"), "Endtime": pd.Timestamp("2020-01-01 00:30:00"), "Duration": 1800.0},
        ]
    )

    out = dataset._expand_change_based_body_positions(df)
    positions = out[out["Label"].isin(dataset.BODY_POSITION_EVENTS)].reset_index(drop=True)

    assert positions["Label"].tolist() == ["links", "rechts"]
    assert positions["Starttime"].tolist() == [
        pd.Timestamp("2020-01-01 00:10:00"),
        pd.Timestamp("2020-01-01 00:20:00"),
    ]
    assert positions["Endtime"].tolist() == [
        pd.Timestamp("2020-01-01 00:20:00"),
        pd.Timestamp("2020-01-01 00:30:00"),
    ]


def test_get_channels_grouped_keeps_candidates_but_exposes_grouped_inputs():
    channels = get_channels(
        ["eeg", "chin_emg"],
        grouped=True,
        include_quality=False,
        normalize=False,
        sample_frequency=100,
    )

    assert len(channels) == 2
    assert channels[0].logical_name == "EEG"
    assert channels[0].physical_names == ["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

    dataset = Ruhrlandklinik(
        channels=channels,
        sample_frequency=100,
        total_input="30s",
        event_mapping={},
        remove_unmapped_events=False,
    )

    assert dataset.get_input_channels() == ["EEG", "Chin EMG"]
