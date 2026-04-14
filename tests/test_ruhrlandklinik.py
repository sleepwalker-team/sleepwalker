import pandas as pd

from sleepwalker.datasets import Ruhrlandklinik


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
