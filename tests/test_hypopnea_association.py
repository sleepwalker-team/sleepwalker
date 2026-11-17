import numpy as np
import pandas as pd
import pytest

from sleepwalker.models.DRIVEN import associate_hypopneas
from sleepwalker.models.DRIVEN import prepare_events


ORIGIN = pd.Timestamp("2025-01-01")


def events(rows):
    return pd.DataFrame({"Label": [row[0] for row in rows], "Starttime": [ORIGIN + pd.Timedelta(seconds=row[1]) for row in rows], "Endtime": [ORIGIN + pd.Timedelta(seconds=row[2]) for row in rows], "desaturation_drop_pp": [row[3] for row in rows], "SourceLabel": [f"source-{index}" for index in range(len(rows))]})


@pytest.mark.parametrize("label,delay,drop,expected", [("desaturation", 55, 3, "associated"), ("desaturation", 55.001, 3, "unassociated"), ("desaturation", 10, 2.9, "unassociated"), ("arousal", 15, np.nan, "associated"), ("arousal", 15.001, np.nan, "unassociated"), ("desaturation", 10, np.nan, "unknown")])
def test_reference_association_bounds_and_unknown_measurements(label, delay, drop, expected):
    source = events([("hypopnea", 0, 10, np.nan), (label, delay, delay + 2, drop)])
    result = associate_hypopneas(source)
    assert result.Label.iloc[0] == f"hypopnea_{expected}"
    pd.testing.assert_frame_equal(result.drop(columns="Label"), source.drop(columns="Label"))
    assert source.Label.iloc[0] == "hypopnea"


def test_latest_pending_hypopnea_is_consumed_once_and_arousal_resolves_unknown():
    source = events([("hypopnea", 0, 10, np.nan), ("hypopnea", 5, 15, np.nan), ("desaturation", 16, 18, np.nan), ("arousal", 19, 21, np.nan), ("desaturation", 25, 30, 4)])
    assert associate_hypopneas(source).Label.tolist()[:2] == ["hypopnea_unassociated", "hypopnea_associated"]


def test_equal_onsets_keep_source_order_and_prior_evidence_is_not_reused():
    source = events([("desaturation", 0, 2, 4), ("hypopnea", 0, 10, np.nan)])
    assert associate_hypopneas(source).Label.iloc[1] == "hypopnea_unassociated"
    source = source.iloc[::-1].reset_index(drop=True)
    assert associate_hypopneas(source).Label.iloc[0] == "hypopnea_associated"


def test_wrapper_processes_annotators_independently_and_missing_column_policy():
    primary = events([("hypopnea", 0, 10, np.nan), ("desaturation", 12, 14, 4)])
    alternate = events([("hypopnea", 0, 10, np.nan), ("desaturation", 12, 14, np.nan)])
    mapping = {"hypopnea_associated": "event", "hypopnea_unknown": "unknown"}
    first, second = prepare_sleep_events(primary, alternate, mapping=mapping)
    assert first.Label.tolist() == ["event"] and second.Label.tolist() == ["unknown"]
    assert prepare_events(primary, None, mapping=mapping)[1] is None
    missing = alternate.drop(columns="desaturation_drop_pp")
    assert prepare_events(primary, missing, mapping=mapping, on_missing_field="reject") is None
    with pytest.raises(ValueError, match="desaturation_drop_pp"):
        prepare_events(primary, missing, mapping=mapping, on_missing_field="error")
