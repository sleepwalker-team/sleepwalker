"""Contract, annotator isolation, and exact retained interval regression tests."""

import numpy as np
import pandas as pd
import pytest

from sleepwalker.config import build_callback
from sleepwalker.datasets.Basedataset import EventIndex
from sleepwalker.datasets.events import exclude_events_on_wake, merge_event_intervals, validate_event_labels, validate_event_table
from sleepwalker.training.callbacks import map_events, prepare_sleep_events


def event_table(rows, *, timezone=None):
    origin = pd.Timestamp("2024-01-01", tz=timezone)
    return pd.DataFrame({
        "Label": pd.Series([row[0] for row in rows], dtype="object"),
        "Starttime": pd.DatetimeIndex([origin + pd.Timedelta(seconds=row[1]) for row in rows], tz=timezone),
        "Endtime": pd.DatetimeIndex([origin + pd.Timedelta(seconds=row[2]) for row in rows], tz=timezone),
    })


def test_map_events_preserves_metadata_and_annotator_availability():
    primary = event_table([("n2", 0, 30), ("desaturation", 10, 20), ("arousal", 5, 7)])
    primary["desaturation_drop_pp"] = [np.nan, 4.2, np.nan]
    extra = event_table([("n2", 0, 30), ("arousal", 15, 18)])
    extra["SourceLabel"] = ["stage 2", "arousal"]
    before = primary.copy(deep=True)
    mapped, mapped_extra = map_events(primary, extra, mapping={"n2": "sleep", "arousal": "arousal"})
    assert mapped["Label"].tolist() == ["sleep", "arousal"]
    assert len(mapped) == 2  # Missing optional fields must not drop these rows.
    assert mapped_extra["Label"].tolist() == ["sleep", "arousal"]
    assert mapped_extra["SourceLabel"].tolist() == ["stage 2", "arousal"]
    pd.testing.assert_frame_equal(primary, before)
    retained, absent = map_events(primary, None, mapping={"n2": "sleep"}, remove_unmapped=False)
    assert retained["Label"].tolist() == ["sleep", "desaturation", "arousal"]
    assert retained.loc[1, "desaturation_drop_pp"] == 4.2
    assert absent is None


def test_mapping_empty_result_preserves_core_schema():
    original = event_table([("n2", 0, 30)])
    mapped, extra = map_events(original, original, mapping={})
    assert mapped.empty and extra.empty
    validate_event_table(mapped)
    validate_event_table(extra)


@pytest.mark.parametrize("labels", [[], ["a", "a"], [""], ["  "], [None]])
def test_invalid_declared_vocabularies(labels):
    with pytest.raises(ValueError):
        validate_event_labels(labels)


def test_declared_labels_are_ordered_and_extensible():
    labels = ["rem", "custom_reader_event", "n2"]
    assert validate_event_labels(labels) == labels
    validate_event_table(event_table([("custom_reader_event", 0, 1)]), allowed_labels=labels)
    with pytest.raises(ValueError, match="recording.edf.*undeclared.*custom_reader_event"):
        validate_event_table(event_table([("custom_reader_event", 0, 1)]), allowed_labels=["n2"], context="recording.edf alternate")
    with pytest.raises(TypeError, match="ordered sequence"):
        validate_event_labels("n2")


@pytest.mark.parametrize("mutation, message", [
    (lambda df: df.drop(columns="Endtime"), "missing required"),
    (lambda df: df.assign(Label=None), "non-empty strings"),
    (lambda df: df.assign(Starttime="2024-01-01"), "pandas timestamps"),
    (lambda df: df.assign(Endtime=pd.NaT), "non-missing"),
    (lambda df: df.assign(Endtime=df.Starttime), "later than"),
    (lambda df: df.assign(Endtime=df.Endtime.dt.tz_localize("UTC")), "same timezone"),
])
def test_invalid_core_tables_fail_before_transform(mutation, message):
    with pytest.raises(ValueError, match=message):
        map_events(mutation(event_table([("n2", 0, 30)])), None, mapping={"n2": "sleep"})


def test_merge_unsorted_nested_intervals_and_conflicting_measurements():
    events = event_table([("desaturation", 10, 20), ("arousal", 4, 5), ("desaturation", 0, 30), ("desaturation", 30, 40)])
    events["desaturation_drop_pp"] = [3.0, np.nan, 4.0, 5.0]
    merged = merge_event_intervals(events, labels=["desaturation"])
    assert len(merged) == 3
    assert merged["Label"].tolist() == ["desaturation", "arousal", "desaturation"]
    assert merged.iloc[0].Starttime == events.iloc[2].Starttime
    assert merged.iloc[0].Endtime == events.iloc[2].Endtime
    assert pd.isna(merged.iloc[0].desaturation_drop_pp)
    assert merged.iloc[2].desaturation_drop_pp == 5.0
    touching = merge_event_intervals(events, labels=["desaturation"], touching_counts_as_overlap=True)
    assert len(touching) == 2
    assert touching.iloc[0].Endtime == events.iloc[3].Endtime
    assert len(events) == 4


def test_merge_preserves_agreed_metadata_and_timezone():
    events = event_table([("limb_movement", 0, 10), ("limb_movement", 5, 15)], timezone="Europe/Berlin")
    events["SourceLabel"] = "lm"
    merged = merge_event_intervals(events, labels=["limb_movement"])
    assert merged.SourceLabel.tolist() == ["lm"]
    assert merged.Starttime.dt.tz == events.Starttime.dt.tz
    validate_event_table(merged)


@pytest.mark.parametrize("touching, kept", [(False, [1, 3, 4]), (True, [4])])
def test_wake_exclusion_endpoint_policy(touching, kept):
    events = event_table([("wake", 10, 20), ("arousal", 0, 10), ("arousal", 15, 25), ("arousal", 20, 30), ("arousal", 30, 40)])
    filtered = exclude_events_on_wake(events, labels=["arousal"], touching_counts_as_overlap=touching)
    assert filtered.index.tolist() == [0, *kept]
    assert len(events) == 5


def test_wake_exclusion_handles_empty_and_nested_wake_sets():
    events = event_table([("arousal", 15, 20)])
    pd.testing.assert_frame_equal(exclude_events_on_wake(events, labels=["arousal"]), events)
    nested = event_table([("wake", 0, 100), ("wake", 1, 2), ("arousal", 15, 20)])
    assert exclude_events_on_wake(nested, labels=["arousal"]).Label.tolist() == ["wake", "wake"]
    empty = event_table([])
    pd.testing.assert_frame_equal(merge_event_intervals(empty, labels=["arousal"]), empty)


def test_wake_exclusion_accepts_different_timestamp_resolutions():
    events = event_table([("wake", 10, 20), ("arousal", 15, 25), ("arousal", 30, 35)])
    events["Starttime"] = events.Starttime.dt.as_unit("us")
    events["Endtime"] = events.Endtime.dt.as_unit("s")
    assert exclude_events_on_wake(events, labels=["arousal"]).index.tolist() == [0, 2]


@pytest.mark.parametrize("processor", [merge_event_intervals, exclude_events_on_wake])
def test_endpoint_configuration_rejects_non_boolean_flags(processor):
    with pytest.raises(TypeError, match="must be a boolean"):
        processor(event_table([]), labels=[], touching_counts_as_overlap="false")


def test_composed_callback_uses_each_annotators_own_wake_and_mapping():
    primary = event_table([("n2", 0, 30), ("wake", 10, 20), ("limb_movement", 8, 12), ("limb_movement", 11, 14)])
    alternate = event_table([("n2", 0, 30), ("limb_movement", 8, 12), ("limb_movement", 11, 14)])
    callback = build_callback({
        "name": "sleepwalker.training.callbacks.prepare_sleep_events",
        "merge_labels": ["limb_movement"], "exclude_on_wake": ["limb_movement"],
        "mapping": {"n2": "sleep", "limb_movement": "limb_movement"},
        "touching_counts_as_overlap": True,
    })
    prepared, prepared_extra = callback(primary, alternate)
    assert prepared.Label.tolist() == ["sleep"]
    assert prepared_extra.Label.tolist() == ["sleep", "limb_movement"]
    assert prepared_extra.iloc[1].Endtime == alternate.iloc[2].Endtime
    _, no_extra = prepare_sleep_events(primary, None, mapping={"n2": "sleep"})
    assert no_extra is None
    mapper = build_callback({"name": "sleepwalker.datasets.utils.map_events", "mapping": {"n2": "sleep"}})
    assert mapper(primary, None)[0].Label.tolist() == ["sleep"]


@pytest.mark.parametrize("timezone", [None, "Europe/Berlin"])
def test_event_index_nested_events_half_open_boundaries_and_retained_dataframe(timezone):
    events = event_table([("arousal", 0, 10), ("arousal", 1, 2), ("arousal", 10, 12), ("sleep", 0, 20)], timezone=timezone)
    events["SourceLabel"] = ["A", "B", "C", "D"]
    events.index = [4, 9, 3, 2]
    expected = events.copy()
    index = EventIndex(events)
    origin = events.iloc[0].Starttime
    timeline = index.query(origin, origin + pd.Timedelta(seconds=21), freq="1s", labels=["arousal", "sleep", "unobserved"])
    assert timeline.arousal.tolist() == [1] * 12 + [0] * 9
    assert timeline.sleep.tolist() == [1] * 20 + [0]
    assert not timeline.unobserved.any()
    pd.testing.assert_frame_equal(index.events, expected)
    events.loc[4, "Endtime"] = origin + pd.Timedelta(seconds=100)
    pd.testing.assert_frame_equal(index.events, expected)
    pd.testing.assert_frame_equal(index.query(origin, origin + pd.Timedelta(seconds=21), freq="1s", labels=["arousal", "sleep", "unobserved"]), timeline)


def test_event_index_empty_and_non_nanosecond_clock():
    empty = EventIndex(event_table([])).events
    assert empty.empty
    validate_event_table(empty)
    events = event_table([("n2", 0, 10)])
    events["Starttime"] = events.Starttime.dt.as_unit("us")
    events["Endtime"] = events.Endtime.dt.as_unit("us")
    index = EventIndex(events)
    origin = events.iloc[0].Starttime
    assert index.query(origin, origin + pd.Timedelta(seconds=11), "1s", labels=["n2"]).n2.tolist() == [1] * 10 + [0]
    pd.testing.assert_frame_equal(index.events, events)
