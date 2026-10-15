"""Dataset-wide preparation order, declared vocabularies and inference isolation."""

from functools import partial

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from sleepwalker.datasets.Basedataset import BaseDataset, EDFFile, EventIndex
from sleepwalker.datasets.SyntheticDataset import SyntheticDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.datasets.utils import estimate_class_cnts
from sleepwalker.datasets.events import validate_event_table
from sleepwalker.training.callbacks import map_events


ORIGIN = pd.Timestamp("2024-01-01")


def annotations(rows):
    return pd.DataFrame({
        "Label": pd.Series([label for label, _, _ in rows], dtype="object"),
        "Starttime": pd.to_datetime([ORIGIN + pd.Timedelta(seconds=start) for _, start, _ in rows]),
        "Endtime": pd.to_datetime([ORIGIN + pd.Timedelta(seconds=end) for _, _, end in rows]),
    })


class AnnotationDataset(BaseDataset):
    def __init__(self, primary=None, alternate=None, supported=None, **kwargs):
        self.primary = annotations([("n2", 0, 60)]) if primary is None else primary
        self.alternate = alternate
        self.supported = ["n2", "arousal", "desaturation"] if supported is None else supported
        self.reads = []
        super().__init__(channels=[], sample_frequency=1, **kwargs)

    def supported_events(self):
        return list(self.supported)

    def get_event_df(self, path, start):
        self.reads.append("primary")
        return self.primary

    def has_extra_target(self):
        return True

    def get_extra_event_df(self, path, start):
        self.reads.append("alternate")
        return self.alternate


def prepare(dataset):
    return dataset.prepare_recording_labels("recording.edf", ORIGIN, ORIGIN + pd.Timedelta("100s"))


def test_complete_metadata_is_available_once_per_preparation_for_both_annotators():
    primary = annotations([("n2", 0, 60), ("desaturation", 20, 30)])
    primary["desaturation_drop_pp"] = [np.nan, 4.2]
    alternate = annotations([("n2", 10, 50), ("arousal", 20, 25)])
    alternate["SourceLabel"] = ["N2", "A"]
    calls = []

    def transform(label_df, label_extra_df):
        calls.append((label_df.copy(), label_extra_df.copy()))
        assert label_df.desaturation_drop_pp.iloc[1] == 4.2
        label_df.loc[:, "SourceLabel"] = "changed"
        return map_events(label_df, label_extra_df, mapping={"n2": "sleep", "desaturation": "event", "arousal": "event"})

    def trim(label_df, label_extra_df, patient):
        assert patient == "recording.edf"
        assert list(label_df) == ["Label", "Starttime", "Endtime"]
        return label_df[label_df.Label == "event"], label_extra_df

    dataset = AnnotationDataset(primary=primary, alternate=alternate, annotation_labels=["event", "sleep", "unknown"], prepare_events=transform, prepare_patient=trim)
    result = prepare(dataset)
    assert dataset.reads == ["primary", "alternate"]
    assert len(calls) == 1
    assert "SourceLabel" not in primary
    assert result["start"] == ORIGIN + pd.Timedelta("20s")
    assert result["end"] == ORIGIN + pd.Timedelta("30s")
    assert result["classes"] == {"event"}
    assert result["extra_classes"] == {"sleep", "event"}
    assert dataset.label_classes == ["event", "sleep", "unknown"]
    index = EventIndex(result["label_df"])
    pd.testing.assert_frame_equal(index.events, result["label_df"])
    timeline = index.query(result["start"], result["end"], "1s", labels=dataset.label_classes)
    assert timeline.columns.tolist() == ["event", "sleep", "unknown"]
    assert timeline.event.eq(1).all() and timeline.sleep.eq(0).all() and timeline.unknown.eq(0).all()
    prepare(dataset)
    assert len(calls) == 2


def test_identity_loading_and_absent_labels_preserve_declared_order():
    dataset = AnnotationDataset(annotation_labels=["arousal", "n2"])
    result = prepare(dataset)
    assert result["label_df"].Label.tolist() == ["n2"]
    assert result["label_extra_df"] is None
    assert dataset.label_classes == ["arousal", "n2"]
    assert dataset.get_classes() == []


@pytest.mark.parametrize("callback", [lambda **kwargs: None, lambda label_df, label_extra_df: (label_df.iloc[:0], label_extra_df)])
def test_explicit_event_rejection_and_empty_primary_are_expected_exclusions(callback):
    assert prepare(AnnotationDataset(annotation_labels=["n2"], prepare_events=callback)) is None


def test_empty_patient_result_is_an_expected_exclusion():
    dataset = AnnotationDataset(annotation_labels=["n2"], prepare_patient=lambda label_df, label_extra_df, patient: (label_df.iloc[:0], label_extra_df))
    assert prepare(dataset) is None


def test_empty_alternate_remains_available_without_fabricated_occurrences():
    dataset = AnnotationDataset(alternate=annotations([]), annotation_labels=["n2", "arousal"])
    result = prepare(dataset)
    assert result["label_extra_df"] is not None and result["label_extra_df"].empty
    validate_event_table(result["label_extra_df"])
    assert result["extra_classes"] == set()


def test_reader_capabilities_are_validated_before_mapping_can_drop_unsupported_labels():
    dataset = AnnotationDataset(primary=annotations([("invalid_raw", 0, 10)]), annotation_labels=["n2"], prepare_events=partial(map_events, mapping={"n2": "n2"}))
    with pytest.raises(ValueError, match="recording.edf primary reader output.*invalid_raw"):
        prepare(dataset)
    for declaration in (["n2", "n2"], [""], [None]):
        invalid = AnnotationDataset(supported=declaration, annotation_labels=["n2"])
        with pytest.raises(ValueError, match="supported_events"):
            prepare(invalid)
        assert invalid.reads == []


@pytest.mark.parametrize("stage", ["events", "patient"])
def test_undeclared_transformed_labels_fail_with_recording_context(stage):
    def transform(label_df, label_extra_df, **kwargs):
        return label_df.assign(Label="undeclared"), label_extra_df

    dataset = AnnotationDataset(annotation_labels=["n2"], **{f"prepare_{stage}": transform})
    with pytest.raises(ValueError, match="recording.edf primary.*undeclared"):
        prepare(dataset)


@pytest.mark.parametrize("alternate", [None, annotations([])])
def test_callbacks_must_preserve_alternate_source_availability(alternate):
    def transform(label_df, label_extra_df):
        return label_df, annotations([]) if label_extra_df is None else None

    with pytest.raises(ValueError, match="alternate annotation availability"):
        prepare(AnnotationDataset(alternate=alternate, annotation_labels=["n2"], prepare_events=transform))


def test_callback_exceptions_and_invalid_tables_are_errors():
    def invalid_schema(label_df, label_extra_df):
        return label_df.drop(columns="Endtime"), label_extra_df

    with pytest.raises(ValueError, match="recording.edf.*missing required"):
        prepare(AnnotationDataset(annotation_labels=["n2"], prepare_events=invalid_schema))

    def broken_callback(**kwargs):
        raise RuntimeError("missing desaturation_drop_pp in alternate")

    with pytest.raises(ValueError, match="recording.edf prepare_events failed.*alternate"):
        prepare(AnnotationDataset(annotation_labels=["n2"], prepare_events=broken_callback))


def test_annotation_errors_are_visible_in_initialization_and_cohort_statistics(monkeypatch):
    def read_file(path):
        return EDFFile(channels=[], path=str(path), start_date=ORIGIN, end_date=ORIGIN + pd.Timedelta("60s"))

    monkeypatch.setattr(EDFFile, "from_edf", staticmethod(read_file))
    dataset = AnnotationDataset(primary=annotations([("invalid_raw", 0, 30)]), annotation_labels=["n2"])
    with pytest.raises(ValueError, match="recording.edf primary reader output.*invalid_raw"):
        dataset.initialize(["recording.edf"], num_workers=0, strict=False)
    with pytest.raises(ValueError, match="recording.edf primary reader output.*invalid_raw"):
        dataset.get_patient_stats(["recording.edf"], summarize_patient=lambda **kwargs: {}, num_workers=0)


@pytest.mark.parametrize("kwargs", [{"annotation_labels": ["n2"]}, {"prepare_events": lambda **kwargs: None}, {"prepare_patient": lambda **kwargs: None}, {"prepare_target": lambda **kwargs: None}])
def test_unlabelled_mode_rejects_annotation_configuration(kwargs):
    with pytest.raises(ValueError, match="load_annotations=False"):
        AnnotationDataset(load_annotations=False, **kwargs)


def test_unlabelled_loading_skips_readers_and_cloning_omits_label_recipe():
    dataset = AnnotationDataset(load_annotations=False)
    result = prepare(dataset)
    assert dataset.reads == [] and result["label_df"] is None
    labelled = AnnotationDataset(annotation_labels=["n2"], prepare_events=partial(map_events, mapping={"n2": "n2"}))
    inference = UnlabelledDataset.from_dataset(labelled)
    assert inference.load_annotations is False and inference.label_classes == []
    assert inference.prepare_events_callback is None and inference.prepare_patient_callback is None and inference.prepare_target_callback is None
    assert inference.supported_events() == []
    restored = cloudpickle.loads(cloudpickle.dumps(labelled))
    assert restored.label_classes == ["n2"]
    assert prepare(restored)["label_df"].Label.tolist() == ["n2"]


def test_non_xml_initialization_never_replaces_vocabularies_with_observed_labels():
    dataset = SyntheticDataset(channels=[], sample_frequency=1, annotation_labels=["rem", "wake", "n2", "n1", "n3"])
    dataset.classes = ["regular", "event"]
    from_path = __file__.replace("test_annotation_lifecycle.py", "data/signals_01.edf")
    dataset.initialize([from_path], num_workers=0, strict=True)
    assert dataset.label_classes == ["rem", "wake", "n2", "n1", "n3"]
    assert dataset.get_classes() == ["regular", "event"]
    assert dataset.observed_labels <= set(dataset.label_classes)


def test_target_counts_use_explicit_outputs_and_masks_without_an_annotation_vocabulary():
    loader = DataLoader([
        {"target": torch.tensor([[1., 0.], [0., 1.]]), "target_mask": torch.tensor([True, False])},
        {"target": torch.tensor([[0., 1.], [0., 1.]]), "target_mask": torch.tensor([True, True])},
    ], batch_size=2)
    assert estimate_class_cnts(loader, classes=["regular", "event"]) == {"regular": 1., "event": 2.}
