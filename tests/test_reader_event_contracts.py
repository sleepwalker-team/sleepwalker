"""Source decoding, optional measurements and reader-independent preparation."""

from functools import partial

import pandas as pd
import pytest

from sleepwalker.cli.train import build_dataset
from sleepwalker.datasets.Apples import Apples
from sleepwalker.datasets.HSP import HSP
from sleepwalker.datasets.ISRUC import ISRUC
from sleepwalker.datasets.MROS import MROS
from sleepwalker.datasets.Ruhrlandklinik import Ruhrlandklinik
from sleepwalker.datasets.SHHS import SHHS
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.events import validate_event_table
from sleepwalker.datasets.utils import read_nsrr, read_profusion
from sleepwalker.training.callbacks import map_events, prepare_sleep_events


ORIGIN = pd.Timestamp("2024-01-01")


def write_nsrr(path, *, measured=True):
    drop = "<SpO2Baseline>96</SpO2Baseline><SpO2Nadir>92</SpO2Nadir>" if measured else ""
    path.write_text(f"""<PSGAnnotation><ScoredEvents>
<ScoredEvent><EventConcept>Recording Start Time</EventConcept><Start>0</Start><Duration>0</Duration></ScoredEvent>
<ScoredEvent><EventConcept>Stage 1 sleep|1</EventConcept><Start>0</Start><Duration>30</Duration></ScoredEvent>
<ScoredEvent><EventConcept>SpO2 desaturation|SpO2 desaturation</EventConcept><Start>10</Start><Duration>10</Duration>{drop}</ScoredEvent>
<ScoredEvent><EventConcept>Unknown source concept|x</EventConcept><Start>20</Start><Duration>2</Duration></ScoredEvent>
</ScoredEvents></PSGAnnotation>""")


def test_nsrr_preserves_metadata_and_standardizes_before_both_xml_readers(tmp_path):
    for reader_class in (SHHS, MROS):
        folder = tmp_path / reader_class.__name__ / "annotations-events-nsrr" / "shhs1"
        folder.mkdir(parents=True)
        path = folder / "shhs1-001-nsrr.xml"
        write_nsrr(path)
        edf_path = folder.parent.parent / "edfs" / "shhs1" / "shhs1-001.edf"
        reader = reader_class(channels=[], sample_frequency=1, load_annotations=False)
        events = reader.get_event_df(str(edf_path), ORIGIN)
        validate_event_table(events, allowed_labels=reader.supported_events())
        assert events.Label.tolist() == ["n1", "desaturation", "other"]
        assert events.loc[2, "desaturation_drop_pp"] == 4.0
        assert pd.isna(events.loc[1, "desaturation_drop_pp"])
        assert events.loc[1, "SourceLabel"] == "Stage 1 sleep|1"
        assert reader.get_extra_event_df(str(edf_path), ORIGIN) is None
        assert reader.supported_events() == reader_class(channels=[], sample_frequency=1, annotation_labels=["task_event"]).supported_events()


def test_xml_missing_measurement_column_differs_from_missing_individual_measurement(tmp_path):
    path = tmp_path / "events.xml"
    write_nsrr(path, measured=False)
    assert "desaturation_drop_pp" not in read_nsrr(path)
    write_nsrr(path, measured=True)
    measured = read_nsrr(path)
    assert "desaturation_drop_pp" in measured
    assert measured.desaturation_drop_pp.isna().sum() == 3


def test_profusion_single_event_and_stage_decoding_does_not_discard_first_event(tmp_path):
    path = tmp_path / "profusion.xml"
    path.write_text("""<CMPStudyConfig><ScoredEvents><ScoredEvent>
<Name>SpO2 desaturation</Name><Start>5</Start><Duration>10</Duration><Desaturation>4.2</Desaturation>
</ScoredEvent></ScoredEvents><SleepStages><SleepStage>1</SleepStage><SleepStage>4</SleepStage></SleepStages></CMPStudyConfig>""")
    events = read_profusion(path)
    assert events.Label.tolist() == ["desaturation", "n1", "n3"]
    assert events.desaturation_drop_pp.iloc[0] == 4.2
    assert events.desaturation_drop_pp.iloc[1:].isna().all()


def test_hsp_optional_drop_extraction_preserves_unmeasured_events(tmp_path):
    edf_path = tmp_path / "patient_task-psg_eeg.edf"
    pd.DataFrame({"event": ["Stage - N2", "Desaturation - min 91 % - drop 4.2 %", "Desaturation - min 91 %", "Respiratory Event - Hypopnea - desat 85 %"],
                  "time": ["00:00:00", "00:00:10", "00:00:20", "00:00:30"], "duration": [60, 10, 10, 10]}).to_csv(tmp_path / "patient_task-psg_annotations.csv", index=False)
    reader = HSP(channels=[], sample_frequency=1, annotation_labels=["sleep", "event"], prepare_events=partial(map_events, mapping={"n2": "sleep", "desaturation": "event", "hypopnea": "event"}))
    events = reader.get_event_df(str(edf_path), ORIGIN)
    validate_event_table(events, allowed_labels=reader.supported_events())
    assert events.Label.tolist() == ["n2", "desaturation", "desaturation", "hypopnea"]
    assert events.desaturation_drop_pp.iloc[1] == 4.2
    assert events.desaturation_drop_pp.iloc[[0, 2, 3]].isna().all()
    result = reader.prepare_recording_labels(str(edf_path), ORIGIN, ORIGIN + pd.Timedelta("60s"))
    assert result["label_df"].Label.tolist() == ["sleep", "event", "event", "event"]


def test_ruhrland_reader_leaves_task_policies_to_explicit_preparation(tmp_path, monkeypatch):
    edf = tmp_path / "recording.edf"
    edf.with_name("recording_MS.xlsx").touch()
    events = pd.DataFrame({"Ereignis": ["header", "Wach", "N2", "LM", "LM", "Arousal"],
                           "Anfangszeit": [ORIGIN] * 3 + [ORIGIN + pd.Timedelta(seconds=5), ORIGIN + pd.Timedelta(seconds=8), ORIGIN + pd.Timedelta(seconds=30)],
                           "Endzeit": [ORIGIN, ORIGIN + pd.Timedelta(seconds=10), ORIGIN + pd.Timedelta(seconds=60), ORIGIN + pd.Timedelta(seconds=9), ORIGIN + pd.Timedelta(seconds=12), ORIGIN + pd.Timedelta(seconds=32)],
                           "Dauer": [0, 10, 60, 4, 4, 2]})
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: events.copy())
    monkeypatch.setattr("sleepwalker.datasets.Ruhrlandklinik.xlrd.open_workbook", lambda *args, **kwargs: object())
    reader = Ruhrlandklinik(channels=[], sample_frequency=1, annotation_labels=["wake", "n2", "limb_movement", "arousal"])
    decoded = reader.get_event_df(str(edf), ORIGIN)
    assert decoded.Label.tolist() == ["wake", "n2", "limb_movement", "limb_movement", "arousal"]
    validate_event_table(decoded, allowed_labels=reader.supported_events())
    primary, alternate = prepare_sleep_events(decoded, decoded.iloc[[1, 2, 3, 4]].copy(), mapping={"n2": "sleep", "limb_movement": "lm", "arousal": "arousal"}, merge_labels=["limb_movement"], exclude_on_wake=["limb_movement"], touching_counts_as_overlap=True)
    assert primary.Label.tolist() == ["sleep", "arousal"]
    assert alternate.Label.tolist() == ["sleep", "lm", "arousal"]
    assert alternate[alternate.Label == "lm"].Duration.iloc[0] == 7


def test_non_xml_apples_reader_has_common_stages_and_explicit_unknowns(tmp_path):
    path = tmp_path / "recording.edf"
    pd.DataFrame({"class": ["W", "N2", "desat", "Unclassified"], "start": ["00:00:00", "00:00:30", "00:00:40", "00:00:45"], "stop": ["00:00:30", "00:01:00", "00:00:50", "00:00:46"]}).to_csv(path.with_suffix(".annot"), sep="\t", index=False)
    reader = Apples(channels=[], sample_frequency=1, annotation_labels=["wake", "n2", "desaturation", "other"])
    decoded = reader.get_event_df(str(path), ORIGIN)
    assert decoded.Label.tolist() == ["wake", "n2", "desaturation", "other"]
    assert decoded.SourceLabel.tolist() == ["W", "N2", "desat", "Unclassified"]
    validate_event_table(decoded, allowed_labels=reader.supported_events())


def test_sleep_edfx_preserves_source_onsets_including_annotation_gaps(tmp_path, monkeypatch):
    (tmp_path / "SC4001EC-Hypnogram.edf").touch()

    class AnnotationReader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def readAnnotations(self):
            return [0, 60], [30, 30], ["Sleep stage 1", "Sleep stage 4"]

    monkeypatch.setattr("sleepwalker.datasets.SleepEDFx.pyedflib.EdfReader", lambda path: AnnotationReader())
    reader = SleepEDFx(channels=[], sample_frequency=1, annotation_labels=["n1", "n3"])
    events = reader.get_event_df(str(tmp_path / "SC4001E0-PSG.edf"), ORIGIN)
    assert events.Label.tolist() == ["n1", "n3"]
    assert events.Starttime.tolist() == [ORIGIN, ORIGIN + pd.Timedelta("60s")]
    assert events.Endtime.tolist() == [ORIGIN + pd.Timedelta("30s"), ORIGIN + pd.Timedelta("90s")]
    assert events.SourceLabel.tolist() == ["Sleep stage 1", "Sleep stage 4"]


def test_yaml_annotation_order_is_preserved_separately_from_outputs():
    spec = {
        "name": "sleepwalker.datasets.HSP.HSP",
        "channels": [],
        "sample_frequency": 1,
        "annotation_labels": ["arousal", "sleep"],
        "output_classes": ["no_arousal", "arousal"],
        "prepare_target": {
            "name": "sleepwalker.trainer.utils.targets.prepare_multiclass_target",
            "target_resolution": "1s",
            "target_classes": ["no_arousal", "arousal"],
        },
    }
    dataset = build_dataset(spec)
    assert dataset.label_classes == spec["annotation_labels"]
    assert dataset.get_classes() == ["no_arousal", "arousal"]
    assert "sleep" in dataset.label_classes and "sleep" not in dataset.get_classes()
    invalid = {**spec, "output_classes": ["arousal", "no_arousal"], "prepare_target": {**spec["prepare_target"], "target_classes": ["no_arousal", "arousal"]}}
    with pytest.raises(ValueError, match="must match output_classes"):
        build_dataset(invalid)


def test_isruc_defaults_to_independent_annotators_and_preserves_missing_alternate(monkeypatch):
    def load_annotations(path, annotator, start):
        if annotator == "2":
            raise FileNotFoundError("alternate not available")
        return pd.DataFrame({"Label": ["N2"], "Starttime": [start], "Endtime": [start + pd.Timedelta("30s")]})

    monkeypatch.setattr("sleepwalker.datasets.ISRUC.load_dataframe", load_annotations)
    reader = ISRUC(channels=[], sample_frequency=1, annotation_labels=["n2"])
    assert reader.has_extra_target()
    assert reader.get_event_df("recording.edf", ORIGIN).Label.tolist() == ["n2"]
    assert reader.get_extra_event_df("recording.edf", ORIGIN) is None
