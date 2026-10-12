import numpy as np
import pandas as pd
import pytest

from sleepwalker.datasets.HSP import HSP


@pytest.mark.parametrize("sane_labels", [True, False])
def test_hsp_recovers_missing_durations_from_stage_changes_and_event_text(tmp_path, sane_labels):
    edf_path = tmp_path / "patient_task-psg_eeg.edf"
    annotations = pd.DataFrame({
        "event": ["Stage - No Stage", "Stage - W", "* Arousal - Dur:  3.0 sec. - PLM", "STAGE TRANSITION", "Stage - N2", "Respiratory Event - Dur: 14.6 sec. - Hypopnea", "Stage - No Stage"],
        "time": ["23:58:00", "23:59:00", "23:59:10", "23:59:20", "00:00:00", "00:00:10", "00:02:00"],
        "duration": [np.nan] * 7,
    })
    annotations.to_csv(edf_path.with_name("patient_task-psg_annotations.csv"), index=False)

    result = HSP(sane_labels=sane_labels, channels=[], sample_frequency=100).get_event_df(str(edf_path), pd.Timestamp("2024-01-01 23:58:00"))

    assert result.loc[[0, 1, 2, 4, 5], "Duration"].tolist() == [60.0, 60.0, 3.0, 120.0, 14.6]
    assert result.loc[1, "Endtime"] == pd.Timestamp("2024-01-02 00:00:00")
    assert result.loc[4, "Endtime"] == pd.Timestamp("2024-01-02 00:02:00")
    assert result.loc[5, "Endtime"] == pd.Timestamp("2024-01-02 00:00:24.600")
    assert pd.isna(result.loc[3, "Endtime"])
    assert pd.isna(result.loc[6, "Endtime"])
    expected = ["stage - no stage", "wake", "arousal", "stage transition", "n2", "hypopnea", "stage - no stage"] if sane_labels else annotations.event.str.lower().tolist()
    assert result.Label.tolist() == expected


def test_hsp_keeps_explicit_durations_and_leaves_unbounded_stage_unknown(tmp_path):
    edf_path = tmp_path / "patient_task-psg_eeg.edf"
    annotations = pd.DataFrame({
        "event": ["Stage - W", "* Arousal - Dur: 3.0 sec. - PLM", "Stage - N2", "LIGHTS ON"],
        "time": ["00:00:00", "00:00:10", "00:01:00", "00:02:00"],
        "duration": [30.0, 5.0, np.nan, np.nan],
    })
    annotations.to_csv(edf_path.with_name("patient_task-psg_annotations.csv"), index=False)

    result = HSP(channels=[], sample_frequency=100).get_event_df(str(edf_path), pd.Timestamp("2024-01-01"))

    assert result.loc[:1, "Duration"].tolist() == [30.0, 5.0]
    assert pd.isna(result.loc[2, "Endtime"])
    assert pd.isna(result.loc[3, "Endtime"])
