"""Regression tests for the grouped-by-fs implementation of
`sleepwalker.core.signal.edf_to_df`.

Asserts that the current implementation (which buckets channels by native
sample frequency and calls pandas.resample once per group) produces
element-wise the same output as the previous per-channel implementation.

Uses only the synthetic EDF files shipped in `tests/data/` so the suite runs
without any private dataset.

The previous per-channel implementation is included inline as
`edf_to_df_per_channel_reference` so the comparison stays stable even if the
library is changed later.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyedflib
import pytest
from pyedflib import DO_NOT_CHECK_FILE_SIZE, DO_NOT_READ_ANNOTATIONS

from sleepwalker.core.signal import edf_to_df, polyphase_resample_frame

DATA_DIR = Path(__file__).parent / "data"
SYNTH_PATHS: List[Path] = sorted(DATA_DIR.glob("signals_*.edf"))

# Channels in the synthetic EDFs (see tests/data/generate_data.py):
#   EEG @200 Hz
#   EOG @100 Hz
#   EMG @100 Hz
#
# That gives us one group of size 1 (EEG) and one group of size 2 (EOG+EMG),
# which covers both the single-group-single-channel and single-group-multi-channel
# branches as well as the multi-group concat+ffill+bfill branch.
ALL_CHANNELS = ["EEG", "EOG", "EMG"]


def edf_to_df_per_channel_reference(
    edf_path: str,
    channels: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    frequency: float,
    how: str = "nearest",
) -> pd.DataFrame:
    """Faithful standalone copy of the previous per-channel implementation of
    `edf_to_df` (pyEDFlib path only, no MNE fallback, no stdout redirect).
    This is the reference against which the grouped variant is compared.
    """
    f = pyedflib.EdfReader(
        edf_path,
        annotations_mode=DO_NOT_READ_ANNOTATIONS,
        check_file_size=DO_NOT_CHECK_FILE_SIZE,
    )
    try:
        labels = f.getSignalLabels()
        if not labels:
            return pd.DataFrame()
        file_start = pd.Timestamp(f.getStartdatetime()).tz_localize(None)
        duration_s = float(f.getFileDuration())
        file_end = file_start + pd.to_timedelta(f"{duration_s}s")
        start_ = start or file_start
        end_ = end or file_end
        resample_rate = pd.to_timedelta(1.0 / frequency, unit="s")

        dfs = []
        for ch in channels:
            if ch not in labels:
                continue
            idx = labels.index(ch)
            fs = float(f.getSampleFrequency(idx))
            dt = pd.to_timedelta(1.0 / fs, unit="s")
            i0 = max(int((start_ - file_start) / dt), 0)
            i1 = max(int((end_ - file_start) / dt), i0 + 1)
            x = f.readSignal(idx, start=i0, n=i1 - i0, digital=False)
            if len(x) == 0:
                continue
            df = pd.DataFrame(
                x,
                columns=[ch],
                index=pd.date_range(start=start_, periods=len(x), freq=dt),
            )
            if how == "mean":
                df = df.resample(resample_rate).mean()
            elif how == "max":
                df = df.resample(resample_rate).max()
            else:
                df = df.resample(resample_rate).nearest()
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        if len(dfs) > 1:
            return pd.concat(dfs, axis=1, join="outer").ffill().bfill()
        return dfs[0]
    finally:
        f.close()


def _assert_frames_equivalent(
    reference: pd.DataFrame, candidate: pd.DataFrame, how: str
) -> None:
    """Value-wise equivalence between reference and candidate."""
    assert list(reference.columns) == list(candidate.columns), (
        f"column order differs: {list(reference.columns)} vs "
        f"{list(candidate.columns)}"
    )
    assert reference.index.equals(candidate.index), "index mismatch"
    if how == "nearest":
        np.testing.assert_array_equal(reference.to_numpy(), candidate.to_numpy())
    else:
        np.testing.assert_allclose(
            reference.to_numpy(),
            candidate.to_numpy(),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.fixture(params=[str(p) for p in SYNTH_PATHS])
def synth_edf(request) -> str:
    return request.param


# ---------------------------------------------------------------------------
# Equivalence across channel selections, target frequencies, aggregation methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["nearest", "mean", "max"])
@pytest.mark.parametrize(
    "channels",
    [
        ["EEG"],                  # single channel, single group
        ["EOG"],                  # single channel, single group
        ["EOG", "EMG"],           # 2 channels, both in the 100 Hz group
        ["EEG", "EOG"],           # 2 channels, 2 different groups
        ["EEG", "EOG", "EMG"],    # 3 channels, 2 groups
        ["EMG", "EEG", "EOG"],    # same set, different request order
        ["EOG", "EEG"],           # reversed order of first multi-group case
    ],
)
@pytest.mark.parametrize("target_fs", [50.0, 100.0, 37.0])
def test_equivalence_full_file(synth_edf, channels, how, target_fs):
    """Read whole file; check shipping impl equals the per-channel reference."""
    reference = edf_to_df_per_channel_reference(
        synth_edf, channels, start=None, end=None, frequency=target_fs, how=how
    )
    current = edf_to_df(
        synth_edf, channels, start=None, end=None, frequency=target_fs, how=how,
    )
    _assert_frames_equivalent(reference, current, how)


@pytest.mark.parametrize("how", ["nearest", "mean", "max"])
def test_equivalence_with_window_offset(synth_edf, how):
    """Window starting mid-file exercises the non-None start/end branch."""
    with pyedflib.EdfReader(
        synth_edf,
        annotations_mode=DO_NOT_READ_ANNOTATIONS,
        check_file_size=DO_NOT_CHECK_FILE_SIZE,
    ) as f:
        file_start = pd.Timestamp(f.getStartdatetime()).tz_localize(None)
        duration_s = float(f.getFileDuration())

    win_start = file_start + pd.Timedelta(seconds=min(137, max(0, duration_s - 400)))
    win_end = win_start + pd.Timedelta(seconds=min(330, duration_s / 2))

    reference = edf_to_df_per_channel_reference(
        synth_edf, ALL_CHANNELS, win_start, win_end, frequency=100.0, how=how
    )
    current = edf_to_df(
        synth_edf, ALL_CHANNELS, win_start, win_end, frequency=100.0, how=how
    )
    _assert_frames_equivalent(reference, current, how)


def test_unknown_channels_are_dropped_consistently(synth_edf):
    reference = edf_to_df_per_channel_reference(
        synth_edf, ["EEG", "DOES_NOT_EXIST", "EOG"], None, None, 100.0
    )
    current = edf_to_df(
        synth_edf, ["EEG", "DOES_NOT_EXIST", "EOG"], None, None, 100.0
    )
    _assert_frames_equivalent(reference, current, "nearest")


def test_all_channels_missing_returns_empty(synth_edf):
    reference = edf_to_df_per_channel_reference(
        synth_edf, ["NOT_A_CHANNEL_1", "NOT_A_CHANNEL_2"], None, None, 100.0
    )
    current = edf_to_df(
        synth_edf, ["NOT_A_CHANNEL_1", "NOT_A_CHANNEL_2"], None, None, 100.0
    )
    assert reference.empty and current.empty


def test_polyphase_resampling_has_exact_target_grid():
    index = pd.date_range("2025-01-01", periods=200, freq=pd.to_timedelta(1 / 200, unit="s"))
    frame = pd.DataFrame({"signal": np.sin(np.arange(200) / 10)}, index=index)

    result = polyphase_resample_frame(frame, source_frequency=200, target_frequency=50)

    assert result.shape == (50, 1)
    assert result.index[0] == index[0]
    assert result.index[1] - result.index[0] == pd.Timedelta(milliseconds=20)
    assert np.isfinite(result.to_numpy()).all()
