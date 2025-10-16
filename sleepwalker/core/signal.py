from __future__ import annotations

from typing import Any,  Dict, List, Optional

import numpy as np
import pandas as pd
import pyedflib
from pyedflib import DO_NOT_READ_ANNOTATIONS, DO_NOT_CHECK_FILE_SIZE
import mne 

from sleepwalker.utils import logger

import os
import numpy as np
from typing import List, Tuple, Optional
from pyedflib import EdfReader


def fix_edf_header(path_in: str, path_out: Optional[str] = None, dry: bool = False) -> Tuple[bool, List[str]]:
    """
    Attempt to repair common header and consistency issues in an EDF file.

    This function:
      1. Rewrites incorrect or unset "number of data records".
      2. Removes overhanging bytes after data records.
      3. Corrects invalid digital min/max values that exceed the EDF limits
         (-32768 .. 32767).

    Parameters
    ----------
    path_in : str
        Path to the input EDF file to be checked and possibly fixed.
    path_out : str, optional
        Output path for the corrected file. If None, fixes are applied in-place.

    Returns
    -------
    success : bool
        True if the resulting EDF file is readable after attempted fixes.

    Notes
    -----
    - This function reimplements minimal EDF header parsing and writing logic.
      It does not depend on pyedflib for writing, as pyedflib cannot repair
      malformed headers.
    - Always verify the output file manually for scientific use.
    """

    def _format_float(x, l):
        s = f"{x:.{l}g}"
        if len(s) <= l:
            return f"{s:<{l}}"
        if s[l - 1] == ".":
            return s[: l - 1] + " "
        return s[:l]

    def _format_int(x, l):
        return f"{x:<{l}}"[:l]

    def _load_edf(path):
        fields = {}
        with open(path, "rb") as f:
            fields["version"] = f.read(8).decode()
            fields["patient_id"] = f.read(80).decode()
            fields["record_id"] = f.read(80).decode()
            fields["start_date"] = f.read(8).decode()
            fields["start_time"] = f.read(8).decode()
            fields["header_bytes"] = int(f.read(8).decode())
            fields["reserved"] = f.read(44).decode()
            fields["num_data_records"] = int(f.read(8).decode())
            fields["dur_data_records"] = int(float(f.read(8).decode()))
            ns = int(f.read(4).decode())
            fields["num_signals"] = ns

            # signal headers
            def _read_block(size):
                return [f.read(size).decode() for _ in range(ns)]

            fields["sig_label"] = _read_block(16)
            fields["sig_type"] = _read_block(80)
            fields["sig_phys_dim"] = _read_block(8)
            fields["sig_phys_min"] = [float(f.read(8).decode()) for _ in range(ns)]
            fields["sig_phys_max"] = [float(f.read(8).decode()) for _ in range(ns)]
            fields["sig_dig_min"] = [int(f.read(8).decode()) for _ in range(ns)]
            fields["sig_dig_max"] = [int(f.read(8).decode()) for _ in range(ns)]
            fields["sig_prefilter"] = _read_block(80)
            fields["sig_num_samples"] = [int(f.read(8).decode()) for _ in range(ns)]
            fields["sig_reserved"] = _read_block(32)

            # read data records
            rec_size = 2 * sum(fields["sig_num_samples"])
            fields["records"] = []
            while True:
                data = f.read(rec_size)
                if len(data) < rec_size:
                    break
                rec, ix = [], 0
                for k in fields["sig_num_samples"]:
                    rec.append(np.frombuffer(data[ix : ix + 2 * k], dtype=np.int16))
                    ix += 2 * k
                fields["records"].append(rec)

            # remainder (should not exist)
            fields["rest"] = f.read()

        return fields

    def _save_edf(path, fields):
        with open(path, "wb") as f:
            for x in [
                "version",
                "patient_id",
                "record_id",
                "start_date",
                "start_time",
            ]:
                f.write(fields[x].encode())

            f.write(_format_int(fields["header_bytes"], 8).encode())
            f.write(fields["reserved"].encode())
            f.write(_format_int(fields["num_data_records"], 8).encode())
            f.write(_format_int(fields["dur_data_records"], 8).encode())
            f.write(_format_int(fields["num_signals"], 4).encode())

            ns = fields["num_signals"]
            f.write("".join(fields["sig_label"]).encode())
            f.write("".join(fields["sig_type"]).encode())
            f.write("".join(fields["sig_phys_dim"]).encode())
            f.write("".join(_format_float(x, 8) for x in fields["sig_phys_min"]).encode())
            f.write("".join(_format_float(x, 8) for x in fields["sig_phys_max"]).encode())
            f.write("".join(_format_int(x, 8) for x in fields["sig_dig_min"]).encode())
            f.write("".join(_format_int(x, 8) for x in fields["sig_dig_max"]).encode())
            f.write("".join(fields["sig_prefilter"]).encode())
            f.write("".join(_format_int(x, 8) for x in fields["sig_num_samples"]).encode())
            f.write("".join(fields["sig_reserved"]).encode())

            for rec in fields["records"]:
                for arr in rec:
                    f.write(arr.tobytes())

            f.write(fields["rest"])

    def _check_if_readable(path):
        try:
            reader = EdfReader(path)
            reader.close()
            return True
        except OSError as e:
            logger.info(str(e))
            return False

    # --- main repair logic ---
    edf = _load_edf(path_in)
    name = os.path.basename(path_in)

    # fix num_data_records
    actual = len(edf["records"])
    if (a := edf["num_data_records"]) != actual:
        logger.info(f"{name}: num_data_records corrected ({a} → {actual})")
        edf["num_data_records"] = actual

    # drop trailing bytes
    if edf.get("rest"):
        logger.info(f"{name}: removed {len(edf['rest'])} overhanging bytes")
        edf["rest"] = b""

    # fix digital min/max
    sdmin = np.asarray(edf["sig_dig_min"])
    sdmax = np.asarray(edf["sig_dig_max"])
    if np.any(sdmin < -32768) or np.any(sdmax > 32767):
        logger.info(f"{name}: fixing invalid digital min/max values")

        n_signals = len(edf["sig_num_samples"])
        true_min = np.full(n_signals,  32767, dtype=np.int32)
        true_max = np.full(n_signals, -32768, dtype=np.int32)

        # Vectorized aggregation per signal
        # Each record is a list of arrays, one per signal
        # We'll aggregate min/max for each signal in one pass
        # without constructing full large arrays in memory.
        for signal_idx in range(n_signals):
            # collect one array per record lazily
            sig_arrays = [rec[signal_idx] for rec in edf["records"]]
            # stack efficiently along first axis
            stacked = np.concatenate(sig_arrays, dtype=np.int16)
            true_min[signal_idx] = stacked.min()
            true_max[signal_idx] = stacked.max()

        edf["sig_dig_min"] = [
            int(true_min[i]) if sdmin[i] < -32768 else int(sdmin[i])
            for i in range(n_signals)
        ]
        edf["sig_dig_max"] = [
            int(true_max[i]) if sdmax[i] > 32767 else int(sdmax[i])
            for i in range(n_signals)
        ]

    # --- Dry-run mode ---
    if dry:
        return False

    # determine output path
    path_out = path_out or path_in
    _save_edf(path_out, edf)

    # verify readability
    readable = _check_if_readable(path_out)
    if readable:
        logger.info(f"{name}: successfully fixed and verified")
    else:
        logger.info(f"{name}: still not readable after fix")

    return readable

def read_edf_meta(edf_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Read basic metadata about an EDF file.

    Attempts pyEDFlib first, then falls back to MNE if pyEDFlib fails.

    Returns
    -------
    dict with keys:
        start : datetime
        end : datetime
        duration_s : float
        signals : list[str]
        fs : dict[str, float]
    """
    try:
        with pyedflib.EdfReader(
            edf_path,
            annotations_mode=DO_NOT_READ_ANNOTATIONS,
            check_file_size=DO_NOT_CHECK_FILE_SIZE,
        ) as f:
            labels = f.getSignalLabels()
            fs = {lab: float(f.getSampleFrequency(i)) for i, lab in enumerate(labels)}
            duration_s = float(f.getFileDuration())
            start = f.getStartdatetime()
            end = start + pd.to_timedelta(f"{duration_s}s")
            return {
                "start": start,
                "end": end,
                "duration_s": duration_s,
                "signals": labels,
                "fs": fs,
                "source": "pyedflib",
            }
    except Exception as e:
        if verbose:
            logger.warning(f"pyEDFlib failed to read {edf_path}: {e}")
            logger.info(f"Falling back to MNE for {edf_path}")

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

        labels = raw.ch_names
        fs = {lab: float(raw.info["sfreq"]) for lab in labels}
        start = raw.info["meas_date"]
        duration_s = raw.n_times / raw.info["sfreq"]
        if start is None:
            end = None
        else:
            end = start + pd.to_timedelta(f"{duration_s}s")

        return {
            "start": pd.Timestamp(start),
            "end": pd.Timestamp(end),
            "duration_s": duration_s,
            "signals": labels,
            "fs": fs,
            "source": "mne",
        }


def edf_to_df(
    edf_path: str,
    channels: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    frequency: float,
    how: str = "nearest",
    verbose : bool = False
) -> pd.DataFrame:
    """Read raw samples for one or more channels between [start, end).

    Attempts pyEDFlib first, then falls back to MNE if pyEDFlib fails.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by timestamps, columns = channels.
    """
    try:
        with pyedflib.EdfReader(
            edf_path,
            annotations_mode=DO_NOT_READ_ANNOTATIONS,
            check_file_size=DO_NOT_CHECK_FILE_SIZE,
        ) as f:
            labels = f.getSignalLabels()
            dfs = []

            for ch in channels:
                if ch not in labels:
                    continue
                idx = labels.index(ch)
                fs = float(f.getSampleFrequency(idx))
                dt = pd.to_timedelta(f"{1.0 / fs}s")
                file_start = pd.Timestamp(f.getStartdatetime())
                duration_s = float(f.getFileDuration())
                file_end = file_start + pd.to_timedelta(f"{duration_s}s")
                start_ = start or file_start
                end_ = end or file_end

                i0 = max(int((start_ - file_start) / dt), 0)
                i1 = max(int((end_ - file_start) / dt), i0 + 1)
                x = f.readSignal(idx, start=i0, n=i1 - i0, digital=False)

                df = pd.DataFrame(x, columns=[ch])
                df.index = pd.date_range(start=start_, periods=len(x), freq=dt)
                resample_rate = pd.to_timedelta(1.0 / frequency, unit="s")
                if how == "mean":
                    df = df.resample(resample_rate).mean()
                elif how == "max":
                    df = df.resample(resample_rate).max()
                else:
                    df = df.resample(resample_rate).nearest()
                dfs.append(df)

            if dfs:
                return pd.concat(dfs, axis=1, join="outer").ffill().bfill()
            return pd.DataFrame()

    except Exception as e:
        if verbose:
            logger.warning(f"pyEDFlib failed to read {edf_path}: {e}")
            logger.info(f"Falling back to MNE for {edf_path}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

        # Filter channels
        available = [ch for ch in channels if ch in raw.ch_names]
        if not available:
            if verbose: logger.warning(f"No requested channels found in {edf_path}")
            return pd.DataFrame()

        sfreq = raw.info["sfreq"]
        meas_date = raw.info.get("meas_date", None)
        if isinstance(meas_date, tuple):
            meas_date = meas_date[0]
        file_start = pd.Timestamp(meas_date or pd.Timestamp.now())
        duration_s = (raw.n_times - 1) / sfreq
        file_end = file_start + pd.to_timedelta(f"{duration_s}s")
        start_ = start or file_start
        end_ = end or file_end
        
        if end_ <= file_start:
            if verbose: logger.warning(f"Invalid crop range for {edf_path}")
            return pd.DataFrame()
        
        tmin = max(0.0, (start_ - file_start).total_seconds())
        tmax = max(0.0, (end_ - file_start).total_seconds())
        if tmax > raw.times[-1]: tmax = raw.times[-1]

        raw.crop(tmin=tmin, tmax=tmax)

        data, times = raw.get_data(picks=available, return_times=True)
        dates = pd.to_datetime(start_.value + (times * 1e9).astype(np.int64))
        df = pd.DataFrame(data.T, index=dates, columns=available)

        # resample
        resample_rate = pd.to_timedelta(1.0 / frequency, unit="s")
        if how == "mean":
            df = df.resample(resample_rate).mean()
        elif how == "max":
            df = df.resample(resample_rate).max()
        else:
            df = df.resample(resample_rate).nearest()

        return df.ffill().bfill()

# def read_edf_meta(edf_path: str) -> Dict[str, Any]:
#     """Read basic metadata about an EDF file.

#     Returns dict with keys: start, end, duration_s, labels, sample_rates.
#     """
#     with pyedflib.EdfReader(edf_path, annotations_mode=DO_NOT_READ_ANNOTATIONS, check_file_size=DO_NOT_CHECK_FILE_SIZE) as f: 
#         labels = f.getSignalLabels()
#         fs = {lab: float(f.getSampleFrequency(i)) for i, lab in enumerate(labels)}
#         duration_s = float(f.getFileDuration())
#         start = f.getStartdatetime()
#         end = start + pd.to_timedelta(f"{duration_s} s")
#         return {
#             "start": start,
#             "end": end,
#             "duration_s": duration_s,
#             "signals": labels,
#             "fs": fs,
#         }

# def edf_to_df(
#     edf_path: str,
#     channels: List[str],
#     start: Optional[pd.Timestamp],
#     end: Optional[pd.Timestamp],
#     frequency: float, 
#     how: str = "nearest"
# ) -> pd.DataFrame:
#     """Read raw samples for a single channel between [start, end).

#     Returns (signal, original_dt) where original_dt is the sampling period.
#     """
#     with pyedflib.EdfReader(edf_path, annotations_mode=DO_NOT_READ_ANNOTATIONS, check_file_size=DO_NOT_CHECK_FILE_SIZE) as f:
#         labels = f.getSignalLabels()
        
#         dfs = []
#         for ch in channels:
#             if ch in labels:
#                 idx = labels.index(ch)
#                 fs = float(f.getSampleFrequency(idx))
#                 dt = pd.to_timedelta(f"{1.0 / fs}s")
#                 file_start = pd.Timestamp(f.getStartdatetime())
#                 duration_s = float(f.getFileDuration())
#                 file_end = file_start + pd.to_timedelta(f"{duration_s}s")
#                 if start is None:
#                     start = file_start 
#                 if end is None:
#                     end = file_end
#                 i0 = max(int((start - file_start) / dt), 0)
#                 i1 = max(int((end - file_start) / dt), i0 + 1)
#                 x = f.readSignal(idx, start=i0, n=i1 - i0, digital=False)

#                 df = pd.DataFrame(x, columns=[ch])
#                 df.index = pd.date_range(start=start, periods=len(x), freq=dt)
#                 if how == "mean":
#                     df = df.resample(pd.to_timedelta(1.0/frequency, unit="s")).mean()
#                 elif how == "max":
#                     df = df.resample(pd.to_timedelta(1.0/frequency, unit="s")).max()
#                 else: 
#                     df = df.resample(pd.to_timedelta(1.0/frequency, unit="s")).nearest()
#                 dfs.append(df)
#         if len(dfs) > 0:
#             return pd.concat(dfs, axis=1, join="outer").ffill().bfill()
#         else:
#             return pd.DataFrame()