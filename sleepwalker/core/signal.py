from __future__ import annotations

from contextlib import redirect_stdout
import io
from typing import Any,  Dict, List, Optional, Union

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
from scipy.signal import resample_poly

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

    def _try_fix_via_mne(path, path_out):
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            raw.export(path_out, fmt="edf", physical_range=(-32768, 32767), verbose="ERROR", overwrite=True)
            logger.info("Re-exported via MNE to fix remaining structural inconsistencies.")
            return True
        except Exception as e:
            logger.info(f"MNE fallback failed: {e}")
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
        # Still check if file can be read by pyedflib or MNE, but don't modify anything
        readable_pyedflib = _check_if_readable(path_in)
        readable_mne = False
        try:
            _ = mne.io.read_raw_edf(path_in, preload=False, verbose="ERROR")
            readable_mne = True
        except Exception as e:
            logger.info(f"MNE read test failed: {e}")

        return False

    # determine output path
    path_out = path_out or path_in
    _save_edf(path_out, edf)

    # verify readability
    readable = _check_if_readable(path_out)
    if not readable:
        success = _try_fix_via_mne(path_out, path_out)
        if success:
            readable = _check_if_readable(path_out)

    if readable:    # fallback: try reloading via MNE and re-exporting
        logger.info(f"{name}: successfully fixed and verified")
    else:
        logger.info(f"{name}: still not readable after fix")

    return readable

def read_edf_meta(edf: Union[str, pyedflib.EdfReader], verbose: bool = False) -> Dict[str, Any]:
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
    close_after = isinstance(edf, str)

    f = None
    try:
        if close_after:
            text_trap = io.StringIO()
            with redirect_stdout(text_trap):
                f = pyedflib.EdfReader(
                    edf,
                    annotations_mode=DO_NOT_READ_ANNOTATIONS,
                    check_file_size=DO_NOT_CHECK_FILE_SIZE,
                )
        else:
            f = edf  # already open handle

        labels = f.getSignalLabels()
        fs = {lab: float(f.getSampleFrequency(i)) for i, lab in enumerate(labels)}
        duration_s = float(f.getFileDuration())
        start = pd.Timestamp(f.getStartdatetime()).tz_localize(None)
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
            logger.warning(f"pyEDFlib failed to read {edf}: {e}")

        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")

        labels = raw.ch_names
        fs = {lab: float(raw.info["sfreq"]) for lab in labels}
        start = raw.info["meas_date"]
        duration_s = raw.n_times / raw.info["sfreq"]
        if start is None:
            end = None
        else:
            end = start + pd.to_timedelta(f"{duration_s}s")

        return {
            "start": pd.Timestamp(start).tz_localize(None),
            "end": pd.Timestamp(end).tz_localize(None),
            "duration_s": duration_s,
            "signals": labels,
            "fs": fs,
            "source": "mne",
        }
    finally:
        if close_after:
            try:
                f.close()
            except Exception:
                pass

def edf_to_df(
    edf: Union[str, pyedflib.EdfReader],
    channels: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    frequency: float,
    how: str = "nearest",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read raw samples for one or more channels between [start, end).

    Parameters
    ----------
    edf : str | pyedflib.EdfReader
        Either the EDF file path (opened/closed internally)
        or an already opened EdfReader handle.
    channels : list of str
        Channel names to extract.
    start, end : pd.Timestamp or None
        Time range to extract. If None, full file is used.
    frequency : float
        Target resampling frequency in Hz.
    how : {'nearest', 'mean', 'max'}, default 'nearest'
        Resampling strategy.
    verbose : bool
        If True, prints diagnostic information.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamps, columns=channels.
    """
    close_after = isinstance(edf, str)

    try:
        if close_after:
            f = pyedflib.EdfReader(edf, annotations_mode=DO_NOT_READ_ANNOTATIONS, check_file_size=DO_NOT_CHECK_FILE_SIZE)
        else:
            f = edf 

        text_trap = io.StringIO()
        with redirect_stdout(text_trap):
            labels = f.getSignalLabels()
            if not labels:
                return pd.DataFrame()

            dfs = []
            file_start = pd.Timestamp(f.getStartdatetime()).tz_localize(None)
            duration_s = float(f.getFileDuration())
            file_end = file_start + pd.to_timedelta(f"{duration_s}s")

            start_ = start or file_start
            end_ = end or file_end

            for ch in channels:
                if ch not in labels:
                    continue
                idx = labels.index(ch)
                
                fs = float(f.getSampleFrequency(idx))
                dt = pd.to_timedelta(1.0 / fs, unit="s")

                # compute indices
                i0 = max(int((start_ - file_start) / dt), 0)
                i1 = max(int((end_ - file_start) / dt), i0 + 1)
                x = f.readSignal(idx, start=i0, n=i1 - i0, digital=False)

                # resample in numpy for speed
                if len(x) == 0:
                    continue

                df = pd.DataFrame(x, columns=[ch], index=pd.date_range(start=start_, periods=len(x), freq=dt))
                # df.index = pd.date_range(start=start_, periods=len(x), freq=dt)
                resample_rate = pd.to_timedelta(1.0 / frequency, unit="s")
                if how == "mean":
                    df = df.resample(resample_rate).mean()
                elif how == "max":
                    df = df.resample(resample_rate).max()
                else:
                    df = df.resample(resample_rate).nearest()
                dfs.append(df)

                # if fs != frequency:
                #     step = fs / frequency
                #     if how == "mean":
                #         step_int = int(round(step))
                #         n_full = len(x) // step_int * step_int
                #         x = x[:n_full].reshape(-1, step_int).mean(axis=1)
                #     elif how == "max":
                #         step_int = int(round(step))
                #         n_full = len(x) // step_int * step_int
                #         x = x[:n_full].reshape(-1, step_int).max(axis=1)
                #     else:  # nearest
                #         x = x[::int(round(step))]

                #     dt = pd.to_timedelta(1.0 / frequency, unit="s")    
                
                # build DataFrame
                #idx_range = pd.date_range(start=start_, periods=len(x), freq=dt)
                #dfs.append(pd.DataFrame({ch: x}, index=idx_range))
            
            if not dfs:
                return pd.DataFrame()
            elif len(dfs) > 1:
                # Merge channels, fill small gaps if needed
                out = pd.concat(dfs, axis=1, join="outer").ffill().bfill()
            else:
                out = dfs[0]

            return out
    except Exception as e:
        if isinstance(edf, str):
            edf_path = edf
        else:
            return pd.DataFrame()
            
        if verbose:
            logger.warning(f"pyEDFlib failed to read {edf_path}: {e} - falling back to mne backend")

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

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
        file_start = file_start.tz_localize(None)
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
        
        # This version should be faster than the above version, but it is not on our system as it seems (tested on 2025-10-20 on a30 node with data loaded from cephfs). In any case, performance difference was in ~20% range
        # data, times = raw.get_data(picks=available, tmin=tmin, tmax=tmax, return_times=True) 

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
    # finally:
    #     if close_after:
    #         try:
    #             f.close()
    #         except Exception:
    #             pass
            
# def edf_to_df(
#     edf_path: str,
#     channels: List[str],
#     start: Optional[pd.Timestamp],
#     end: Optional[pd.Timestamp],
#     frequency: float,
#     how: str = "nearest",
#     verbose : bool = False
# ) -> pd.DataFrame:
#     """Read raw samples for one or more channels between [start, end).

#     Attempts pyEDFlib first, then falls back to MNE if pyEDFlib fails.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame indexed by timestamps, columns = channels.
#     """
#     try:
#         text_trap = io.StringIO()
#         with redirect_stdout(text_trap), pyedflib.EdfReader(
#             edf_path,
#             annotations_mode=DO_NOT_READ_ANNOTATIONS,
#             check_file_size=DO_NOT_CHECK_FILE_SIZE,
#         ) as f:
#             labels = f.getSignalLabels()
#             dfs = []

#             for ch in channels:
#                 if ch not in labels:
#                     continue
#                 idx = labels.index(ch)
#                 fs = float(f.getSampleFrequency(idx))
#                 dt = pd.to_timedelta(f"{1.0 / fs}s")
#                 file_start = pd.Timestamp(f.getStartdatetime()).tz_localize(None)
#                 duration_s = float(f.getFileDuration())
#                 file_end = file_start + pd.to_timedelta(f"{duration_s}s")
#                 start_ = start or file_start
#                 end_ = end or file_end

#                 i0 = max(int((start_ - file_start) / dt), 0)
#                 i1 = max(int((end_ - file_start) / dt), i0 + 1)
#                 x = f.readSignal(idx, start=i0, n=i1 - i0, digital=False)

#                 df = pd.DataFrame(x, columns=[ch])
#                 df.index = pd.date_range(start=start_, periods=len(x), freq=dt)
#                 resample_rate = pd.to_timedelta(1.0 / frequency, unit="s")
#                 if how == "mean":
#                     df = df.resample(resample_rate).mean()
#                 elif how == "max":
#                     df = df.resample(resample_rate).max()
#                 else:
#                     df = df.resample(resample_rate).nearest()
#                 dfs.append(df)

#             if dfs:
#                 return pd.concat(dfs, axis=1, join="outer").ffill().bfill()
#             return pd.DataFrame()

#     except Exception as e:
#         if verbose:
#             logger.warning(f"pyEDFlib failed to read {edf_path}: {e}")

#         raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

#         # Filter channels
#         available = [ch for ch in channels if ch in raw.ch_names]
#         if not available:
#             if verbose: logger.warning(f"No requested channels found in {edf_path}")
#             return pd.DataFrame()

#         sfreq = raw.info["sfreq"]
#         meas_date = raw.info.get("meas_date", None)
#         if isinstance(meas_date, tuple):
#             meas_date = meas_date[0]
#         file_start = pd.Timestamp(meas_date or pd.Timestamp.now())
#         file_start = file_start.tz_localize(None)
#         duration_s = (raw.n_times - 1) / sfreq
#         file_end = file_start + pd.to_timedelta(f"{duration_s}s")
#         start_ = start or file_start
#         end_ = end or file_end
        
#         if end_ <= file_start:
#             if verbose: logger.warning(f"Invalid crop range for {edf_path}")
#             return pd.DataFrame()
        
#         tmin = max(0.0, (start_ - file_start).total_seconds())
#         tmax = max(0.0, (end_ - file_start).total_seconds())
#         if tmax > raw.times[-1]: tmax = raw.times[-1]

#         raw.crop(tmin=tmin, tmax=tmax)
#         data, times = raw.get_data(picks=available, return_times=True)
        
#         # This version should be faster than the above version, but it is not on our system as it seems (tested on 2025-10-20 on a30 node with data loaded from cephfs). In any case, performance difference was in ~20% range
#         # data, times = raw.get_data(picks=available, tmin=tmin, tmax=tmax, return_times=True) 

#         dates = pd.to_datetime(start_.value + (times * 1e9).astype(np.int64))
#         df = pd.DataFrame(data.T, index=dates, columns=available)

#         # resample
#         resample_rate = pd.to_timedelta(1.0 / frequency, unit="s")
#         if how == "mean":
#             df = df.resample(resample_rate).mean()
#         elif how == "max":
#             df = df.resample(resample_rate).max()
#         else:
#             df = df.resample(resample_rate).nearest()

#         return df.ffill().bfill()

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