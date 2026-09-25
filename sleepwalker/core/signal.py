"""Low-level EDF metadata, signal loading, and header-repair helpers.

These utilities are shared by dataset adapters and training scripts throughout
the repository. The code prefers pyEDFlib when possible and falls back to MNE
for some read paths and for one repair path in ``fix_edf_header``.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
from fractions import Fraction
import io
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyedflib
from pyedflib import DO_NOT_CHECK_FILE_SIZE, DO_NOT_READ_ANNOTATIONS, EdfReader
import mne 
from scipy.signal import resample_poly

from sleepwalker.utils import logger

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

def read_edf_meta(
    edf: Union[str, os.PathLike, pyedflib.EdfReader],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Read basic metadata for an EDF file or already-open reader.

    Args:
        edf: EDF path or open ``pyedflib.EdfReader`` handle.
        verbose: Whether to log read failures before falling back to MNE.

    Returns:
        A dictionary containing at least ``start``, ``end``,
        ``duration_s``, ``signals``, ``fs``, and ``source``.

    Notes:
        The helper tries pyEDFlib first and falls back to MNE on failure. When
        MNE is used, per-channel sampling frequencies are inferred from the
        global raw object frequency.
    """
    close_after = isinstance(edf, (str, os.PathLike))
    if isinstance(edf, os.PathLike):
        edf = os.fspath(edf)

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
        units = {lab: str(f.getPhysicalDimension(i)).strip() for i, lab in enumerate(labels)}
        duration_s = float(f.getFileDuration())
        start = pd.Timestamp(f.getStartdatetime()).tz_localize(None)
        end = start + pd.to_timedelta(f"{duration_s}s")
        return {
            "start": start,
            "end": end,
            "duration_s": duration_s,
            "signals": labels,
            "fs": fs,
            "units": units,
            "source": "pyedflib",
        }
    except Exception as e:
        if verbose:
            logger.warning(f"pyEDFlib failed to read {edf}: {e}")

        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")

        labels = raw.ch_names
        fs = {lab: float(raw.info["sfreq"]) for lab in labels}
        original_units = getattr(raw, "_orig_units", {})
        units = {lab: str(original_units.get(lab, "")).strip() for lab in labels}
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
            "units": units,
            "source": "mne",
        }
    finally:
        if close_after:
            try:
                f.close()
            except Exception:
                pass

def polyphase_resample_frame(frame: pd.DataFrame, source_frequency: float, target_frequency: float) -> pd.DataFrame:
    """Resample uniformly sampled channels with an antialiasing polyphase filter."""
    if frame.empty:
        return frame
    if source_frequency <= 0 or target_frequency <= 0:
        raise ValueError("Source and target frequencies must be positive.")
    if np.isclose(source_frequency, target_frequency):
        return frame
    ratio = Fraction(float(target_frequency) / float(source_frequency)).limit_denominator(10000)
    values = resample_poly(frame.to_numpy(), ratio.numerator, ratio.denominator, axis=0)
    index = pd.date_range(start=frame.index[0], periods=len(values), freq=pd.to_timedelta(1.0 / target_frequency, unit="s"))
    return pd.DataFrame(values, columns=frame.columns, index=index)


def read_edf_native(
    edf: Union[str, os.PathLike, pyedflib.EdfReader],
    channels: Sequence[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    verbose: bool = False,
) -> dict[float, pd.DataFrame]:
    """Read one EDF interval without changing any channel's sampling rate.

    Channels with the same native sampling frequency share a DataFrame. This
    representation keeps the original sample grids intact while allowing one
    physical EDF read to supply several independently resampled model windows.
    The returned timestamps describe the actual native sample positions.
    """
    close_after = isinstance(edf, (str, os.PathLike))
    if isinstance(edf, os.PathLike):
        edf = os.fspath(edf)
    reader = None

    try:
        if close_after:
            reader = pyedflib.EdfReader(edf, annotations_mode=DO_NOT_READ_ANNOTATIONS, check_file_size=DO_NOT_CHECK_FILE_SIZE)
        else:
            reader = edf

        text_trap = io.StringIO()
        with redirect_stdout(text_trap):
            labels = reader.getSignalLabels()
            if not labels:
                return {}

            file_start = pd.Timestamp(reader.getStartdatetime()).tz_localize(None)
            file_end = file_start + pd.to_timedelta(float(reader.getFileDuration()), unit="s")
            start_date = file_start if start is None else pd.Timestamp(start)
            end_date = file_end if end is None else pd.Timestamp(end)
            if end_date <= start_date:
                raise ValueError(f"EDF interval must have positive duration, got [{start_date}, {end_date}).")

            groups: dict[float, list[tuple[str, int]]] = defaultdict(list)
            for channel in channels:
                if channel in labels:
                    index = labels.index(channel)
                    groups[float(reader.getSampleFrequency(index))].append((channel, index))

            result = {}
            for frequency, group in groups.items():
                period = pd.to_timedelta(1.0 / frequency, unit="s")
                first = max(int((start_date - file_start) / period), 0)
                stop = max(int((end_date - file_start) / period), first + 1)
                arrays = {}
                for channel, index in group:
                    values = reader.readSignal(index, start=first, n=stop - first, digital=False)
                    if len(values) > 0:
                        arrays[channel] = values
                if not arrays:
                    continue
                length = min(len(values) for values in arrays.values())
                arrays = {channel: values[:length] for channel, values in arrays.items()}
                native_start = file_start + first * period
                result[frequency] = pd.DataFrame(arrays, index=pd.date_range(start=native_start, periods=length, freq=period))
            return result
    except Exception as error:
        if not isinstance(edf, str):
            return {}
        if verbose:
            logger.warning(f"pyEDFlib failed to read {edf}: {error} - falling back to mne backend")

        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
        available = [channel for channel in channels if channel in raw.ch_names]
        if not available:
            if verbose:
                logger.warning(f"No requested channels found in {edf}")
            return {}

        frequency = float(raw.info["sfreq"])
        measurement_date = raw.info.get("meas_date")
        if isinstance(measurement_date, tuple):
            measurement_date = measurement_date[0]
        file_start = pd.Timestamp(measurement_date or pd.Timestamp.now()).tz_localize(None)
        file_end = file_start + pd.to_timedelta(raw.n_times / frequency, unit="s")
        start_date = file_start if start is None else pd.Timestamp(start)
        end_date = file_end if end is None else pd.Timestamp(end)
        first = max(int((start_date - file_start).total_seconds() * frequency), 0)
        stop = min(max(int((end_date - file_start).total_seconds() * frequency), first + 1), raw.n_times)
        data = raw.get_data(picks=available, start=first, stop=stop)
        native_start = file_start + pd.to_timedelta(first / frequency, unit="s")
        frame = pd.DataFrame(data.T, index=pd.date_range(start=native_start, periods=data.shape[1], freq=pd.to_timedelta(1.0 / frequency, unit="s")), columns=available)
        return {frequency: frame}
    finally:
        if close_after and reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def resample_native_signals(
    signals: Mapping[float, pd.DataFrame],
    channels: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: float,
    how: str = "nearest",
) -> pd.DataFrame:
    """Slice one logical window from native signals and resample it.

    Slicing happens before resampling so the result is independent of other
    windows that happened to share the same native EDF read.
    """
    if how not in {"nearest", "mean", "max", "polyphase"}:
        raise ValueError(f"Unknown EDF resampling mode {how!r}.")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"Signal interval must have positive duration, got [{start}, {end}).")

    resample_period = pd.to_timedelta(1.0 / frequency, unit="s")
    frames = []
    for source_frequency, native in signals.items():
        selected = [channel for channel in channels if channel in native.columns]
        if not selected or native.empty:
            continue

        first = max(int(native.index.searchsorted(start, side="right")) - 1, 0)
        stop = max(int(native.index.searchsorted(end, side="left")), first + 1)
        frame = native.iloc[first:stop][selected].copy()
        if frame.empty:
            continue
        frame.index = pd.date_range(start=start, periods=len(frame), freq=pd.to_timedelta(1.0 / source_frequency, unit="s"))
        if how == "polyphase":
            frame = polyphase_resample_frame(frame, source_frequency, frequency)
        elif how == "mean":
            frame = frame.resample(resample_period).mean()
        elif how == "max":
            frame = frame.resample(resample_period).max()
        else:
            frame = frame.resample(resample_period).nearest()
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    output = frames[0] if len(frames) == 1 else pd.concat(frames, axis=1, join="outer").ffill().bfill()
    return output[[channel for channel in channels if channel in output.columns]]


def edf_to_df(
    edf: Union[str, os.PathLike, pyedflib.EdfReader],
    channels: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    frequency: float,
    how: str = "nearest",
    verbose: bool = False,
) -> pd.DataFrame:
    """Read EDF signal samples into a pandas DataFrame.

    Args:
        edf: EDF path or open ``pyedflib.EdfReader`` handle.
        channels: Channel names to extract. Missing channels are skipped.
        start: Optional extraction start timestamp. ``None`` means start of
            file.
        end: Optional extraction end timestamp. ``None`` means end of file.
        frequency: Target resampling frequency in Hz.
        how: Resampling mode. Supported values are ``"nearest"``, ``"mean"``,
            ``"max"``, and antialiased ``"polyphase"``.
        verbose: Whether to log backend failures and some missing-channel
            situations.

    Returns:
        A time-indexed DataFrame whose columns correspond to the requested
        channels that were actually found.

    Notes:
        The helper uses pyEDFlib first and falls back to MNE when pyEDFlib
        fails. The current code treats the output as suitable for downstream
        resampling and gap filling, but exact backend equivalence is not
        documented.
    """
    native = read_edf_native(edf, channels, start, end, verbose=verbose)
    if not native:
        return pd.DataFrame()
    effective_start = pd.Timestamp(start) if start is not None else min(frame.index[0] for frame in native.values())
    effective_end = pd.Timestamp(end) if end is not None else max(frame.index[-1] + pd.to_timedelta(1.0 / native_frequency, unit="s") for native_frequency, frame in native.items())
    return resample_native_signals(native, channels, effective_start, effective_end, frequency, how)
