"""Core dataset abstractions for EDF-backed Sleepwalker workflows.

This module defines the base dataset lifecycle used throughout the repository:
configure channels and label mappings, prepare patient-level metadata, build a
sliding-window index, and lazily materialize model-ready samples on demand.

The code is used by dataset adapters under :mod:`sleepwalker.datasets`, by
training scripts, and by deployment code. Tests in ``tests/test_datasets.py`` and
``tests/test_deployment.py`` exercise the window-building, lazy loading, and
export-related behavior documented here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import bisect
from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import partial
import multiprocessing
import numbers
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, cast

import numpy as np
import pandas as pd
import pyedflib
import torch
from torch.utils.data import Dataset, default_collate

from sleepwalker.utils import logger
from sleepwalker.core.signal import edf_to_df, read_edf_meta, read_edf_native, resample_native_signals
from sleepwalker.datasets.EDFCache import EDFCache
from sleepwalker.datasets.normalizer import Normalizer


def prepare_tensor_sample(
    data,
    quality_data=None,
    target=None,
    target_extra=None,
    patient=None,
    time=None,
    *,
    max_nan_fraction: float | None = None,
    quality_max_mean: Mapping[str, float] | None = None,
    valid_ranges: Mapping[str, Sequence[float]] | None = None,
    max_out_of_range_fraction: float = 0.05,
    min_std: Mapping[str, float] | None = None,
    **item,
):
    """Build the standard tensor sample with optional signal-quality checks."""
    if max_nan_fraction is not None and float(data.isna().mean().mean()) > float(max_nan_fraction):
        return None

    for channel, maximum in dict(quality_max_mean or {}).items():
        if quality_data is not None and channel in quality_data.columns and float(quality_data[channel].astype(float).mean()) > float(maximum):
            return None

    for channel, bounds in dict(valid_ranges or {}).items():
        if channel not in data.columns:
            continue
        if len(bounds) != 2:
            raise ValueError(f"valid_ranges[{channel!r}] must contain [minimum, maximum].")
        lower, upper = float(bounds[0]), float(bounds[1])
        invalid = (data[channel] < lower) | (data[channel] > upper)
        if float(invalid.mean()) > float(max_out_of_range_fraction):
            return None

    for channel, minimum in dict(min_std or {}).items():
        if channel in data.columns and float(data[channel].std()) < float(minimum):
            return None

    result = {"data": torch.from_numpy(data.values).float(), "patient": patient, "time": time, **item}
    if target is not None:
        result["target"] = target
    if target_extra is not None:
        result["target_extra"] = target_extra
    return result

@dataclass
class ChannelConfig:
    """Describe one logical model input and its physical EDF alternatives.

    Args:
        logical_name: Stable channel name exposed to models and callbacks.
        physical_names: Alternative channel names accepted from an EDF file.
        normalizer: One normalizer shared by every physical alternative, or a
            mapping from physical name to a channel-specific normalizer.
        quality_name: One companion quality channel shared by every physical
            alternative, or a mapping from physical name to its companion.
        unit: Canonical physical unit delivered to normalizers, callbacks, and
            the model. Compatible EDF units are converted automatically.
    """

    logical_name: str
    physical_names: Sequence[str]
    normalizer: Optional[Normalizer | Mapping[str, Optional[Normalizer]]] = None
    quality_name: Optional[str | Mapping[str, Optional[str]]] = None
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.physical_names, (str, bytes)):
            raise TypeError("physical_names must be a sequence of channel names, not a string.")
        self.physical_names = list(self.physical_names)
        if not self.logical_name:
            raise ValueError("logical_name must not be empty.")
        if not self.physical_names:
            raise ValueError(f"Channel '{self.logical_name}' requires physical_names.")
        if len(set(self.physical_names)) != len(self.physical_names):
            raise ValueError(f"Channel '{self.logical_name}' contains duplicate physical names.")
        physical_names = set(self.physical_names)
        for field_name, value in (("normalizer", self.normalizer), ("quality_name", self.quality_name)):
            if isinstance(value, Mapping):
                unknown = sorted(set(value) - physical_names)
                if unknown:
                    raise ValueError(
                        f"Channel '{self.logical_name}' has {field_name} entries for unknown "
                        f"physical channels: {unknown}."
                    )

    def normalizer_for(self, physical_name: str) -> Optional[Normalizer]:
        """Return the normalizer configured for one physical alternative."""
        if isinstance(self.normalizer, Mapping):
            return self.normalizer.get(physical_name)
        return self.normalizer

    def quality_name_for(self, physical_name: str) -> Optional[str]:
        """Return the quality channel configured for one physical alternative."""
        if isinstance(self.quality_name, Mapping):
            return self.quality_name.get(physical_name)
        return self.quality_name


_UNIT_DEFINITIONS = {
    "v": ("voltage", 1.0),
    "mv": ("voltage", 1e-3),
    "uv": ("voltage", 1e-6),
    "v/s": ("voltage_rate", 1.0),
    "mv/s": ("voltage_rate", 1e-3),
    "uv/s": ("voltage_rate", 1e-6),
    "%": ("percentage", 1.0),
    "percent": ("percentage", 1.0),
    "1": ("dimensionless", 1.0),
    "dimensionless": ("dimensionless", 1.0),
}


def _normalize_unit(unit: str) -> str:
    return unit.strip().replace("µ", "u").replace("μ", "u").lower()


def unit_conversion_factor(
    source_unit: Optional[str],
    target_unit: str,
    *,
    assume_if_missing: bool,
) -> float:
    """Return the multiplier from an EDF physical unit to ``target_unit``."""
    source = "" if source_unit is None else _normalize_unit(source_unit)
    target = _normalize_unit(target_unit)
    if target not in _UNIT_DEFINITIONS:
        raise ValueError(f"Unsupported target channel unit '{target_unit}'.")
    if not source:
        if assume_if_missing:
            return 1.0
        raise ValueError(
            f"EDF channel has no unit metadata; expected '{target_unit}'. "
            "Set assume_units_if_missing=True only when the stored values are "
            "known to already use the expected unit."
        )
    if source not in _UNIT_DEFINITIONS:
        raise ValueError(f"Unsupported EDF channel unit '{source_unit}'.")
    source_kind, source_scale = _UNIT_DEFINITIONS[source]
    target_kind, target_scale = _UNIT_DEFINITIONS[target]
    if source_kind != target_kind:
        raise ValueError(
            f"Incompatible channel units '{source_unit}' and '{target_unit}'."
        )
    return source_scale / target_scale


def apply_normalizers(data_df: pd.DataFrame, normalizers: Optional[Mapping[str, Normalizer]]) -> pd.DataFrame:
    """Apply configured channel normalizers in place and return the frame."""
    if normalizers:
        for col, normalizer in normalizers.items():
            if col in data_df.columns:
                values = data_df[col].to_numpy(dtype=float).reshape(-1, 1)
                data_df[col] = normalizer.transform(values).ravel()
    return data_df

@dataclass
class EDFFile: 
    """Prepared patient descriptor used after dataset initialization.

    The object stores lightweight metadata plus lazily queried label indices.
    Signal values are normally re-read from disk in :meth:`get_x` rather than
    kept resident after initialization.
    """

    channels: List[str]
    path: str
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp] = None
    start_offsets: Optional[np.ndarray] = None
    handle: Optional[pyedflib.EdfReader] = None
    length: int = 0
    X: Optional[pd.DataFrame] = None
    classes: Optional[Set[str]] = None
    labels: Optional[EventIndex] = None
    labels_extra: Optional[EventIndex] = None
    normalizers: Optional[dict[str, Normalizer]] = None
    unit_factors: Optional[dict[str, float]] = None
    z_statistics: Optional[dict[str, tuple[float, float]]] = None

    def get_x(self, start_date: pd.Timestamp, end_date: pd.Timestamp, sample_frequency: float, resample_type: str, channels: Optional[Sequence[str]] = None, cache: Optional[EDFCache] = None) -> pd.DataFrame:
        """Load and resample signals for one contiguous patient interval.

        Args:
            start_date: Inclusive interval start.
            end_date: Exclusive interval end.
            channels: Prepared physical channels to load. Defaults to all
                channels prepared for this file.

            sample_frequency: Target sampling frequency in Hz.
            resample_type: Resampling method understood by
                :func:`resample_native_signals`.
            cache: Optional process-shared complete-recording cache.
        """
        requested_channels = self.channels if channels is None else list(channels)
        unknown_channels = sorted(set(requested_channels) - set(self.channels))
        if unknown_channels:
            raise ValueError(f"Requested channels not prepared for {self.path}: {unknown_channels}.")
        def load(interval_start, interval_end):
            if self.X is None:
                native = read_edf_native(self.path, requested_channels, interval_start, interval_end, verbose=True)
            else:
                if len(self.X.index) < 2:
                    raise ValueError("A preloaded EDFFile requires at least two samples to infer its native frequency.")
                periods = np.diff(self.X.index.asi8)
                if not np.all(periods == periods[0]):
                    raise ValueError("A preloaded EDFFile must have a uniform sampling frequency.")
                frequency = 1e9 / float(periods[0])
                native = {frequency: self.X.loc[interval_start:interval_end, requested_channels].copy()}
            return resample_native_signals(native, requested_channels, interval_start, interval_end, sample_frequency, resample_type)

        if cache is None:
            return load(start_date, end_date)
        if self.end_date is None:
            raise ValueError(f"Cached EDF access requires an end timestamp for {self.path}.")
        key = (str(self.path), tuple(requested_channels), int(self.start_date.value), int(self.end_date.value), float(sample_frequency), str(resample_type))
        with cache.acquire(key, str(self.path), lambda: load(self.start_date, self.end_date)) as entry:
            return entry.copy_window(start_date, end_date, requested_channels)

    def apply_unit_conversion(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Convert loaded physical channels to their configured units in place."""
        for column, factor in dict(self.unit_factors or {}).items():
            if column in data_df.columns and factor != 1.0:
                data_df[column] = data_df[column] * factor
        return data_df

    def apply_z_normalization(self, data_df: pd.DataFrame, channels: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Apply fixed full-recording z-score statistics in place."""
        if self.z_statistics is None:
            return data_df
        required_channels = list(self.z_statistics) if channels is None else list(channels)
        if not required_channels:
            return data_df
        missing = sorted(set(required_channels) - set(data_df.columns))
        if missing:
            raise ValueError(f"Missing channels required for recording z-normalization: {missing}.")
        missing_statistics = sorted(set(required_channels) - set(self.z_statistics))
        if missing_statistics:
            raise ValueError(f"Missing recording z-normalization statistics for channels {missing_statistics}.")
        means = np.asarray([self.z_statistics[channel][0] for channel in required_channels])
        standard_deviations = np.asarray([self.z_statistics[channel][1] for channel in required_channels])
        column_indices = [int(data_df.columns.get_loc(channel)) for channel in required_channels]
        values = data_df.to_numpy(copy=False)
        contiguous = column_indices == list(range(column_indices[0], column_indices[0] + len(column_indices)))
        shares_data = np.shares_memory(values, data_df[required_channels[0]].to_numpy(copy=False))
        if contiguous and shares_data:
            selected = values[:, column_indices[0]:column_indices[0] + len(column_indices)]
            selected -= means
            selected /= standard_deviations
        else:
            for channel, mean, standard_deviation in zip(required_channels, means, standard_deviations):
                data_df[channel] = (data_df[channel] - mean) / standard_deviation
        return data_df
    
    def get_y_extra(self, start_date: pd.Timestamp, end_date: pd.Timestamp, sample_frequency, classes):
        """Sample the optional secondary target timeline for one window."""
        if self.labels_extra:
            freq = pd.to_timedelta(1.0 / sample_frequency, unit="s")
            return self.labels_extra.query(start_date, end_date, freq=freq, sparse=False, labels=classes)
        else:
            return None

    def get_y(self, start_date: pd.Timestamp, end_date: pd.Timestamp, sample_frequency, classes):
        """Sample the primary target timeline for one window."""
        if self.labels:
            freq = pd.to_timedelta(1.0 / sample_frequency, unit="s")
            return self.labels.query(start_date, end_date, freq=freq, sparse=False, labels=classes)
        else:
            return None

def batch_collate(batch, ignore_list = ["time", "patient"]):
    """Stack tensor-like batch fields while preserving metadata lists.

    Args:
        batch: Sequence of per-item dictionaries returned by the dataset.
        ignore_list: Keys that should stay as Python lists instead of being
            stacked with ``torch.stack``.

    Returns:
        A dictionary with one entry per observed key. Tensor-valued fields are
        stacked, while metadata such as ``time`` and ``patient`` remains a
        list.
    """
    valid_samples = [sample for sample in batch if sample is not None]
    if len(valid_samples) == 0:
        return None

    final_dict = defaultdict(list)

    for sample in valid_samples:
        if not isinstance(sample, Mapping):
            raise TypeError(f"Dataset samples must be mappings or None, got {type(sample).__name__}.")
        for key, value in sample.items():
            final_dict[key].append(value)
    
    collated = {}
    for key, values in final_dict.items():
        if key in ignore_list:
            collated[key] = values
        elif all(isinstance(value, Mapping) for value in values):
            keys = set(values[0])
            if any(set(value) != keys for value in values[1:]):
                raise ValueError(f"Cannot collate mapping key '{key}' with inconsistent nested keys.")
            collated[key] = {
                nested_key: torch.stack([value[nested_key] for value in values])
                for nested_key in values[0]
            }
        elif all(isinstance(value, torch.Tensor) for value in values):
            collated[key] = torch.stack(values)
        elif all(isinstance(value, numbers.Number) for value in values):
            collated[key] = torch.as_tensor(values)
        else:
            raise TypeError(
                f"Cannot collate key '{key}' with values of type "
                f"{sorted({type(value).__name__ for value in values})}."
            )
    return collated

def stack_repeated_views(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(views) == 0:
        raise ValueError("Cannot stack an empty repeated-view sequence.")
    keys = set(views[0])
    if "data" not in keys:
        raise ValueError("Repeated views require a 'data' field.")
    if any(set(view) != keys for view in views[1:]):
        raise ValueError("Repeated views returned inconsistent fields.")

    view_data = [view["data"] for view in views]
    if not isinstance(view_data[0], (torch.Tensor, Mapping)):
        raise TypeError(f"Repeated views require tensor or mapping-valued data, got {type(view_data[0]).__name__}.")

    result = dict(views[0])
    result["data"] = default_collate(view_data)
    for key in keys - {"data"}:
        first = views[0][key]
        values = [view[key] for view in views[1:]]
        if isinstance(first, torch.Tensor):
            matches = all(isinstance(value, torch.Tensor) and torch.equal(first, value) for value in values)
        elif isinstance(first, np.ndarray):
            matches = all(isinstance(value, np.ndarray) and np.array_equal(first, value) for value in values)
        else:
            matches = all(first == value for value in values)
        if not matches:
            raise ValueError(f"Repeated views disagree on fixed field '{key}'.")
    return result

class EventIndex:
    """
    Fast indicator sampling for labeled time intervals.
    Semantics: event is active at t iff start <= t < end  (left-closed, right-open).
    All timestamps should be tz-aligned (ideally UTC).
    """
    def __init__(self, events: pd.DataFrame): #merge_overlaps: bool = True
        # expected columns: starttime, endtime, label
        df = events.copy()

        # self.labels: list = labels #sorted(df["Label"].unique())
        self.starts: Dict[str, np.ndarray] = {}
        self.ends:   Dict[str, np.ndarray] = {}

        for lab, g in df.groupby("Label", sort=False):
            s = g["Starttime"].astype("int64").to_numpy()
            e = g["Endtime"].astype("int64").to_numpy()
            # drop empty/invalid intervals
            keep = e > s
            s, e = s[keep], e[keep]

            # sort and (optionally) merge overlaps to shrink index size
            order = np.argsort(s, kind="mergesort")
            s, e = s[order], e[order]
            # if merge_overlaps and s.size:
            #     s, e = self._merge_overlaps(s, e)

            self.starts[lab] = s
            self.ends[lab]   = e  # already sorted after merge

    def query(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        freq: str|pd.Timedelta,
        dtype: str = "int8",
        sparse: bool = False,
        labels: list = []
    ) -> pd.DataFrame:
        """Sample active labels on a regular time grid.

        Args:
            start: Window start timestamp.
            end: Window end timestamp.
            freq: Sampling frequency expressed as a pandas offset or timedelta.
            dtype: Output dtype for dense arrays.
            sparse: Whether to use pandas sparse arrays for the result.
            labels: Labels to materialize as output columns.

        Returns:
            A time-indexed DataFrame with one column per requested label. Each
            value is ``1`` when any interval with that label is active at the
            sample time and ``0`` otherwise.

        Notes:
            Interval semantics are left-closed and right-open:
            ``start <= t < end``.
        """
        # left-closed, right-open grid to match interval semantics
        idx = pd.date_range(start, end, freq=freq, inclusive="left")
        t = idx.astype("int64").to_numpy()  # ns since epoch

        out = {}
        for lab in labels:
            s = self.starts.get(lab, np.empty(0, np.int64))
            e = self.ends.get(lab,   np.empty(0, np.int64))
            # count how many have started minus how many have ended → active count
            # active at t if start <= t < end  → use right for starts, left for ends
            started = np.searchsorted(s, t, side="right")
            ended   = np.searchsorted(e, t, side="left")
            active  = (started - ended) > 0
            out[lab] = active.astype(dtype)

        df = pd.DataFrame(out, index=idx)
        if sparse:
            # drastically cuts memory when events are rare
            for c in df.columns:
                df[c] = pd.arrays.SparseArray(df[c], dtype=dtype)
        return df

class BaseDataset(Dataset, ABC):
    """
    Base class for EDF-backed datasets that turn raw patient recordings into
    model-ready training items.

    In practice, a `BaseDataset` does three jobs for you:

    1. load and resample signals from EDF files
    2. load and map raw event labels to task labels
    3. expose sliding-window items that contain at least `data`, `target`,
       `patient`, and `time`

    The class is designed for research workflows where different experiments
    need slightly different filtering or target-building logic without having to
    reimplement EDF loading every time.

    Typical usage
    -------------
    1. Configure a dataset object with channels, label mapping, and optional
       preparation callbacks.
    2. Optionally inspect patients with `get_patient_stats(...)` if you want to
       filter patients before training.
    3. Call `initialize(patients, num_workers)` once you know which patients to
       keep.
    4. Use the dataset with a `DataLoader`.

    The dataset is intentionally **not initialized automatically**. This allows
    you to:
    - inspect/filter patients before the expensive sliding-window index is built
    - reuse one configured dataset object for both patient statistics and final training

    Loading model and performance
    -----------------------------
    `BaseDataset` is designed for large EDF collections. It does **not** keep
    full patient signals in memory after initialization.

    Instead, the workflow is:
    - `initialize(...)` reads headers and annotations, then builds an index of
      valid sliding-window positions without loading complete signals unless
      `z_normalize=True`
    - `__getitem__` uses that index to choose a candidate item
    - the corresponding EDF signal window is loaded lazily only when needed

    This means you can work with very large sets of EDF files without large
    memory overhead. The main bottleneck is usually disk / network I/O and EDF
    parsing, not RAM usage.

    Item retrieval fundamentally uses rejection sampling:
    - a candidate window is selected from the precomputed index
    - `prepare_target` may reject it before signal loading
    - `prepare_sample` may reject it after signal loading
    - the configured rejection strategy may try fallback candidates
    - exhausted expected rejection returns ``None``; construction errors raise

    In practice, performance depends strongly on where you place filtering:
    - filtering in `prepare_target` is cheap and usually preferable
    - filtering in `prepare_sample` is more expensive because signal I/O has
      already happened
    - `prepare_patient` is annotation-only; use
      `get_patient_stats(...)` for explicit full-signal cohort analysis

    Sparse window indexing
    ----------------------
    After `prepare_patient` returns its possibly filtered `label_df`,
    `BaseDataset` no longer assumes that valid windows form one continuous
    `[start, end]` region. Instead, it scans the remaining label intervals for
    timestamp discontinuities and builds a sparse per-patient index of valid
    stride offsets.

    This means user code can trim labels down to several disjoint retained
    regions and the dataset will sample only windows fully contained in those
    connected components, instead of densely sampling the gaps in between.

    Preparation hooks
    -----------------
    There are three optional hooks. They are ordered from expensive to cheap to
    help you place logic in the right stage.

    `prepare_patient`
        Runs once per patient after metadata and mapped labels have been loaded.

        Use this for whole-patient logic such as:
        - trimming leading/trailing wake
        - rejecting patients with too few valid labels

        Use this hook for annotation-wide logic. Signal-wide cohort logic
        belongs in `get_patient_stats(...)` so ordinary indexing stays cheap.

    `prepare_target`
        Runs once per candidate item before the signal window is loaded. It
        receives the target interval labels and should either:
        - return `None` to reject the item cheaply, or
        - return a dictionary with the prepared target payload

        Use this for:
        - cheap label-based filtering
        - multiclass / multilabel target construction
        - class-balancing metadata

        If your logic can be expressed here, this is usually the best place for
        it because no signal window has to be loaded yet.

    `prepare_sample`
        Runs after the signal window has been loaded, re-referenced and resized.
        It should return the final sample dictionary or `None` to reject the
        sample.

        Use this for:
        - converting the signal DataFrame to tensors
        - signal-dependent rejection, e.g. flatline or NaN-heavy windows
        - attaching final `data` / `target` outputs

        This is more expensive than `prepare_target` because the signal window
        has already been loaded.

    Which hook should I use?
    ------------------------
    Use `prepare_patient` when the decision depends on the full annotation set.

    Example:
    - trim wake at the start/end of the night
    - reject patients with almost no sleep left after trimming

    Use `prepare_target` when the decision depends only on labels or timestamps
    for one candidate item.

    Example:
    - reject windows outside sleep
    - build one-hot multiclass targets

    Use `prepare_sample` when the decision depends on the loaded signal window.

    Example:
    - reject windows with too many NaNs
    - convert signal frames to tensors

    If you want to compare patients to each other, for example “drop the
    shortest sleepers” or “keep only the top-K by REM time”, use
    `get_patient_stats(...)` before calling `initialize(...)`.

    Parameters
    ----------
    channels
        Sequence of `ChannelConfig` objects describing logical model inputs and
        their accepted physical EDF alternatives. One available alternative is
        sampled per logical input when `group_sampling_strategy == 'random'`;
        all available alternatives are returned when it is `'none'`.

    sample_frequency
        Target sampling frequency in Hz used when loading signal windows and
        querying label timelines.

    resample_type
        Resampling mode passed to `edf_to_df(...)`. This controls how signals
        are aligned to `sample_frequency`. `"nearest"` is the default;
        `"polyphase"` applies an antialiasing filter for model inputs.

    total_input
        Length of the signal window returned for each item. Accepts values such
        as `"30s"`, `"5min"`, or a `pd.Timedelta`.

    stride
        Time between consecutive sampled windows. Defaults to `total_input`.
        Set `stride < total_input` to sample overlapping windows.

    event_mapping
        Mapping from raw dataset-specific event labels to the labels used by
        your task, e.g. `{"Sleep stage W": "wake"}`.

    remove_unmapped_events
        If `True`, labels that are not found in `event_mapping` are dropped. If
        `False`, unmapped labels are kept as-is.

    prepare_patient
        Optional callback for whole-patient annotation preparation. It receives
        `label_df`, `label_extra_df`, and `patient`.
        It should return `(label_df, label_extra_df)` or `None`.

    prepare_target
        Optional callback for per-item target preparation. It receives
        `target`, `target_extra`, `patient`, and `time`. It should return a
        dictionary or `None`.

    prepare_sample
        Optional callback for final sample preparation. It receives the loaded
        signal window as `data` plus the current item fields. It should return
        the final item dictionary or `None`.

    online_max_tries
        Retry budget used by the configured rejection strategy. The
        ``patient_then_global`` strategy gives this many attempts in each
        phase, counting the requested window as the first patient attempt.

    rejection_strategy
        ``none`` tries only the requested candidate. ``patient`` and ``global``
        sample fallbacks from the corresponding range. ``patient_then_global``
        preserves the stochastic training behavior of trying both phases.

    n_views
        Number of independently prepared views returned for one accepted
        candidate. Only the ``data`` field is stacked; targets and metadata
        must agree across views.

    force_one_day
        If `True`, reject patients whose mapped labels appear to span more than
        one day. This catches some common loading/pathology issues.

    rereference
        Optional list of channel groups that should be average-referenced after
        the signal window is loaded.

        Example:
        `rereference=[["C3", "C4"]]`
        means both channels are replaced by their values minus the mean of
        `C3`/`C4` at each time step.

    z_normalize
        If `True`, load every complete recording during initialization and
        compute a separate mean and standard deviation for each available
        physical signal. Configured channel normalizers run first,
        rereferencing runs second, and recording z-normalization is always the
        final built-in signal normalization step. The resulting scalar
        statistics are stored on the prepared `EDFFile`; complete signals are
        not retained in memory. This option makes initialization substantially
        more expensive.

    assume_units_if_missing
        If `False`, a configured channel unit requires unit metadata in the EDF
        header. If `True`, missing metadata is treated as already matching the
        configured unit. Explicitly incompatible units are always rejected.

    edf_unit_overrides
        Explicit corrections for EDF headers whose unit labels are known to be
        wrong for a particular dataset release. Corrections are applied before
        conversion and become part of the stored expert contract.

    Examples
    --------
    Trim wake once per patient:

    ```python
    def prepare_patient(label_df, label_extra_df, patient=None):
        trimmed = trim_event(label_df, label_extra_df)
        if trimmed is None:
            return None
        return trimmed
    ```

    Build a multiclass target and reject low-sleep windows cheaply:

    ```python
    from functools import partial
    from sleepwalker.trainer.utils.targets import prepare_multiclass_target

    prepare_target = partial(
        prepare_multiclass_target,
        target_classes=["wake", "n1", "n2", "n3", "rem"],
        filters=[{"columns": ["n1", "n2", "n3", "rem"], "percentage": 0.5, "mode": "min"}],
    )
    ```

    Convert signals to tensors and reject bad windows:

    ```python
    def prepare_sample(data, target, target_extra=None, patient=None, time=None):
        if data.isna().mean().mean() > 0.1:
            return None
        item = {
            "data": torch.from_numpy(data.values).float(),
            "target": target,
            "patient": patient,
            "time": time,
        }
        if target_extra is not None:
            item["target_extra"] = target_extra
        return item
    ```
    """
    def __init__(
        self,
        *,
        channels: Sequence[ChannelConfig],
        sample_frequency: float,
        resample_type: str = "nearest",
        total_input: str | pd.Timedelta = "30s",
        stride: Optional[str | pd.Timedelta] = None,
        event_mapping: Optional[Mapping[str, str]] = None, 
        remove_unmapped_events : bool = True, 
        prepare_patient: Optional[Callable[..., Optional[tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]]] = None,
        prepare_target: Optional[Callable] = None,
        prepare_sample: Callable = prepare_tensor_sample,
        online_max_tries:int = 16,
        rejection_strategy: str = "patient_then_global",
        n_views: int = 1,
        force_one_day: bool = True,
        rereference: Optional[List[List[str]]] = None, # [ ["C3-A1", "C4-A2"] ]
        z_normalize: bool = False,
        group_sampling_strategy: Optional[str] = 'random', # [random, None]
        assume_units_if_missing: bool = False,
        edf_unit_overrides: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__()
        
        # Config
        self.channels = list(channels)
        self.event_mapping = event_mapping
        self.remove_unmapped_events = remove_unmapped_events
        self.sample_frequency = sample_frequency
        self.resample_type = resample_type
        self.total_input = pd.to_timedelta(total_input)
        if self.total_input <= pd.Timedelta(0):
            raise ValueError("total_input must be positive.")
        self.stride = pd.to_timedelta(stride) if stride is not None else self.total_input
        if self.stride <= pd.Timedelta(0):
            raise ValueError("stride must be positive.")
        if self.stride > self.total_input:
            raise ValueError("stride must be smaller than or equal to total_input.")
        self.prepare_target_callback = prepare_target
        self.prepare_sample_callback = prepare_sample
        self.prepare_patient_callback = prepare_patient
        self.all_patients: list[str | os.PathLike] = []
        self.online_max_tries = int(online_max_tries)
        if self.online_max_tries < 0:
            raise ValueError("online_max_tries must not be negative.")
        self.rejection_strategy = str(rejection_strategy)
        if self.rejection_strategy not in {"none", "patient", "global", "patient_then_global"}:
            raise ValueError("rejection_strategy must be 'none', 'patient', 'global', or 'patient_then_global'.")
        self.n_views = int(n_views)
        if self.n_views < 1:
            raise ValueError("n_views must be positive.")
        self.initialized = False
        self.force_one_day = force_one_day
        self.rereference = rereference
        self.z_normalize = bool(z_normalize)
        if self.z_normalize and self.rereference:
            logger.warning("z_normalize=True with rereferencing enabled: recording z-normalization is applied after rereferencing.")
        self.assume_units_if_missing = bool(assume_units_if_missing)
        self.edf_unit_overrides = dict(edf_unit_overrides or {})
        self.edf_files: list[EDFFile] = []
        self.lower_bounds: list[int] = []
        self.upper_bounds: list[int] = []
        self.channel_configs_by_logical_name: Dict[str, ChannelConfig] = {}
        physical_channels: dict[str, list[ChannelConfig]] = defaultdict(list)
        for cfg in self.channels:
            if cfg.logical_name in self.channel_configs_by_logical_name:
                raise ValueError(
                    f"Duplicate logical channel name '{cfg.logical_name}'."
                )
            for physical_name in cfg.physical_names:
                physical_channels[physical_name].append(cfg)
            self.channel_configs_by_logical_name[cfg.logical_name] = cfg

        self.shared_physical_channels = {name for name, configs in physical_channels.items() if len(configs) > 1}
        for physical_name in self.shared_physical_channels:
            units = {None if cfg.unit is None else _normalize_unit(cfg.unit) for cfg in physical_channels[physical_name]}
            if len(units) > 1:
                raise ValueError(f"Shared physical channel '{physical_name}' has incompatible target units: {sorted(map(str, units))}.")
        if self.shared_physical_channels and self.rereference:
            raise ValueError("Shared physical channels cannot be combined with rereferencing.")
        self.group_sampling_strategy = group_sampling_strategy
        self.edf_cache: Optional[EDFCache] = None

        # Events/classes
        if event_mapping is not None:
            self.event_mapping = {k: v for k, v in event_mapping.items()}
            self.classes = sorted(list(set(self.event_mapping.values())))
            self.label_classes = list(self.classes)
        else:
            self.event_mapping = None
            self.classes = []
            self.label_classes = []

        if self.event_mapping is not None and self.remove_unmapped_events and len(self.event_mapping) == 0:
            logger.warning(f"You set remove_unmapped_events to true but provided an empty mapping. If you want to not extract any labels, set event_mapping to None. If you want to extract all labels, set event_mapping to an empty dictionary and remove_unmapped_events to false.")

    @abstractmethod
    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Return raw event intervals for one patient recording.

        Args:
            edf_path: Path to the patient EDF file.
            start_datetime: Recording start time inferred from EDF metadata.

        Returns:
            A DataFrame containing at least ``Starttime``, ``Endtime``, and
            ``Label`` columns.

        Notes:
            Concrete dataset adapters must implement this method.
        """
        ...

    def get_extra_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Return optional secondary event intervals.

        Notes:
            The default implementation signals that extra targets are not
            supported for this dataset.
        """
        raise ValueError(f"This function should not be called")

    def has_extra_target(self) -> bool:
        """Report whether this dataset exposes a secondary target timeline."""
        return False

    def get_classes(self) -> list[str]:
        """Return the canonical class labels currently known to the dataset."""
        return self.classes

    def get_timeseries_len(self) -> int:
        """Return the expected number of signal samples per item."""
        return int(round(self.total_input.total_seconds() * float(self.sample_frequency)))

    def get_n_patients(self) -> int:
        """Return the number of prepared patient recordings."""
        return len(self.edf_files)

    def set_edf_cache(self, cache: Optional[EDFCache]) -> None:
        """Use one loading-session cache for complete resampled recordings."""
        self.edf_cache = cache

    def set_rejection_strategy(self, strategy: str) -> None:
        strategy = str(strategy)
        if strategy not in {"none", "patient", "global", "patient_then_global"}:
            raise ValueError("rejection strategy must be 'none', 'patient', 'global', or 'patient_then_global'.")
        self.rejection_strategy = strategy

    def set_n_views(self, n_views: int) -> None:
        n_views = int(n_views)
        if n_views < 1:
            raise ValueError("n_views must be positive.")
        self.n_views = n_views

    def get_patient_ranges(self) -> list[tuple[int, int]]:
        """Return each prepared patient's half-open range in the global index."""
        if not self.initialized:
            raise ValueError(f"{self.__class__.__name__} is not initialized. Call initialize(...) before requesting patient ranges.")
        return list(zip(self.lower_bounds, self.upper_bounds))

    def get_input_channels(self) -> list[str]:
        """Return the logical model input channel names."""
        return list(self.channel_configs_by_logical_name)

    def apply_rereference(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured average-reference groups in place."""
        if self.rereference:
            for reference_channels in self.rereference:
                available = [channel for channel in reference_channels if channel in data_df.columns]
                if available:
                    data_df[available] = data_df[available].values - data_df[available].values.mean(axis=1)[:, None]
        return data_df

    def calculate_z_statistics(self, data_df: pd.DataFrame, channels: Sequence[str]) -> dict[str, tuple[float, float]]:
        """Calculate finite full-recording mean/std pairs for physical channels."""
        statistics = {}
        for channel in channels:
            values = data_df[channel].to_numpy(dtype=float)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"Cannot z-normalize channel '{channel}' with empty or non-finite recording data.")
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values))
            if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
                raise ValueError(f"Cannot z-normalize channel '{channel}' with mean={mean} and std={standard_deviation}.")
            statistics[channel] = (mean, standard_deviation)
        return statistics

    def __len__(self):
        return sum([f.length for f in self.edf_files])

    def _build_connected_segments(self, label_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Collapse filtered labels into connected timestamp segments.

        The function treats intervals as connected when they overlap in time.
        It intentionally does not merge merely nearby intervals with a gap.
        """
        if len(label_df) == 0:
            return []

        df = label_df.sort_values(["Starttime", "Endtime"]).reset_index(drop=True)

        segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        seg_start = cast(pd.Timestamp, df.iloc[0]["Starttime"])
        seg_end = cast(pd.Timestamp, df.iloc[0]["Endtime"])

        for i in range(1, len(df)):
            row_start = cast(pd.Timestamp, df.iloc[i]["Starttime"])
            row_end = cast(pd.Timestamp, df.iloc[i]["Endtime"])
            if row_start <= seg_end:
                seg_end = max(seg_end, row_end)
            else:
                segments.append((seg_start, seg_end))
                seg_start = row_start
                seg_end = row_end

        segments.append((seg_start, seg_end))
        return segments

    def _segment_to_offsets(
        self,
        base_start: pd.Timestamp,
        seg_start: pd.Timestamp,
        seg_end: pd.Timestamp,
    ) -> np.ndarray:
        """Convert one connected time segment into candidate stride offsets.

        The filtered ``label_df`` defines where annotations may be available.
        Keep input windows that overlap a retained segment. The target callback
        later decides which part of those raw annotations is supervised.
        """
        earliest_valid_start = seg_start - self.total_input
        latest_valid_start = seg_end
        if latest_valid_start < earliest_valid_start:
            return np.empty(0, dtype=np.int64)

        first_offset = max(0, int(np.ceil((earliest_valid_start - base_start) / self.stride)))
        last_offset = int(np.floor((latest_valid_start - base_start) / self.stride))
        if last_offset < first_offset:
            return np.empty(0, dtype=np.int64)

        return np.arange(first_offset, last_offset + 1, dtype=np.int64)

    def _build_start_offsets(
        self,
        base_start: pd.Timestamp,
        label_df: pd.DataFrame,
    ) -> np.ndarray:
        """Build sparse valid window starts from the filtered patient labels."""
        segments = self._build_connected_segments(label_df)
        offsets = [
            self._segment_to_offsets(base_start=base_start, seg_start=seg_start, seg_end=seg_end)
            for seg_start, seg_end in segments
        ]
        offsets = [arr for arr in offsets if len(arr) > 0]
        if len(offsets) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(offsets)

    def _prepare_patient_artifacts(self, edf_path, *, load_signal: bool = False):
        channel_names = []
        for cfg in self.channels:
            channel_names.extend(cfg.physical_names)
            channel_names.extend(
                quality_name
                for physical_name in cfg.physical_names
                if (quality_name := cfg.quality_name_for(physical_name)) is not None
            )
        channel_names = list(dict.fromkeys(channel_names))
        normalizers = {
            physical_name: copy.deepcopy(normalizer)
            for cfg in self.channels
            for physical_name in cfg.physical_names
            if physical_name not in self.shared_physical_channels
            if (normalizer := cfg.normalizer_for(physical_name)) is not None
        }

        meta = read_edf_meta(edf_path)
        available_channels = set(meta["signals"])
        missing_channels = [
            cfg.logical_name
            for cfg in self.channels
            if not any(name in available_channels for name in cfg.physical_names)
        ]
        if missing_channels:
            raise ValueError(
                f"Missing required logical channels {missing_channels}; "
                f"available EDF channels are {sorted(available_channels)}."
            )
        if meta.get("source") != "pyedflib" and any(cfg.unit is not None for cfg in self.channels):
            raise ValueError(
                "Unit-aware loading requires an EDF header readable by pyEDFlib; "
                "the MNE fallback does not expose model-input units reliably."
            )
        source_units = dict(meta.get("units", {}))
        source_units.update(
            {
                channel: unit
                for channel, unit in self.edf_unit_overrides.items()
                if channel in available_channels
            }
        )
        unit_factors = {
            physical_name: unit_conversion_factor(
                source_units.get(physical_name),
                cfg.unit,
                assume_if_missing=self.assume_units_if_missing,
            )
            for cfg in self.channels
            for physical_name in cfg.physical_names
            if physical_name in available_channels and cfg.unit is not None
        }

        channels = [channel for channel in channel_names if channel in available_channels]
        classes = set()
        extra_classes = set()
        start = meta["start"]
        end = meta["end"]
        if start is None or end is None:
            raise ValueError(f"EDF file {edf_path} has no recording timestamps.")

        data_df = None
        if load_signal:
            data_df = edf_to_df(edf_path, channels, start=None, end=None, frequency=self.sample_frequency, how=self.resample_type, verbose=True)
            if data_df is None or len(data_df) == 0:
                raise ValueError("Found empty EDF file")
            for col, factor in unit_factors.items():
                if col in data_df.columns and factor != 1.0:
                    data_df[col] = data_df[col] * factor
            start = max(data_df.index[0], start)
            end = min(data_df.index[-1], end)

        if self.event_mapping is not None:
            if self.has_extra_target():
                df = self.get_event_df(edf_path, start)
                df_additional = self.get_extra_event_df(edf_path, start)

                if self.remove_unmapped_events:
                    df_additional["Label"] = df_additional["Label"].apply(
                        lambda x: self.event_mapping[x] if x in self.event_mapping else None
                    )
                else:
                    df_additional["Label"] = df_additional["Label"].apply(
                        lambda x: self.event_mapping[x] if x in self.event_mapping else x
                    )
                df_additional = df_additional.dropna()
                extra_classes = set(df_additional["Label"].unique())
            else:
                df = self.get_event_df(edf_path, start)
                df_additional = None

            if self.remove_unmapped_events:
                df["Label"] = df["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else None)
            else:
                df["Label"] = df["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else x)
            df = df.dropna()

            if self.force_one_day and (len(df["Starttime"].dt.date.unique()) > 2 or len(df["Endtime"].dt.date.unique()) > 2):
                raise ValueError(
                    f"Edf file: {edf_path} appears to be longer than one entire day. Is this a loading error? If not, set force_one_day = False"
                )

            classes = set(df["Label"].unique())

            start = max(start, df["Starttime"].min())
            end = min(end, df["Endtime"].max())
        else:
            df = None
            df_additional = None

        if self.prepare_patient_callback is not None:
            prepared = self.prepare_patient_callback(
                label_df=df,
                label_extra_df=df_additional,
                patient=edf_path,
            )
            if prepared is None:
                return None
            df, df_additional = prepared
            if self.event_mapping is not None and (df is None or len(df) == 0):
                raise ValueError(f"Edf file: {edf_path} was filtered out in prepare_patient")

        if data_df is not None:
            start = max(start, data_df.index[0])
            end = min(end, data_df.index[-1])

        z_statistics = None
        if self.z_normalize:
            if data_df is None:
                raise ValueError("z_normalize=True requires complete recording data during patient preparation.")
            data_df = apply_normalizers(data_df, normalizers)
            data_df = self.apply_rereference(data_df)
            signal_channels = list(dict.fromkeys(
                physical_name
                for cfg in self.channels
                for physical_name in cfg.physical_names
                if physical_name in available_channels
            ))
            z_statistics = self.calculate_z_statistics(data_df, signal_channels)

        return {
            "path": edf_path,
            "data_df": data_df,
            "channels": channels,
            "start": start,
            "end": end,
            "label_df": df,
            "label_extra_df": df_additional,
            "classes": classes,
            "extra_classes": extra_classes,
            "normalizers": normalizers,
            "unit_factors": unit_factors,
            "z_statistics": z_statistics,
        }

    def prepare_patient(self, edf_path, *, raise_errors: bool = False) -> Optional[EDFFile]:
        """Prepare one patient recording for lazy window sampling.

        Args:
            edf_path: Path to an EDF file.

        Returns:
            An :class:`EDFFile` descriptor with fixed normalizers, event
            indices, and a precomputed window count, or ``None`` if patient
            preparation rejects the file.

        Raises:
            ValueError: If the file cannot produce at least one valid window or
                appears inconsistent with the configured assumptions.
        """
        started_at = time.perf_counter()
        slow_warning_seconds = 30
        try:
            artifacts = self._prepare_patient_artifacts(edf_path, load_signal=self.z_normalize)
            if artifacts is None:
                return None

            label_df = artifacts["label_df"]
            label_extra_df = artifacts["label_extra_df"]
            window_start = artifacts["start"]
            start_offsets = None
            if label_df is not None and len(label_df) > 0:
                start_offsets = self._build_start_offsets(
                    base_start=window_start,
                    label_df=label_df,
                )
                max_offset = int(np.floor((artifacts["end"] - self.total_input - window_start) / self.stride))
                start_offsets = start_offsets[start_offsets <= max_offset]
                n_items = len(start_offsets)
            else:
                n_items = int((artifacts["end"] - self.total_input - window_start) / self.stride)

            if n_items <= 0:
                raise ValueError(
                    f"Edf file: {edf_path} appears to be empty between {artifacts['start']} - {artifacts['end']} "
                    f"with a total signal length of {artifacts['end'] - artifacts['start']}s"
                )

            result = EDFFile(
                path=edf_path,
                X=None,
                channels=artifacts["channels"],
                end_date=artifacts["end"],
                start_offsets=start_offsets,
                length=n_items,
                labels=EventIndex(label_df) if label_df is not None else None,
                labels_extra=EventIndex(label_extra_df) if label_extra_df is not None else None,
                start_date=window_start,
                classes=artifacts["classes"].union(artifacts["extra_classes"]),
                normalizers=artifacts["normalizers"],
                unit_factors=artifacts["unit_factors"],
                z_statistics=artifacts["z_statistics"],
            )
            elapsed = time.perf_counter() - started_at
            if elapsed >= slow_warning_seconds:
                logger.warning(
                    f"Slow patient preparation: {edf_path} took {elapsed:.1f}s "
                    f"and produced {n_items} windows."
                )
            return result
        except Exception as e:
            elapsed = time.perf_counter() - started_at
            if raise_errors:
                raise ValueError(f"Cannot prepare EDF file {edf_path}: {e}") from e
            logger.warning(f"Cannot read edf file: {edf_path} after {elapsed:.1f}s due to {e}")

            return None #EDFFile(path=edf_path, classes=classes.union(extra_classes))

    def get_patient_stats(self, patients: Sequence[str | os.PathLike], summarize_patient: Callable, num_workers: int = 4) -> pd.DataFrame:
        """Compute per-patient summary rows before dataset initialization.

        Args:
            patients: Patient EDF paths to inspect.
            summarize_patient: Callback receiving ``patient``, ``data_df``,
                ``label_df``, and ``label_extra_df``. It should return a row
                dictionary or ``None``.
            num_workers: Worker count used for parallel summarization.

        Returns:
            A DataFrame built from the rows returned by ``summarize_patient``.

        Notes:
            This helper is intended for cohort-level filtering before
            :meth:`initialize`.
        """
        worker_count = num_workers
        rows: list[dict] = []
        patient_list = list(patients)

        logger.progress_start(len(patient_list), desc="Collecting patient stats", leave=True)
        if worker_count > 1:
            with multiprocessing.Pool(worker_count) as pool:
                iter_objects = pool.imap_unordered(partial(self._summarize_patient, summarize_patient=summarize_patient), patient_list)
                for result in iter_objects:
                    try:
                        if result is not None:
                            rows.append(result)
                    finally:
                        logger.progress_advance(1)
        else:
            for patient in patient_list:
                try:
                    result = self._summarize_patient(patient, summarize_patient)
                    if result is not None:
                        rows.append(result)
                finally:
                    logger.progress_advance(1)

        logger.progress_close()
        logger.info(f"Collected patient stats for {len(rows)}/{len(patient_list)} patients.")
        return pd.DataFrame(rows)

    def _summarize_patient(self, edf_path, summarize_patient: Callable):
        try:
            artifacts = self._prepare_patient_artifacts(edf_path, load_signal=True)
            if artifacts is None:
                return None
            row = summarize_patient(
                patient=edf_path,
                data_df=artifacts["data_df"],
                label_df=artifacts["label_df"],
                label_extra_df=artifacts["label_extra_df"],
            )
            if row is None:
                return None
            result = dict(row)
            result.setdefault("patient", edf_path)
            return result
        except Exception as exc:
            logger.warning(f"Failed to summarize patient {edf_path}: {exc}")
            return None

    def initialize(
        self,
        patients: Sequence[str | os.PathLike],
        num_workers: int = 4,
        *,
        strict: bool = False,
        multiprocessing_start_method: Optional[str] = None,
    ) -> None:
        """Prepare patient metadata and build the sliding-window index.

        Args:
            patients: EDF paths to include in the dataset.
            num_workers: Worker count used while calling
                :meth:`prepare_patient`.
            multiprocessing_start_method: Optional multiprocessing start
                method. ``PairedDataset`` uses ``forkserver`` so workers do
                not inherit previously initialized datasets from the parent.

        Notes:
            Initialization is intentionally explicit because it can be
            expensive. Training scripts often call :meth:`get_patient_stats`
            first to filter patients before paying the full initialization
            cost. When ``z_normalize=True``, initialization also loads every
            complete recording once to calculate its channel statistics.
        """
        self.all_patients = list(patients)
        self.initialized = False
        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        total_n_patients = len(patients)
        file_handles, lower_bounds, upper_bounds = [], [], []
        all_classes = set()
        lower, n_windows = 0, 0

        logger.progress_start(len(patients), desc="Preparing labels and sliding windows", leave=True)

        def collect(iter_objects, prepare_inline):
            nonlocal lower, n_windows, all_classes
            for edf in iter_objects:
                if prepare_inline:
                    edf = self.prepare_patient(edf, raise_errors=strict)

                if edf:
                    edf = cast(EDFFile, edf)
                    file_handles.append(edf)
                    lower_bounds.append(lower)
                    upper_bounds.append(lower + edf.length)
                    lower += edf.length
                    n_windows += edf.length

                    if edf.classes is not None:
                        all_classes |= set(edf.classes)

                logger.progress_advance(1)

        if num_workers > 1:
            context = multiprocessing if multiprocessing_start_method is None else multiprocessing.get_context(multiprocessing_start_method)
            with context.Pool(num_workers) as pool:
                collect(pool.imap_unordered(partial(self.prepare_patient, raise_errors=strict), patients), prepare_inline=False)
        else:
            collect(patients, prepare_inline=True)

        logger.progress_close()

        self.edf_files = file_handles
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if not self.event_mapping:
            self.classes = sorted(list(all_classes))
            self.label_classes = list(self.classes)

        logger.info(
            f"Dataset initialized with {len(self.edf_files)}/{total_n_patients} patients. "
            f"Total windows: {len(self)}. Classes: {len(self.classes)}"
        )
        self.initialized = True

    def select_channels(self, available_columns: Sequence[str]) -> list[tuple[ChannelConfig, str]]:
        """Select the physical alias used for each logical model input."""
        available_set = set(available_columns)
        selected_channels = []
        for cfg in self.channels:
            available = [name for name in cfg.physical_names if name in available_set]
            if len(available) == 0:
                raise ValueError(f"No physical channels found for logical channel '{cfg.logical_name}'.")
            if self.group_sampling_strategy in {'random', 'first'}:
                selected_name = available[int(np.random.choice(len(available)))] if self.group_sampling_strategy == 'random' else available[0]
                selected_channels.append((cfg, selected_name))
            elif self.group_sampling_strategy == 'none':
                selected_channels.extend((cfg, name) for name in available)
            else:
                raise NotImplementedError('Cannot sample for group_sampling_strategy', self.group_sampling_strategy)
        return selected_channels

    def channels_to_load(self, selected_channels: Sequence[tuple[ChannelConfig, str]], available_columns: Sequence[str]) -> list[str]:
        """Return selected signals plus quality and rereference dependencies."""
        available_set = set(available_columns)
        channels = [physical_name for cfg, physical_name in selected_channels]
        if self.group_sampling_strategy in {'random', 'first'}:
            channels.extend(
                quality_name
                for cfg, physical_name in selected_channels
                if (quality_name := cfg.quality_name_for(physical_name)) is not None
            )
        if self.rereference:
            channels.extend(channel for group in self.rereference for channel in group if channel in available_set)
        return list(dict.fromkeys(channels))

    def run_build_sample(self, item: Dict[str, Any], x_df: pd.DataFrame, selected_channels: Optional[Sequence[tuple[ChannelConfig, str]]] = None) -> Optional[Dict[str, Any]]:
        """Finalize one loaded signal window into a training or inference item."""
        quality_df = None
        if len(self.channels) > 0:
            selected_channels = self.select_channels(x_df.columns) if selected_channels is None else list(selected_channels)
            renamed_columns = []
            selected_quality = {}

            for cfg, selected_name in selected_channels:
                if selected_name not in x_df.columns:
                    raise ValueError(f"Selected physical channel '{selected_name}' was not loaded.")
                renamed_columns.append(cfg.logical_name)
                if self.group_sampling_strategy in {'random', 'first'}:
                    quality_name = cfg.quality_name_for(selected_name)
                    if quality_name is not None:
                        if quality_name not in x_df.columns:
                            raise ValueError(
                                f"Missing quality channel '{quality_name}' "
                                f"for selected channel '{selected_name}'."
                            )
                        selected_quality[cfg.logical_name] = x_df[quality_name].copy()

            selected_values = []
            for cfg, selected_name in selected_channels:
                values = x_df[selected_name].copy()
                normalizer = cfg.normalizer_for(selected_name) if selected_name in self.shared_physical_channels else None
                if normalizer is not None:
                    values[:] = normalizer.transform(values.to_numpy(dtype=float).reshape(-1, 1)).ravel()
                selected_values.append(values)
            x_selected = pd.concat(selected_values, axis=1)
            x_selected.columns = renamed_columns
            x_df = x_selected
            if len(selected_quality) > 0:
                quality_df = pd.DataFrame(selected_quality, index=x_df.index)

        return self.prepare_sample_callback(
            data=x_df,
            quality_data=quality_df,
            target=item.get("target"),
            target_extra=item.get("target_extra"),
            patient=item.get("patient"),
            time=item.get("time"),
            **{k: v for k, v in item.items() if k not in {"data", "target", "target_extra", "patient", "time"}},
        )

    def get_target_item(self, file: EDFFile, start_date: pd.Timestamp):
        """Build target and timestamp fields without loading signals."""
        end_date = start_date + self.total_input
        item: Dict[str, Any] = {"patient": file.path, "time": start_date}

        if file.labels:
            item["target"] = file.get_y(start_date, end_date, self.sample_frequency, self.label_classes)
            if file.labels_extra:
                item["target_extra"] = file.get_y_extra(start_date, end_date, self.sample_frequency, self.label_classes)

        if self.prepare_target_callback is not None:
            prepared_target = self.prepare_target_callback(target=item.get("target"), target_extra=item.get("target_extra"), patient=item.get("patient"), time=item.get("time"))
            if prepared_target is None:
                return None
            if not isinstance(prepared_target, dict):
                raise ValueError(f"prepare_target must return dict or None, but received {type(prepared_target)}.")
            item.pop("target", None)
            item.pop("target_extra", None)
            item.update(prepared_target)
        return item

    def ensure_timeseries_length(self, data_df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
        """Pad or truncate a resampled window to the dataset input length."""
        expected = self.get_timeseries_len()
        if data_df.empty:
            raise ValueError(f"EDF read returned no samples for window starting at {start_date}.")
        if len(data_df) < expected:
            missing = expected - len(data_df)
            period = pd.to_timedelta(1.0 / self.sample_frequency, unit="s")
            padding = pd.DataFrame([data_df.iloc[-1].values] * missing, columns=data_df.columns, index=pd.date_range(data_df.index[-1] + period, periods=missing, freq=period))
            return pd.concat([data_df, padding])
        if len(data_df) > expected:
            return data_df.head(expected)
        return data_df

    def get_items(self, file: EDFFile, start_dates: Sequence[pd.Timestamp]) -> list[Optional[Dict[str, Any]]]:
        """Build several model windows through the canonical EDF access path.

        Args:
            file: Prepared patient descriptor returned by :meth:`initialize`.
            start_dates: Logical model-window starts. Every window has this
                dataset's ``total_input`` duration and is returned in the same
                order as the supplied timestamps.

        Returns:
            One prepared sample dictionary per requested timestamp. An entry
            is ``None`` when ``prepare_target`` or ``prepare_sample`` rejects
            that particular window.

        Notes:
            Target preparation and channel selection happen separately for
            each request and before signal I/O. ``EDFFile.get_x`` performs
            resampling and optionally serves the requested interval from a
            complete-recording cache.

            ``PairedDataset`` uses this method to load the native calls needed
            by one or more compatible packaged experts. Ordinary datasets use
            the same implementation because :meth:`get_item` is a one-window
            wrapper around this method.
        """
        start_dates = [pd.Timestamp(start_date) for start_date in start_dates]
        if not start_dates:
            raise ValueError("start_dates must not be empty.")

        plans = []
        for start_date in start_dates:
            item = self.get_target_item(file, start_date)
            if item is None:
                plans.append(None)
                continue
            selected_channels = self.select_channels(file.channels)
            channels_to_load = self.channels_to_load(selected_channels, file.channels)
            plans.append((item, start_date, selected_channels, channels_to_load))

        if not any(plan is not None for plan in plans):
            return [None] * len(plans)

        results = []
        for plan in plans:
            if plan is None:
                results.append(None)
                continue
            item, start_date, selected_channels, channels_to_load = plan
            data_df = file.get_x(start_date, start_date + self.total_input, self.sample_frequency, self.resample_type, channels=channels_to_load, cache=self.edf_cache)
            missing_channels = sorted(set(channels_to_load) - set(data_df.columns))
            if missing_channels:
                raise ValueError(f"EDF read for {file.path} is missing required channels {missing_channels} at {start_date}.")
            file.apply_unit_conversion(data_df)
            apply_normalizers(data_df, file.normalizers)
            self.apply_rereference(data_df)
            file.apply_z_normalization(data_df, channels=list(dict.fromkeys(physical_name for cfg, physical_name in selected_channels)))
            data_df = self.ensure_timeseries_length(data_df, start_date)

            transformed_item = self.run_build_sample(item, data_df, selected_channels=selected_channels)
            if transformed_item is None:
                results.append(None)
                continue
            item.update(transformed_item)
            results.append(item)
        return results

    def get_item(self, file: EDFFile, start_date: pd.Timestamp):
        """Build one candidate item through the canonical multi-window path."""
        return self.get_items(file, [start_date])[0]

    def candidate_indices(self, original_idx: int):
        """Yield the requested candidate followed by configured rejection fallbacks."""
        original_pidx = bisect.bisect_right(self.upper_bounds, original_idx)
        yield original_idx
        strategy = self.rejection_strategy
        if strategy == "none":
            return
        if strategy in {"patient", "patient_then_global"}:
            for _ in range(max(0, self.online_max_tries - 1)):
                yield int(np.random.randint(self.lower_bounds[original_pidx], self.upper_bounds[original_pidx]))
        if strategy in {"global", "patient_then_global"}:
            for _ in range(self.online_max_tries):
                yield int(np.random.choice(len(self)))

    def item_from_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """Materialize one exact indexed candidate without fallback sampling."""
        pidx = bisect.bisect_right(self.upper_bounds, idx)
        file = self.edf_files[pidx]
        new_idx = idx - self.lower_bounds[pidx]
        if file.start_offsets is not None:
            cur_date = file.start_date + self.stride * int(file.start_offsets[new_idx])
        else:
            cur_date = file.start_date + self.stride * new_idx
        views = []
        for _ in range(self.n_views):
            item = self.get_item(file, cur_date)
            if item is None:
                return None
            views.append(item)
        return views[0] if self.n_views == 1 else stack_repeated_views(views)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Return one logical sample, using configured fallbacks after rejection.

        Args:
            idx: Global window index into the prepared patient list.

        Returns:
            A sample dictionary produced by :meth:`get_item`.

        Expected candidate rejection returns ``None`` after the configured
        strategy is exhausted. Exceptions raised while constructing a
        candidate are not rejection and propagate immediately.
        """
        if not self.initialized:
            raise ValueError(f"{self.__class__.__name__} is not initialized. Call initialize(...) before using __getitem__.")
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        for candidate_idx in self.candidate_indices(idx):
            item = self.item_from_index(candidate_idx)
            if item is not None:
                return item
        return None
