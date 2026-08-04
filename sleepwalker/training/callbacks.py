"""Common dataset callbacks used by standard training configurations."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.trainer.utils.filtering import trim_event


def trim_patient_to_events(
    data_df,
    label_df,
    label_extra_df,
    patient=None,
    *,
    keep_events: Sequence[str] | None = None,
):
    """Adapt :func:`trim_event` to the ``prepare_patient`` callback contract."""
    trimmed = trim_event(data_df, label_df, label_extra_df, keep_events)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df


def tensor_sample(
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
    """Build the standard tensor sample with optional signal-quality checks.

    All checks are opt-in. Missing configured channels are ignored so the same
    callback can be shared by grouped and literal-channel configurations.
    """
    if max_nan_fraction is not None:
        if float(data.isna().mean().mean()) > float(max_nan_fraction):
            return None

    for channel, maximum in dict(quality_max_mean or {}).items():
        if quality_data is not None and channel in quality_data.columns:
            if float(quality_data[channel].astype(float).mean()) > float(maximum):
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

    result = {
        "data": torch.from_numpy(data.values).float(),
        "target": target,
        "patient": patient,
        "time": time,
        **item,
    }
    if target_extra is not None:
        result["target_extra"] = target_extra
    return result


def robust_scaler(
    *,
    n_channels: int,
    skip_first: int = 0,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> RobustScaler:
    """Construct a scaler for a dataset-derived channel range."""
    return RobustScaler(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        channels=list(range(int(skip_first), int(n_channels))),
    )
