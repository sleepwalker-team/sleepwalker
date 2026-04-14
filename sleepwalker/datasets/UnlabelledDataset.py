"""Inference-time dataset template without label extraction.

`UnlabelledDataset` reuses the EDF loading and window-building logic from
`BaseDataset` but disables all label handling. It is primarily used by the
prediction-package export/load path and by inference helpers that score raw EDF
files without ground-truth annotations.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import pandas as pd

from sleepwalker.datasets.Basedataset import BaseDataset, ChannelConfig


class UnlabelledDataset(BaseDataset):
    """Dataset template for EDF inference without labels.

    Args:
        channels: Signal channels to load.
        sample_frequency: Resampling frequency in Hz.
        resample_type: Signal resampling mode.
        total_input: Input window duration.
        stride: Time between consecutive inference windows.
        prepare_patient: Optional whole-patient callback reused from
            `BaseDataset`.
        prepare_sample: Optional final sample callback reused from
            `BaseDataset`.
        online_max_tries: Retry budget when `prepare_sample` rejects a window.
        force_one_day: Whether to keep the one-day sanity check from the base
            dataset.
        rereference: Optional rereferencing groups applied after loading.
    """
    def __init__(
        self,
        *,
        channels: Sequence[ChannelConfig],
        sample_frequency: float,
        resample_type: str = "nearest",
        total_input: str | pd.Timedelta = "30s",
        stride: str | pd.Timedelta = "30s",
        prepare_patient: Optional[Callable] = None,
        prepare_sample: Optional[Callable] = None,
        online_max_tries: int = 128,
        force_one_day: bool = True,
        rereference=None,
    ) -> None:
        self._init_kwargs = {
            "channels": list(channels),
            "sample_frequency": sample_frequency,
            "resample_type": resample_type,
            "total_input": total_input,
            "stride": stride,
            "prepare_patient": prepare_patient,
            "prepare_sample": prepare_sample,
            "online_max_tries": online_max_tries,
            "force_one_day": force_one_day,
            "rereference": rereference,
        }
        super().__init__(
            channels=channels,
            sample_frequency=sample_frequency,
            resample_type=resample_type,
            total_input=total_input,
            target_resolution=total_input,
            stride=stride,
            event_mapping=None,
            prepare_patient=prepare_patient,
            prepare_target=None,
            prepare_sample=prepare_sample,
            online_max_tries=online_max_tries,
            force_one_day=force_one_day,
            rereference=rereference,
        )

    def clone(self) -> UnlabelledDataset:
        """Return a fresh dataset template with the same configuration."""
        return UnlabelledDataset(**self._init_kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Signal that unlabelled datasets do not provide event annotations."""
        raise ValueError("UnlabelledDataset does not provide labels.")

    def get_item(self, file, start_date: pd.Timestamp):
        """Build one inference sample without any target fields.

        Args:
            file: Prepared EDF descriptor from `BaseDataset.initialize`.
            start_date: Window start timestamp.

        Returns:
            A sample dictionary containing at least `data`, `patient`, and
            `time`, or `None` when `prepare_sample` rejects the window.
        """
        end_date = start_date + self.total_input
        item = {
            "patient": file.path,
            "time": start_date + self.total_input / 2,
        }
        x_df = file.get_x(start_date, end_date, self.sample_frequency, self.resample_type)

        if self.rereference:
            for refchannels in self.rereference:
                ref_cols = [r for r in refchannels if r in x_df.columns]
                if ref_cols:
                    x_df[ref_cols] = x_df[ref_cols].values - x_df[ref_cols].values.mean(axis=1)[:, None]

        if len(x_df) < self.get_timeseries_len():
            freq = x_df.index.freq or pd.infer_freq(x_df.index)
            n = self.get_timeseries_len() - len(x_df)
            pad_idx = pd.date_range(start=x_df.index[-1] + freq, periods=n, freq=freq)
            pad_df = pd.DataFrame([x_df.iloc[-1].values] * n, columns=x_df.columns, index=pad_idx)
            x_df = pd.concat([x_df, pad_df])
        elif len(x_df) > self.get_timeseries_len():
            x_df = x_df.head(n=self.get_timeseries_len())

        transformed_item = self.run_build_sample(item, x_df)
        if transformed_item is None:
            return None
        item.update(transformed_item)
        return item
