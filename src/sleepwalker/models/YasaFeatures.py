"""YASA's sleep-staging features as an encoder: [B, T, C] windows in, [B, n_features] features out."""

from typing import Any

import antropy
import numpy as np
import scipy.integrate
import scipy.signal
import scipy.stats
import torch
import yasa

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.utils import logger

SAMPLING_FREQUENCY = 100  # YASA computes its staging features at 100 Hz
BANDS = [(0.4, 1, "sdelta"), (1, 4, "fdelta"), (4, 8, "theta"), (8, 12, "alpha"), (12, 16, "sigma"), (16, 30, "beta")]  # as in yasa.SleepStaging.fit
STAGING_FEATURES = {  # as in yasa.SleepStaging.fit, in the same order
    "eeg": ["std", "iqr", "skew", "kurt", "nzc", "hmob", "hcomp", "sdelta", "fdelta", "theta", "alpha", "sigma", "beta", "dt", "ds", "db", "at", "abspow", "perm", "higuchi", "petrosian"],
    "eog": ["std", "iqr", "skew", "kurt", "nzc", "hmob", "hcomp", "sdelta", "fdelta", "theta", "alpha", "sigma", "beta", "abspow", "perm", "higuchi", "petrosian"],
    "emg": ["std", "iqr", "skew", "kurt", "nzc", "hmob", "hcomp", "abspow", "perm", "higuchi", "petrosian"],
}
GROUPS = ("staging", "spindles", "slow_waves", "rem")


class YasaFeatures(BaseModel, EmbeddingModel):
    """YASA's sleep-staging features (Vallat & Walker, 2021) per channel, optionally with spindles, slow waves and REMs.

    YASA's features and detection thresholds are in microvolts, so the input must be band-passed microvolts, not z-scored.
    A channel that is flat or swamped by artifacts in a window (it fails YASA's amplitude check) gets its training means there instead of meaningless features.

    Args:
        ts_len: Samples per window, filled in from the data config; must be 30 s.
        n_channels: Number of input channels, filled in from the data config.
        input_channels: Logical channel names, filled in from the data config.
        sampling_frequency: Sampling rate, filled in from the data config; must be 100 Hz.
        eeg: Logical names of the EEG channels; spindles and slow_waves run on them.
        eog: Logical names of the EOG channels; rem needs two, [left, right], and looks for opposite deflections in them.
        emg: Logical names of the EMG channels.
        groups: Any of "staging", "spindles", "slow_waves", "rem".
        fit_windows: Number of training windows the training means are taken from.
    """

    def __init__(self, *, ts_len: int, n_channels: int, input_channels: list[str], sampling_frequency: float, eeg: list[str] | None = None, eog: list[str] | None = None, emg: list[str] | None = None, groups: tuple[str, ...] = ("staging",), fit_windows: int | None = 1024):
        super().__init__()
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.input_channels = list(input_channels)
        self.groups = tuple(groups)
        self.fit_windows = fit_windows

        # YASA computes its features on 30-s epochs at 100 Hz
        if float(sampling_frequency) != SAMPLING_FREQUENCY:
            raise ValueError(f"YasaFeatures needs {SAMPLING_FREQUENCY} Hz, got {sampling_frequency}; set sample_frequency: {SAMPLING_FREQUENCY} (ideally with resample_type: polyphase) in the data config.")
        if self.ts_len != 30 * SAMPLING_FREQUENCY:
            raise ValueError(f"YasaFeatures needs 30-s windows ({30 * SAMPLING_FREQUENCY} samples), got {self.ts_len}.")

        # look up each type's channels and check that the requested groups can run on them
        if not self.groups or not set(self.groups) <= set(GROUPS):
            raise ValueError(f"groups must be one or more of {GROUPS}, got {self.groups}.")
        self.channels = {"eeg": self.channel_indices(eeg or []), "eog": self.channel_indices(eog or []), "emg": self.channel_indices(emg or [])}
        listed = [i for indices in self.channels.values() for i in indices]
        if not listed:
            raise ValueError("YasaFeatures needs at least one channel in eeg, eog or emg.")
        if len(listed) != len(set(listed)):
            raise ValueError(f"A channel is listed under more than one type: eeg={eeg}, eog={eog}, emg={emg}.")
        if {"spindles", "slow_waves"} & set(self.groups) and not self.channels["eeg"]:
            raise ValueError("The spindles and slow_waves groups need at least one EEG channel.")
        if "rem" in self.groups and len(self.channels["eog"]) != 2:
            raise ValueError("The rem group needs exactly two EOG channels, [left, right].")

        # the output's column names; their number is the head's input size, known before any data is seen
        self.feature_names = []
        if "staging" in self.groups:
            for kind, indices in self.channels.items():
                for i in indices:
                    self.feature_names += [f"{self.input_channels[i]}__{feature}" for feature in STAGING_FEATURES[kind]]
        for i in self.channels["eeg"]:
            name = self.input_channels[i]
            if "spindles" in self.groups:
                self.feature_names += [f"{name}__n_spindles", f"{name}__spindle_seconds"]
            if "slow_waves" in self.groups:
                self.feature_names += [f"{name}__n_slow_waves", f"{name}__slow_wave_seconds"]
        if "rem" in self.groups:
            self.feature_names += ["eog__n_rems", "eog__rem_seconds"]

        # set by fit: the training means that replace a failing channel's features
        self.register_buffer("fill_values", torch.zeros(len(self.feature_names), dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))

    def channel_indices(self, names: list[str]) -> list[int]:
        """Positions of logical channel names in input_channels."""
        unknown = [name for name in names if name not in self.input_channels]
        if unknown or len(set(names)) != len(names):
            raise ValueError(f"Channels {names} must be distinct names from the data config's {self.input_channels}.")
        return [self.input_channels.index(name) for name in names]

    @torch.no_grad()
    def warmup_preprocessor(self, data_loader, device: str):
        """Fit the fill values on the first fit_windows training windows, then stop. The trainer calls this before training."""
        self.to(device)
        if self.fitted:
            return self
        windows, collected = [], 0
        for batch in data_loader:
            if batch is None:
                continue
            windows.append(batch["data"] if self.fit_windows is None else batch["data"][: self.fit_windows - collected])
            collected += len(windows[-1])
            if self.fit_windows is not None and collected >= self.fit_windows:
                break
        self.fit(torch.cat(windows))
        return self

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """Fill values: the mean of every feature over the training windows whose channel passes the amplitude check."""
        x = data.transpose(1, 2).double().cpu().numpy()
        for indices in self.channels.values():
            for i in indices:
                if not amplitude_valid(x[:, i]).any():
                    raise ValueError(f"Channel {self.input_channels[i]} fails YASA's amplitude check in every training window; is the data in microvolts?")
        features, valid = self.raw_features(x)
        counts = valid.sum(0)
        # the mean over the windows where the channel passed; the others only hold placeholders
        means = np.where(valid, features, 0.0).sum(0) / np.maximum(counts, 1)  # rem's two channels may never pass together
        self.fill_values = torch.as_tensor(means, dtype=torch.float64, device=self.fill_values.device)
        self.fitted = torch.tensor(True, device=self.fill_values.device)
        logger.info(f"YasaFeatures fitted on {len(data)} windows.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, n_features]: per channel its staging features, then the detections."""
        if tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"YasaFeatures expects [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        if not torch.isfinite(x).all():
            raise ValueError("YasaFeatures got non-finite input.")
        if not self.fitted:
            logger.warning("YasaFeatures is not fitted yet; channels failing the amplitude check get zeros until the trainer's warmup has run.")
        features, valid = self.raw_features(x.detach().transpose(1, 2).double().cpu().numpy())
        features = np.where(valid, features, self.fill_values.cpu().numpy())
        if not np.isfinite(features).all():
            raise ValueError("YasaFeatures produced non-finite features.")
        return torch.as_tensor(features, dtype=torch.float32, device=x.device)

    def raw_features(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[B, C, T] microvolts -> features and their validity, both [B, n_features], before filling."""
        channel_valid = {i: amplitude_valid(x[:, i]) for indices in self.channels.values() for i in indices}
        columns, valid = [], []
        # like the detectors, the staging features see only the windows that pass; the others keep 0 until they are filled
        if "staging" in self.groups:
            for kind, indices in self.channels.items():
                for i in indices:
                    rows = np.flatnonzero(channel_valid[i])
                    block = np.zeros((len(x), len(STAGING_FEATURES[kind])))
                    if len(rows):
                        block[rows] = staging_features(x[rows, i], kind)
                    columns.append(block)
                    valid += [channel_valid[i]] * len(STAGING_FEATURES[kind])
        for i in self.channels["eeg"]:
            if "spindles" in self.groups:
                columns.append(spindle_features(x[:, i]))
                valid += [channel_valid[i]] * 2
            if "slow_waves" in self.groups:
                columns.append(slow_wave_features(x[:, i]))
                valid += [channel_valid[i]] * 2
        if "rem" in self.groups:
            left, right = self.channels["eog"]
            columns.append(rem_features(x[:, left], x[:, right]))
            valid += [channel_valid[left] & channel_valid[right]] * 2
        return np.concatenate(columns, axis=1), np.stack(valid, axis=1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Number of output features."""
        return len(self.feature_names)

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}


def staging_features(x: np.ndarray, kind: str) -> np.ndarray:
    """[N, T] microvolts -> [N, len(STAGING_FEATURES[kind])]: YASA's per-epoch staging features."""
    # The same calls as yasa.SleepStaging.fit.
    x = np.ascontiguousarray(x)  # antropy's numba functions need contiguous rows
    hmob, hcomp = antropy.hjorth_params(x, axis=1)
    features = {
        "std": np.std(x, ddof=1, axis=1),
        "iqr": scipy.stats.iqr(x, rng=(25, 75), axis=1),
        "skew": scipy.stats.skew(x, axis=1),
        "kurt": scipy.stats.kurtosis(x, axis=1),
        "nzc": antropy.num_zerocross(x, axis=1),
        "hmob": hmob,
        "hcomp": hcomp,
    }
    freqs, psd = scipy.signal.welch(x, SAMPLING_FREQUENCY, window="hamming", nperseg=5 * SAMPLING_FREQUENCY, average="median")
    if kind != "emg":
        powers = yasa.bandpower_from_psd_ndarray(psd, freqs, bands=BANDS)  # one row per band: its share of the 0.4-30 Hz power
        for j, (_, _, name) in enumerate(BANDS):
            features[name] = powers[j]
    if kind == "eeg":
        delta = features["sdelta"] + features["fdelta"]
        features["dt"] = delta / features["theta"]
        features["ds"] = delta / features["sigma"]
        features["db"] = delta / features["beta"]
        features["at"] = features["alpha"] / features["theta"]
    broad = (freqs >= BANDS[0][0]) & (freqs <= BANDS[-1][1])
    features["abspow"] = scipy.integrate.trapezoid(psd[:, broad], dx=freqs[1] - freqs[0])
    features["perm"] = np.apply_along_axis(antropy.perm_entropy, 1, x, normalize=True)
    features["higuchi"] = np.apply_along_axis(antropy.higuchi_fd, 1, x)
    features["petrosian"] = antropy.petrosian_fd(x, axis=1)
    return np.stack([features[name] for name in STAGING_FEATURES[kind]], axis=1)


def amplitude_valid(x: np.ndarray) -> np.ndarray:
    """[N, T] -> [N] bool: YASA's amplitude check, a 5 %-trimmed std between 0.1 (flat) and 1000 microvolts (huge artifacts)."""
    # As in the amplitude check of YASA's detectors (_check_data_hypno).
    trimmed_std = yasa.others.trimbothstd(x, cut=0.05)
    return (trimmed_std > 0.1) & (trimmed_std < 1e3)


def spindle_features(x: np.ndarray) -> np.ndarray:
    """[N, T] -> [N, 2]: number and total duration of YASA's spindles per window."""
    # One spindles_detect call with the windows as rows; windows failing the amplitude check are left out (they get the
    # fill values later), so that YASA does not log an error for each of them.
    rows = np.flatnonzero(amplitude_valid(x))
    result = yasa.spindles_detect(x[rows], SAMPLING_FREQUENCY, verbose="error") if len(rows) else None
    return event_counts(result, rows, len(x))


def slow_wave_features(x: np.ndarray) -> np.ndarray:
    """[N, T] -> [N, 2]: number and total duration of YASA's slow waves per window."""
    # One sw_detect call with the windows as rows, as for the spindles.
    rows = np.flatnonzero(amplitude_valid(x))
    result = yasa.sw_detect(x[rows], SAMPLING_FREQUENCY, verbose="error") if len(rows) else None
    return event_counts(result, rows, len(x))


def rem_features(loc: np.ndarray, roc: np.ndarray) -> np.ndarray:
    """[N, T] left and right EOG -> [N, 2]: number and total duration of YASA's rapid eye movements per window."""
    # rem_detect is one-dimensional, so one call per window.
    features = np.zeros((len(loc), 2))
    for row in np.flatnonzero(amplitude_valid(loc) & amplitude_valid(roc)):
        result = yasa.rem_detect(loc[row], roc[row], SAMPLING_FREQUENCY, verbose="error")
        if result is not None:
            events = result.summary()
            features[row] = len(events), events["Duration"].sum()
    return features


def event_counts(result, rows: np.ndarray, n_windows: int) -> np.ndarray:
    """[n_windows, 2]: number and total duration of a batched YASA detection's events per window."""
    features = np.zeros((n_windows, 2))
    if result is None:  # YASA returns None when it finds nothing
        return features
    events = result.summary()
    window = rows[events["IdxChannel"].to_numpy()]
    features[:, 0] = np.bincount(window, minlength=n_windows)
    features[:, 1] = np.bincount(window, weights=events["Duration"].to_numpy(), minlength=n_windows)
    return features
