#!/usr/bin/env python3
"""
Generate synthetic EDF dataset for testing.

- 5 patients, ~1h recording each
- Channels:
    - EEG (200 Hz)
    - EOG (100 Hz)
    - EMG (100 Hz)
    - MOVEMENT (50 Hz) for patients 4 and 5
- Sleep stages: random labels per 30s interval -> written to text files (stages.txt, extra_stages.txt)
"""

import os
import numpy as np
import datetime
import pyedflib

def make_signal(freq_hz, duration_s, rng):
    """Generate random noise signal for one channel."""
    n_samples = int(freq_hz * duration_s)
    sig = rng.normal(loc=0, scale=50, size=n_samples).astype(np.float64)
    sig = sig / np.std(sig) * 50.0
    return sig

def write_edf(file_path, channels, fs_map, phys_limits, duration_s, rng):
    """Write EDF file with given channels."""
    n_channels = len(channels)
    signals = []
    ch_info = []

    for ch in channels:
        fs = fs_map[ch]
        sig = make_signal(fs, duration_s, rng)
        
        phys_min, phys_max = phys_limits[ch]
        signals.append(sig)
        ch_info.append({
            "label": ch,
            "dimension": "uV",
            "sample_frequency": fs,
            "physical_min": phys_min,
            "physical_max": phys_max,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        })

    writer = pyedflib.EdfWriter(str(file_path), n_channels=n_channels,file_type=pyedflib.FILETYPE_EDFPLUS)
    start = datetime.datetime(2025, 1, 1, 0, 0, 0)
    writer.setStartdatetime(start)
    writer.setSignalHeaders(ch_info)
    writer.writeSamples(signals)
    writer.close()

def write_stage_file(file_path, n_intervals, rng):
    """Write fake sleep stages to a txt file."""
    with open(file_path, "w") as f:
        for i in range(n_intervals):
            stage = rng.choice(["wake","rem", "n1", "n2", "n3"])
            start = i * 30
            end = start + 30
            f.write(f"{start},{end},{stage}\n")

def main(out_dir=".", seed=0):
    duration_s = 60 * 60  # 1h
    rng = np.random.default_rng(seed)

    fs_map = {
        "EEG": 200,
        "EOG": 100,
        "EMG": 100,
        "MOVEMENT": 50,
    }

    phys_limits = {
        "EEG": (-500.0, 500.0),
        "EOG": (-500.0, 500.0),
        "EMG": (-500.0, 500.0),
        "MOVEMENT": (-2000.0, 2000.0),
    }

    for pid in range(1, 6):
        # channels (movement only for last two patients)
        channels = ["EEG", "EOG", "EMG"]
        if pid in (4, 5):
            channels.append("MOVEMENT")

        # write EDF
        edf_path = os.path.join(out_dir, f"signals_{pid:02d}.edf")
        write_edf(edf_path, channels, fs_map, phys_limits, duration_s, rng)

        # number of 30s epochs
        n_intervals = duration_s // 30

        # write stages
        stage_path = os.path.join(out_dir, f"stages_{pid:02d}.txt")
        write_stage_file(stage_path, n_intervals, rng)

        # write extra stages
        extra_stage_path = os.path.join(out_dir, f"extra_stages_{pid:02d}.txt")
        write_stage_file(extra_stage_path, n_intervals, rng)

if __name__ == "__main__":
    main()