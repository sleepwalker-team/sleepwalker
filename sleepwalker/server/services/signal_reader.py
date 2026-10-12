import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyedflib


class SignalNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class SignalSamples:
    times: list[float]
    values: list[float]


class SignalReader(Protocol):
    def read(
        self,
        source_path: Path,
        signal_id: str,
        start: float,
        end: float,
    ) -> SignalSamples:
        ...


class EdfSignalReader:
    def read(
        self,
        source_path: Path,
        signal_id: str,
        start: float,
        end: float,
    ) -> SignalSamples:
        reader = pyedflib.EdfReader(str(source_path))

        try:
            labels = [label.strip() for label in reader.getSignalLabels()]

            try:
                signal_index = labels.index(signal_id)
            except ValueError as exc:
                raise SignalNotFoundError(signal_id) from exc

            sample_rate = float(reader.getSampleFrequency(signal_index))
            sample_count = int(reader.getNSamples()[signal_index])
            start_index = max(0, math.floor(start * sample_rate))
            end_index = min(sample_count, math.ceil(end * sample_rate))

            if end_index <= start_index:
                return SignalSamples(times=[], values=[])

            values = reader.readSignal(
                signal_index,
                start=start_index,
                n=end_index - start_index,
            )
            times = [index / sample_rate for index in range(start_index, end_index)]
            filtered = [
                (time, float(value))
                for time, value in zip(times, values)
                if start <= time < end
            ]
            return SignalSamples(
                times=[time for time, _ in filtered],
                values=[value for _, value in filtered],
            )
        finally:
            reader.close()


def downsample_extrema(samples: SignalSamples, max_points: int) -> SignalSamples:
    sample_count = len(samples.times)

    if sample_count <= max_points:
        return samples

    if max_points == 2:
        indices = [0, sample_count - 1]
    else:
        interior_count = sample_count - 2
        bucket_count = min(max_points - 2, interior_count)
        indices = [0]

        for bucket in range(bucket_count):
            bucket_start = 1 + math.floor(bucket * interior_count / bucket_count)
            bucket_end = 1 + math.floor((bucket + 1) * interior_count / bucket_count)
            strongest = max(
                range(bucket_start, bucket_end),
                key=lambda index: abs(samples.values[index]),
            )
            indices.append(strongest)

        indices.append(sample_count - 1)

    return SignalSamples(
        times=[samples.times[index] for index in indices],
        values=[samples.values[index] for index in indices],
    )
