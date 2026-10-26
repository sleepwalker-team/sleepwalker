import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyedflib
import numpy as np


SIGNAL_CHUNK_SIZE = 65536


class SignalNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class SignalSamples:
    times: list[float]
    values: list[float]
    sample_count: int | None = None


class SignalReader(Protocol):
    def read(
        self,
        source_path: Path,
        signal_id: str,
        start: float,
        end: float,
        *,
        max_points: int = 2000,
    ) -> SignalSamples:
        ...


class EdfSignalReader:
    def read(
        self,
        source_path: Path,
        signal_id: str,
        start: float,
        end: float,
        *,
        max_points: int = 2000,
    ) -> SignalSamples:
        if max_points < 2 or max_points > 100_000:
            raise ValueError("max_points must be between 2 and 100000.")
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
            while start_index < end_index and start_index / sample_rate < start:
                start_index += 1
            while end_index > start_index and (end_index - 1) / sample_rate >= end:
                end_index -= 1

            if end_index <= start_index:
                return SignalSamples(times=[], values=[], sample_count=0)

            count = end_index - start_index
            if count <= max_points:
                values = []
                for _, chunk in _signal_chunks(reader, signal_index, start_index, end_index):
                    values.extend(chunk.tolist())
                return SignalSamples(
                    times=[index / sample_rate for index in range(start_index, end_index)],
                    values=values, sample_count=count,
                )

            if max_points == 2:
                first = float(reader.readSignal(signal_index, start=start_index, n=1)[0])
                last = float(reader.readSignal(signal_index, start=end_index - 1, n=1)[0])
                return SignalSamples(
                    times=[start_index / sample_rate, (end_index - 1) / sample_rate],
                    values=[first, last], sample_count=count,
                )

            indices, values = _stream_extrema(reader, signal_index, start_index, end_index, max_points)
            return SignalSamples(
                times=[index / sample_rate for index in indices], values=values, sample_count=count,
            )
        finally:
            reader.close()


def _signal_chunks(reader, signal_index: int, start: int, end: int):
    for index in range(start, end, SIGNAL_CHUNK_SIZE):
        size = min(SIGNAL_CHUNK_SIZE, end - index)
        values = reader.readSignal(signal_index, start=index, n=size)
        if len(values) != size:
            raise ValueError("EDF reader returned fewer samples than requested.")
        yield index, np.asarray(values)


def _stream_extrema(reader, signal_index: int, start: int, end: int, max_points: int):
    interior_count = end - start - 2
    bucket_count = max_points - 2
    indices = [start] * max_points
    values = [None] * max_points
    indices[-1] = end - 1
    bucket = 0
    for chunk_start, chunk in _signal_chunks(reader, signal_index, start, end):
        chunk_end = chunk_start + len(chunk)
        if chunk_start == start:
            values[0] = float(chunk[0])
        if chunk_end == end:
            values[-1] = float(chunk[-1])
        segment_start = max(start + 1, chunk_start)
        segment_end = min(end - 1, chunk_end)
        while segment_start < segment_end:
            bucket_end = start + 1 + math.floor((bucket + 1) * interior_count / bucket_count)
            stop = min(segment_end, bucket_end)
            segment = chunk[segment_start - chunk_start:stop - chunk_start]
            local_index = int(np.argmax(np.abs(segment)))
            value = float(segment[local_index])
            slot = bucket + 1
            if values[slot] is None or abs(value) > abs(values[slot]):
                indices[slot] = segment_start + local_index
                values[slot] = value
            segment_start = stop
            if stop == bucket_end:
                bucket += 1

    return indices, values


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
        sample_count=samples.sample_count if samples.sample_count is not None else sample_count,
    )
