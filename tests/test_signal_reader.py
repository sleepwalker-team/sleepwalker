from sleepwalker.server.services.signal_reader import SignalSamples, downsample_extrema


def test_downsample_extrema_preserves_boundaries_and_strongest_values() -> None:
    samples = SignalSamples(
        times=[float(index) for index in range(10)],
        values=[0.0, 1.0, -8.0, 2.0, 1.0, 9.0, 2.0, -7.0, 1.0, 0.0],
    )

    reduced = downsample_extrema(samples, max_points=4)

    assert reduced.times[0] == 0.0
    assert reduced.times[-1] == 9.0
    assert len(reduced.times) == 4
    assert -8.0 in reduced.values
    assert 9.0 in reduced.values
