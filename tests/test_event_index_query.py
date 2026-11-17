"""Compare indexed queries with direct half-open interval membership."""

import numpy as np
import pandas as pd
import pytest

from sleepwalker.datasets.Basedataset import EventIndex


@pytest.mark.parametrize('timezone', [None, 'Europe/Berlin'])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('dtype', ['int8', 'bool'])
def test_queries_match_interval_membership(timezone, sparse, dtype):
    rng = np.random.default_rng(713)
    origin = pd.Timestamp('2024-10-27 00:00', tz=timezone)
    starts = rng.integers(0, 20000, 100)
    ends = starts + rng.integers(1, 5000, 100)
    # Unsorted, duplicate, touching and nested intervals, including a long outer one.
    starts = np.concatenate([starts, [0, 2000, 2000, 3000, 1000]])
    ends = np.concatenate([ends, [30000, 3000, 3000, 4000, 1001]])
    labels = list(rng.choice(['a', 'b'], 100)) + ['a'] * 5
    events = pd.DataFrame({'Label': labels,
                           'Starttime': origin + pd.to_timedelta(starts, unit='ms'),
                           'Endtime': origin + pd.to_timedelta(ends, unit='ms')})
    events['metadata'] = np.arange(len(events))
    before = events.copy(deep=True)
    index = EventIndex(events)
    # Empty, out-of-range, whole-recording, aligned and sub-sample queries.
    requests = [(-2, -1), (-1, 32), (0, 0), (2, 4), (28, 29), (40, 41)]
    requests += [(float(t), float(t + 2)) for t in rng.uniform(0, 28, 8)]
    for start, end in requests:
        for frequency in ['1s', '137ms']:
            grid = pd.date_range(origin + pd.Timedelta(seconds=start), origin + pd.Timedelta(seconds=end), freq=frequency, inclusive='left')
            output = {}
            for label in ['b', 'a', 'missing']:
                active = np.zeros(len(grid), dtype=bool)
                for row in events.loc[events.Label == label].itertuples():
                    active |= (grid >= row.Starttime) & (grid < row.Endtime)
                output[label] = active.astype(dtype)
            expected = pd.DataFrame(output, index=grid)
            if sparse:
                expected = expected.astype(pd.SparseDtype(dtype))
            actual = index.query(origin + pd.Timedelta(seconds=start), origin + pd.Timedelta(seconds=end), frequency, dtype=dtype, sparse=sparse, labels=['b', 'a', 'missing'])
            pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(index.events, before)
    pd.testing.assert_frame_equal(events, before)


def test_empty_events_and_empty_requested_labels():
    events = pd.DataFrame({'Label': pd.Series(dtype='object'),
                           'Starttime': pd.Series(dtype='datetime64[ns]'),
                           'Endtime': pd.Series(dtype='datetime64[ns]')})
    index = EventIndex(events)
    grid = pd.date_range('2024-01-01', periods=3, freq='1s')
    pd.testing.assert_frame_equal(index.query(grid[0], grid[-1] + pd.Timedelta(seconds=1), '1s', labels=['missing']),
                                  pd.DataFrame({'missing': np.zeros(3, dtype='int8')}, index=grid))
    pd.testing.assert_frame_equal(index.query(grid[0], grid[-1] + pd.Timedelta(seconds=1), '1s', labels=[]), pd.DataFrame(index=grid))
