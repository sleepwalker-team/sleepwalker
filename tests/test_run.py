from sleepwalker.trainer.Run import _infer_summary_input_size


class DummyGroupedDataset:
    channels = list(range(9))

    def get_timeseries_len(self):
        return 3000

    def get_input_channels(self):
        return ["EEG", "Chin EMG"]


def test_infer_summary_input_size_prefers_effective_grouped_channels():
    assert _infer_summary_input_size(DummyGroupedDataset()) == (3000, 2)
