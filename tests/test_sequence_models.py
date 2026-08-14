import torch

from sleepwalker.models.USleep import USleep
from sleepwalker.models.UTime import UTime
from sleepwalker.models.TinySleepNet import TinySleepNet


def test_utime_processes_full_context_and_returns_sequence_logits():
    model = UTime(
        ts_len=80,
        n_channels=2,
        sampling_frequency="100ms",
        classes=["a", "b", "c"],
        channel=[4, 8],
        maxpool=[2, 2],
        kernel=[3, 3],
        norm="batch",
        dropout_p=0,
        epoch_len="1s",
        sequence_len=4,
    )
    features = model.features(torch.randn(2, 80, 2))
    encoder_inputs = []
    handle = model.encoder.enc_blocks[0].conv1.register_forward_hook(lambda module, args, output: encoder_inputs.append(args[0].shape))

    output = model(torch.randn(2, 80, 2))
    handle.remove()

    assert output.shape == (2, 4, 3)
    assert features.shape == (2, 4 * model.mlp_size)
    assert model.feature_dim() == 4 * model.mlp_size
    assert encoder_inputs == [torch.Size([2, 2, 80])]


def test_usleep_processes_full_context_and_returns_sequence_logits():
    model = USleep(
        ts_len=80,
        n_channels=2,
        sampling_frequency=10,
        classes=["a", "b", "c"],
        depth=2,
        init_filters=4,
        kernel_size=3,
        complexity_factor=1,
        epoch_len="1s",
        sequence_len=4,
    )
    encoder_inputs = []
    handle = model.encoder_convs[0].register_forward_hook(lambda module, args, output: encoder_inputs.append(args[0].shape))

    output = model(torch.randn(2, 80, 2))
    handle.remove()

    assert output.shape == (2, 4, 3)
    assert encoder_inputs == [torch.Size([2, 2, 80])]


def test_tinysleepnet_returns_center_aligned_sequence_logits():
    model = TinySleepNet(
        ts_len=9000,
        n_channels=1,
        classes=["a", "b", "c"],
        seq_len=3,
        sampling_frequency=100,
        use_lstm=True,
        sequence_len=2,
    )

    output = model(torch.randn(2, 9000, 1))

    assert output.shape == (2, 2, 3)
