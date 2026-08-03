import pytest
import torch

from sleepwalker.models.ExpertPortComposer import (
    ExpertPortComposer,
    ExpertPortSpec,
    PortInterfaceEdge,
)


SPECS = [
    ExpertPortSpec("sleep", feature_dim=4, probability_dim=3),
    ExpertPortSpec("arousal", feature_dim=5, probability_dim=2),
    ExpertPortSpec("breathing", feature_dim=6, probability_dim=3),
]


def ports(batch_size=7):
    features = {spec.name: torch.randn(batch_size, spec.feature_dim) for spec in SPECS}
    probabilities = {
        spec.name: torch.softmax(torch.randn(batch_size, spec.probability_dim), dim=1)
        for spec in SPECS
    }
    available = {spec.name: torch.ones(batch_size, 1) for spec in SPECS}
    return features, probabilities, available


@pytest.mark.parametrize("method", ["independent", "stacking", "concat"])
def test_baselines_emit_receiver_logits(method):
    model = ExpertPortComposer(
        receiver="arousal",
        port_specs=SPECS,
        num_classes=2,
        method=method,
    )
    assert model(*ports()).shape == (7, 2)
    assert model.communication_scalars_per_sample() == 0


def test_structured_composer_exposes_only_incoming_bottleneck_messages():
    model = ExpertPortComposer(
        receiver="arousal",
        port_specs=SPECS,
        num_classes=2,
        method="structured",
        edges=[
            PortInterfaceEdge("sleep", "arousal", bottleneck_dim=3),
            PortInterfaceEdge("breathing", "arousal", bottleneck_dim=2),
            PortInterfaceEdge("arousal", "breathing", bottleneck_dim=4),
        ],
        receiver_adapter_dim=2,
    )
    assert model(*ports()).shape == (7, 2)
    assert model.communication_scalars_per_sample() == 5
    assert set(model.gate_values()) == {"sleep__to__arousal", "breathing__to__arousal"}


def test_missing_source_is_masked_and_bad_contracts_fail_early():
    model = ExpertPortComposer(
        receiver="arousal",
        port_specs=SPECS,
        num_classes=2,
        method="structured",
        edges=[PortInterfaceEdge("sleep", "arousal", bottleneck_dim=2)],
    )
    features, probabilities, available = ports()
    available["sleep"].zero_()
    first = model(features, probabilities, available)
    features["sleep"].normal_(mean=1000, std=100)
    second = model(features, probabilities, available)
    assert torch.allclose(first, second)

    del features["sleep"]
    with pytest.raises(ValueError, match="features must contain exactly"):
        model(features, probabilities, available)
