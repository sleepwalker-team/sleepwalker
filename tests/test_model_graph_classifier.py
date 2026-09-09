from pathlib import Path

import pandas as pd
import pytest
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.SyntheticDataset import SyntheticDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel
from iclr2026.pipeline import pair_datasets
from sleepwalker.models.ModelGraphClassifier import GraphNode, ModelGraphClassifier, PairedDataset, load_graph_node


DATA = Path(__file__).parent / "data"


class DummyExpert(BaseModel, EmbeddingModel, ClassifierModel):
    def __init__(self, ts_len, n_channels, feature_dim, classes, sequence_len):
        super().__init__()
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.classes = classes
        self.sequence_len = sequence_len
        self.projection = torch.nn.Linear(n_channels, feature_dim)
        self.head = torch.nn.Linear(feature_dim, sequence_len * len(classes))
        self.last_input = None

    def encode(self, x):
        self.last_input = x.detach().clone()
        return self.projection(x.mean(dim=1))

    def feature_dim(self):
        return self.projection.out_features

    def compute(self, x):
        features = self.encode(x)
        return self.head(features).view(x.shape[0], self.sequence_len, len(self.classes))

    def input_spec(self):
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels}


class DummyMultitaskExpert(BaseModel, ClassifierModel):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(1, 2)
        self.second = torch.nn.Linear(1, 6)

    def compute(self, x):
        features = x.mean(dim=1)
        return {"first": self.first(features).view(x.shape[0], 1, 2), "second": self.second(features).view(x.shape[0], 2, 3)}

    def input_spec(self):
        return (1, 2, 1), {"layout": "BTC", "ts_len": 2, "n_channels": 1}


def output(classes, sequence_len):
    return {"classes": classes, "sequence_len": sequence_len}


def make_nodes(trainable=False):
    return {
        "sleep": GraphNode(DummyExpert(4, 2, 3, ["wake", "sleep"], 1), {"sleep": output(["wake", "sleep"], 1)}, trainable=trainable),
        "arousal": GraphNode(DummyExpert(2, 1, 4, ["no", "yes"], 2), {"arousal": output(["no", "yes"], 2)}, trainable=trainable),
    }


@pytest.mark.parametrize("method", ["probability", "latent", "sei"])
def test_graph_classifier_routes_native_windows_and_returns_all_tasks(method):
    nodes = make_nodes()
    model = ModelGraphClassifier(
        nodes=nodes,
        method=method,
        edges=[("sleep", "arousal")],
        message_dim=2,
    )
    x = {
        "sleep": torch.arange(16, dtype=torch.float32).view(2, 4, 2),
        "arousal": torch.arange(4, dtype=torch.float32).view(2, 2, 1),
    }
    values = model(x)

    assert values["sleep"].shape == (2, 1, 2)
    assert values["arousal"].shape == (2, 2, 2)
    assert torch.equal(nodes["sleep"].model.last_input, x["sleep"])
    assert torch.equal(nodes["arousal"].model.last_input, x["arousal"])
    assert all(not parameter.requires_grad for node in nodes.values() for parameter in node.model.parameters())
    model.train()
    assert not model.models["sleep"].training
    assert not model.models["arousal"].training


def test_graph_is_acyclic_and_unknown_nodes_fail_naturally():
    with pytest.raises(ValueError, match="acyclic"):
        ModelGraphClassifier(nodes=make_nodes(), method="sei", edges=[("sleep", "arousal"), ("arousal", "sleep")])
    with pytest.raises(KeyError):
        ModelGraphClassifier(nodes=make_nodes(), method="sei", edges=[("missing", "sleep")])


def test_latent_and_sei_propagate_states_in_topological_order():
    nodes = {
        "sleep": GraphNode(DummyExpert(2, 1, 2, ["a", "b"], 1), {"sleep": output(["a", "b"], 1)}),
        "breathing": GraphNode(DummyExpert(2, 1, 3, ["a", "b"], 1), {"breathing": output(["a", "b"], 1)}),
        "arousal": GraphNode(DummyExpert(2, 1, 4, ["a", "b"], 1), {"arousal": output(["a", "b"], 1)}),
    }
    latent = ModelGraphClassifier(nodes=nodes, method="latent", edges=[("sleep", "breathing"), ("breathing", "arousal")])
    sei = ModelGraphClassifier(nodes={name: GraphNode(DummyExpert(2, 1, dim, ["a", "b"], 1), {name: output(["a", "b"], 1)}) for name, dim in [("sleep", 2), ("breathing", 3), ("arousal", 4)]}, method="sei", edges=[("sleep", "breathing"), ("breathing", "arousal")], message_dim=2)

    assert latent.heads["arousal"].in_features == 9
    assert sei.heads["arousal"].in_features == 6


def test_one_node_can_own_multiple_output_tasks():
    node = GraphNode(
        DummyExpert(2, 1, 3, ["unused", "classes"], 1),
        {
            "arousal": output(["no", "yes"], 2),
            "breathing": output(["regular", "event"], 1),
        },
    )
    model = ModelGraphClassifier(nodes={"shared": node}, method="latent")

    values = model(torch.randn(4, 2, 1))

    assert values["arousal"].shape == (4, 2, 2)
    assert values["breathing"].shape == (4, 1, 2)


def test_regular_tensor_is_broadcast_to_all_compatible_nodes():
    nodes = {
        "first": GraphNode(DummyExpert(2, 1, 3, ["no", "yes"], 1), {"first": output(["no", "yes"], 1)}),
        "second": GraphNode(DummyExpert(2, 1, 3, ["no", "yes"], 1), {"second": output(["no", "yes"], 1)}),
    }
    model = ModelGraphClassifier(nodes=nodes, method="probability")
    inputs = torch.randn(4, 2, 1)

    values = model(inputs)

    assert set(values) == {"first", "second"}
    assert torch.equal(nodes["first"].model.last_input, inputs)
    assert torch.equal(nodes["second"].model.last_input, inputs)


def test_packaged_node_loader_preserves_input_and_output_contracts(tmp_path):
    expert = DummyExpert(4, 1, 3, ["wake", "sleep"], 1)
    dataset = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=100, total_input="40ms", stride="40ms")
    package = PackagedModel(name="sleep", task="sleep", model=expert, dataset=dataset, classification_contract={"type": "single-head-multiclass", "classes": ["wake", "sleep"], "sequence_len": 1, "target_resolution": "40ms"})
    node = load_graph_node(package.save(tmp_path / "sleep"))

    assert node.outputs == {"sleep": output(["wake", "sleep"], 1)}
    assert node.native_outputs == {"sleep": output(["wake", "sleep"], 1)}
    assert not node.trainable


def test_probability_graph_uses_all_native_steps_and_has_an_independent_graph_output():
    expert = DummyExpert(4, 1, 3, ["wake", "sleep"], 3)
    node = GraphNode(expert, {"sleep": output(["wake", "sleep"], 1)}, native_outputs={"sleep": output(["wake", "sleep"], 3)})
    model = ModelGraphClassifier(nodes={"sleep": node}, method="probability")

    values = model(torch.randn(2, 4, 1))

    assert model.heads["sleep"].in_features == 6
    assert values["sleep"].shape == (2, 1, 2)


def test_probability_graph_runs_native_expert_multiple_times_and_aligns_to_graph_sequence():
    expert = DummyExpert(2, 1, 3, ["no", "yes"], 1)
    package = PackagedModel(name="event", task="event", model=expert, dataset=UnlabelledDataset(channels=[ChannelConfig("event", ["EEG"])], sample_frequency=100, total_input="20ms", stride="20ms"), classification_contract={"type": "single-head-multiclass", "classes": ["no", "yes"], "sequence_len": 1, "target_resolution": "20ms"})
    node = load_graph_node(package, graph_outputs={"event": {"labels": ["no", "yes"], "sequence_len": 5, "target_resolution": "10ms", "target_offset": "15ms"}})
    model = ModelGraphClassifier(nodes={"event": node}, method="probability")
    inputs = torch.stack([torch.full((2, 1), float(value)) for value in range(3)]).unsqueeze(0).repeat(2, 1, 1, 1)

    aligned = model.aligned_probabilities("event", inputs).view(2, 5, 2)
    outputs = model({"event": inputs})

    assert node.input_offsets == [10_000_000, 30_000_000, 50_000_000]
    assert node.prediction_indices == {"event": [0, 0, 1, 1, 2]}
    assert model.input_spec()[0]["event"] == (1, 3, 2, 1)
    assert expert.last_input.shape == (6, 2, 1)
    assert torch.equal(aligned[:, 0], aligned[:, 1])
    assert torch.equal(aligned[:, 2], aligned[:, 3])
    assert outputs["event"].shape == (2, 5, 2)


@pytest.mark.parametrize("method", ["latent", "sei"])
def test_embedding_graph_concatenates_all_scheduled_native_calls(method):
    expert = DummyExpert(2, 1, 3, ["no", "yes"], 1)
    node = GraphNode(expert, {"event": output(["no", "yes"], 1)}, input_offsets=[0, 20_000_000, 40_000_000])
    model = ModelGraphClassifier(nodes={"event": node}, method=method)
    inputs = torch.randn(2, 3, 2, 1)

    outputs = model({"event": inputs})

    assert model.feature_dims["event"] == 9
    assert model.heads["event"].in_features == 9
    assert expert.last_input.shape == (6, 2, 1)
    assert outputs["event"].shape == (2, 1, 2)


def test_scheduled_node_aligns_each_output_of_a_multitask_expert():
    dataset = UnlabelledDataset(channels=[ChannelConfig("signal", ["EEG"])], sample_frequency=100, total_input="20ms", stride="20ms")
    contract = {
        "type": "multitask",
        "tasks": {
            "first": {"classes": ["no", "yes"], "n_steps": 1, "target_resolution": "20ms", "target_offset": "0ms"},
            "second": {"classes": ["a", "b", "c"], "n_steps": 2, "target_resolution": "10ms", "target_offset": "0ms"},
        },
    }
    graph_outputs = {
        "first": {"labels": ["no", "yes"], "sequence_len": 5, "target_resolution": "10ms", "target_offset": "15ms"},
        "second": {"labels": ["a", "b", "c"], "sequence_len": 3, "target_resolution": "5ms", "target_offset": "30ms"},
    }
    package = PackagedModel(name="multitask", task="multitask", model=DummyMultitaskExpert(), dataset=dataset, classification_contract=contract)
    node = load_graph_node(package, graph_outputs=graph_outputs)
    model = ModelGraphClassifier(nodes={"shared": node}, method="probability")

    outputs = model({"shared": torch.randn(2, len(node.input_offsets), 2, 1)})

    assert set(node.prediction_indices) == {"first", "second"}
    assert outputs["first"].shape == (2, 5, 2)
    assert outputs["second"].shape == (2, 3, 3)


def test_paired_dataset_loads_every_planned_native_window():
    native = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=10, total_input="20s", stride="20s")
    reference = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=10, total_input="80s", stride="30s")
    package = PackagedModel(name="event", task="event", model=DummyExpert(200, 1, 3, ["no", "yes"], 1), dataset=native, classification_contract={"type": "single-head-multiclass", "classes": ["no", "yes"], "sequence_len": 1, "target_resolution": "20s"})
    node = load_graph_node(package, graph_outputs={"event": {"labels": ["no", "yes"], "sequence_len": 5, "target_resolution": "10s", "target_offset": "15s"}})
    dataset = PairedDataset({"event": native}, base=reference, input_offsets={"event": node.input_offsets})
    dataset.initialize([DATA / "signals_01.edf"], num_workers=1, strict=True)

    item = dataset[0]

    assert dataset.input_spec() == {"event": (1, 3, 200, 1)}
    assert item["data"]["event"].shape == (3, 200, 1)


def test_paired_dataset_applies_external_channel_aliases_to_every_node():
    native = UnlabelledDataset(channels=[ChannelConfig("eeg", ["C3-M2"])], sample_frequency=10, total_input="20s", stride="20s")
    reference = SyntheticDataset(channels=[ChannelConfig("eeg", ["EEG"])], sample_frequency=10, total_input="20s", stride="20s", event_mapping={"wake": "wake"})
    paired = PairedDataset({"sleep": native}, base=UnlabelledDataset(channels=[], sample_frequency=1, total_input="20s", stride="20s"))

    labelled = paired.with_labels(reference)

    assert labelled.datasets["sleep"].get_input_channels() == ["eeg"]
    assert labelled.datasets["sleep"].channels[0].physical_names == ["EEG"]
    assert native.channels[0].physical_names == ["C3-M2"]


def test_graph_pairs_a_base_dataset_with_each_package_input_contract():
    long_dataset = UnlabelledDataset(channels=[ChannelConfig("long", ["EEG"])], sample_frequency=100, total_input="40ms", stride="20ms")
    short_dataset = UnlabelledDataset(channels=[ChannelConfig("short", ["EMG"])], sample_frequency=50, total_input="20ms", stride="20ms")
    packages = {
        "long": PackagedModel(name="long", task="long", model=DummyExpert(4, 1, 3, ["no", "yes"], 1), dataset=long_dataset, classification_contract={"type": "single-head-multiclass", "classes": ["no", "yes"], "sequence_len": 1, "target_resolution": "20ms"}),
        "short": PackagedModel(name="short", task="short", model=DummyExpert(1, 1, 3, ["no", "yes"], 1), dataset=short_dataset, classification_contract={"type": "single-head-multiclass", "classes": ["no", "yes"], "sequence_len": 1, "target_resolution": "20ms"}),
    }

    model = ModelGraphClassifier(nodes={name: load_graph_node(package) for name, package in packages.items()}, method="probability")
    dataset = model.pair_dataset(SyntheticDataset(channels=[], sample_frequency=10, total_input="40ms", stride="10ms", event_mapping={"wake": "wake"}))
    dataset.initialize([DATA / "signals_01.edf"], num_workers=1, strict=True)
    item = dataset[0]

    assert isinstance(dataset.base, SyntheticDataset)
    assert isinstance(dataset.datasets["long"], UnlabelledDataset)
    assert isinstance(dataset.datasets["short"], UnlabelledDataset)
    assert dataset.input_spec() == {"long": (1, 4, 1), "short": (1, 1, 1)}
    assert dataset.stride == pd.Timedelta("10ms")
    assert item["data"]["long"].shape == (4, 1)
    assert item["data"]["short"].shape == (1, 1)


def test_paired_graph_uses_each_nodes_package_native_dataset(tmp_path):
    sleep_dataset = UnlabelledDataset(channels=[ChannelConfig("eeg", ["EEG"])], sample_frequency=100, total_input="40ms", stride="40ms")
    arousal_dataset = UnlabelledDataset(channels=[ChannelConfig("emg", ["EMG"])], sample_frequency=50, total_input="40ms", stride="20ms")
    nodes = {
        "sleep": GraphNode(DummyExpert(4, 1, 3, ["wake", "sleep"], 1), {"sleep": output(["wake", "sleep"], 1)}),
        "arousal": GraphNode(DummyExpert(2, 1, 3, ["no", "yes"], 1), {"arousal": output(["no", "yes"], 1)}),
    }
    dataset = pair_datasets({"sleep": sleep_dataset, "arousal": arousal_dataset})
    model = ModelGraphClassifier(nodes=nodes, method="probability", edges=[("sleep", "arousal")])
    contract = {
        "type": "multitask",
        "tasks": {
            "sleep": {"classes": ["wake", "sleep"], "n_steps": 1, "target_resolution": "40ms", "target_offset": "0ms"},
            "arousal": {"classes": ["no", "yes"], "n_steps": 1, "target_resolution": "20ms", "target_offset": "10ms"},
        },
    }
    package = PackagedModel(name="paired", task="multitask", model=model, dataset=dataset, classification_contract=contract)

    outputs = package.model({"sleep": torch.randn(3, 4, 1), "arousal": torch.randn(3, 2, 1)})
    loaded = load_packaged_model(package.save(tmp_path / "paired"))

    assert outputs["sleep"].shape == (3, 1, 2)
    assert outputs["arousal"].shape == (3, 1, 2)
    assert dataset.stride == sleep_dataset.stride
    assert loaded.model.input_spec()[1]["layout"] == "mapping"
    assert set(loaded.dataset.datasets) == {"sleep", "arousal"}


def test_paired_graph_package_predicts_an_edf_through_the_regular_api():
    sleep_dataset = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=10, total_input="60s", stride="30s")
    arousal_dataset = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=5, total_input="20s", stride="20s")
    nodes = {
        "sleep": GraphNode(DummyExpert(600, 1, 3, ["wake", "sleep"], 1), {"sleep": output(["wake", "sleep"], 1)}),
        "arousal": GraphNode(DummyExpert(100, 1, 3, ["no", "yes"], 1), {"arousal": output(["no", "yes"], 1)}),
    }
    package = PackagedModel(
        name="paired",
        task="multitask",
        model=ModelGraphClassifier(nodes=nodes, method="probability", edges=[("sleep", "arousal")]),
        dataset=pair_datasets({"sleep": sleep_dataset, "arousal": arousal_dataset}),
        classification_contract={
            "type": "multitask",
            "tasks": {
                "sleep": {"classes": ["wake", "sleep"], "n_steps": 1, "target_resolution": "60s", "target_offset": "0s"},
                "arousal": {"classes": ["no", "yes"], "n_steps": 1, "target_resolution": "20s", "target_offset": "20s"},
            },
        },
    )

    predictions = package.predict_patient(DATA / "signals_01.edf", batch_size=16)

    assert not predictions.empty
    assert set(predictions["task"]) == {"sleep", "arousal"}
    assert "prediction" in predictions.columns


def test_graph_classifier_is_a_regular_reloadable_package(tmp_path):
    model = ModelGraphClassifier(nodes=make_nodes(), method="sei", edges=[("sleep", "arousal")])
    datasets = {
        "sleep": UnlabelledDataset(channels=[ChannelConfig("eeg", ["eeg"]), ChannelConfig("eog", ["eog"])], sample_frequency=100, total_input="40ms", stride="40ms"),
        "arousal": UnlabelledDataset(channels=[ChannelConfig("emg", ["emg"])], sample_frequency=100, total_input="20ms", stride="20ms"),
    }
    dataset = pair_datasets(datasets)
    contract = {
        "type": "multitask",
        "tasks": {
            "sleep": {"classes": ["wake", "sleep"], "n_steps": 1},
            "arousal": {"classes": ["no", "yes"], "n_steps": 2},
        },
    }
    path = PackagedModel(name="graph", task="multitask", model=model, dataset=dataset, classification_contract=contract).save(tmp_path / "graph")

    loaded = load_packaged_model(path)
    outputs = loaded.model({"sleep": torch.randn(2, 4, 2), "arousal": torch.randn(2, 2, 1)})

    assert set(outputs) == {"sleep", "arousal"}
