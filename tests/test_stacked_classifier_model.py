from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleepwalker.datasets.Basedataset import ChannelConfig, EDFFile
from sleepwalker.datasets.PairedDataset import PairedDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel
from sleepwalker.models.StackedClassifierModel import AlignedPackagedClassifier, RateResampler, StackedClassifierModel, timeline_resampler


DATA = Path(__file__).parent / "data"


class SpanExpert(BaseModel, ClassifierModel):
    def __init__(self, ts_len, classes, sequence_len):
        super().__init__()
        self.ts_len = ts_len
        self.classes = list(classes)
        self.sequence_len = sequence_len
        self.encoder = torch.nn.Linear(1, 4)
        self.head = torch.nn.Linear(4, sequence_len * len(classes))
        self.last_input = None

    def compute(self, x):
        self.last_input = x.detach().clone()
        features = self.encoder(x.mean(dim=1))
        return self.head(features).view(x.shape[0], self.sequence_len, len(self.classes))

    def input_spec(self):
        return (1, self.ts_len, 1), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": 1}


def package(name, model, *, span, offset="0s"):
    dataset = UnlabelledDataset(channels=[ChannelConfig(name, ["EEG"])], sample_frequency=10, total_input=f"{model.ts_len * 100}ms", stride=span)
    return PackagedModel(
        name=name,
        task=name,
        model=model,
        dataset=dataset,
        classification_contract={"type": "single-head-multiclass", "classes": model.classes, "sequence_len": model.sequence_len, "target_resolution": span, "target_offset": offset},
    )


def output(classes, sequence_len, resolution, offset="0s"):
    return {"labels": classes, "sequence_len": sequence_len, "target_resolution": resolution, "target_offset": offset}


def two_expert_stack(regime, context_init_std=0.1, alignment="contract"):
    source_model = SpanExpert(40, ["off", "on"], 4)
    target_model = SpanExpert(40, ["no", "yes"], 4)
    experts = {
        "source": AlignedPackagedClassifier(package("source", source_model, span="4s"), output(source_model.classes, 4, "1s")),
        "target": AlignedPackagedClassifier(package("target", target_model, span="4s"), output(target_model.classes, 4, "1s")),
    }
    return StackedClassifierModel(experts=experts, regime=regime, alignment=alignment, context_init_std=context_init_std), source_model, target_model


def test_temporal_alignment_uses_mean_when_downsampling():
    second = 1_000_000_000
    resampler = timeline_resampler(torch.tensor([0, 1, 2, 3]) * second, torch.tensor([1, 2, 3, 4]) * second, torch.tensor([0, 2]) * second, torch.tensor([2, 4]) * second, 0, 4 * second)

    assert torch.allclose(resampler(torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])), torch.tensor([[[1.5], [3.5]]]))


def test_temporal_alignment_uses_nearest_when_upsampling():
    second = 1_000_000_000
    resampler = timeline_resampler(torch.tensor([0, 2]) * second, torch.tensor([2, 4]) * second, torch.tensor([0, 1, 2, 3]) * second, torch.tensor([1, 2, 3, 4]) * second, 0, 4 * second)

    assert torch.equal(resampler(torch.tensor([[[1.0], [2.0]]])), torch.tensor([[[1.0], [1.0], [2.0], [2.0]]]))


def test_rate_only_alignment_ignores_timestamps_and_resizes_sequences():
    upsample = RateResampler(2, 4)
    downsample = RateResampler(4, 2)

    assert torch.equal(upsample(torch.tensor([[[1.0], [2.0]]])), torch.tensor([[[1.0], [1.0], [2.0], [2.0]]]))
    assert torch.allclose(downsample(torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])), torch.tensor([[[1.5], [3.5]]]))


def test_stack_starts_at_native_expert_predictions_with_zero_correction():
    model, source, target = two_expert_stack("frozen", context_init_std=0)
    data = {name: torch.randn(*shape) for name, shape in model.input_spec()[0].items()}

    outputs = model(data)

    assert torch.allclose(outputs["source"], source(data["source"][:, 0]))
    assert torch.allclose(outputs["target"], target(data["target"][:, 0]))


@pytest.mark.parametrize(
    "regime,source_gradient,target_gradient",
    [("frozen", False, False), ("task-local", False, True), ("joint", True, True)],
)
def test_training_regimes_route_target_loss(regime, source_gradient, target_gradient):
    model, source, target = two_expert_stack(regime)
    data = {name: torch.randn(*shape) for name, shape in model.input_spec()[0].items()}

    model(data)["target"].square().mean().backward()

    assert any(parameter.grad is not None for parameter in source.parameters()) is source_gradient
    assert any(parameter.grad is not None for parameter in target.parameters()) is target_gradient
    assert any(parameter.grad is not None for parameter in model.correction_heads["target"].parameters())


def test_complete_stack_envelope_schedules_minimum_native_calls():
    sleep = SpanExpert(300, ["wake", "sleep"], 1)
    desaturation = SpanExpert(800, ["no", "yes"], 8)
    experts = {
        "sleep": AlignedPackagedClassifier(package("sleep", sleep, span="30s", offset="25s"), output(sleep.classes, 1, "30s", "25s")),
        "desaturation": AlignedPackagedClassifier(package("desaturation", desaturation, span="80s"), output(desaturation.classes, 8, "10s")),
    }

    model = StackedClassifierModel(experts=experts, regime="frozen", context_init_std=0)
    outputs = model({name: torch.randn(*shape) for name, shape in model.input_spec()[0].items()})

    assert model.experts["sleep"].input_offsets == [pd.Timedelta("-30s"), pd.Timedelta("0s"), pd.Timedelta("30s")]
    assert tuple(sleep.last_input.shape) == (3, 300, 1)
    assert tuple(outputs["sleep"].shape) == (1, 1, 2)
    assert tuple(outputs["desaturation"].shape) == (1, 8, 2)


def test_rate_only_stack_executes_experts_for_their_own_output_spans():
    sleep = SpanExpert(300, ["wake", "sleep"], 1)
    desaturation = SpanExpert(800, ["no", "yes"], 8)
    experts = {
        "sleep": AlignedPackagedClassifier(package("sleep", sleep, span="30s", offset="25s"), output(sleep.classes, 1, "30s", "25s")),
        "desaturation": AlignedPackagedClassifier(package("desaturation", desaturation, span="80s"), output(desaturation.classes, 8, "10s")),
    }

    model = StackedClassifierModel(experts=experts, regime="frozen", alignment="rate-only", context_init_std=0)
    outputs = model({name: torch.randn(*shape) for name, shape in model.input_spec()[0].items()})

    assert model.experts["sleep"].input_offsets == [pd.Timedelta("0s")]
    assert tuple(sleep.last_input.shape) == (1, 300, 1)
    assert tuple(outputs["sleep"].shape) == (1, 1, 2)
    assert tuple(outputs["desaturation"].shape) == (1, 8, 2)


def test_naive_stack_uses_one_shared_origin_and_learns_correction_heads():
    sleep = SpanExpert(300, ["wake", "sleep"], 1)
    desaturation = SpanExpert(800, ["no", "yes"], 8)
    experts = {
        "sleep": AlignedPackagedClassifier(package("sleep", sleep, span="30s", offset="25s"), output(sleep.classes, 1, "30s", "25s")),
        "desaturation": AlignedPackagedClassifier(package("desaturation", desaturation, span="80s"), output(desaturation.classes, 8, "10s")),
    }

    model = StackedClassifierModel(experts=experts, regime="frozen", alignment="naive", context_init_std=0)
    data = {name: torch.randn(*shape) for name, shape in model.input_spec()[0].items()}
    initial = model(data)["desaturation"]
    initial.square().mean().backward()

    assert all(expert.input_offsets == [pd.Timedelta("0s")] for expert in model.experts.values())
    assert tuple(sleep.last_input.shape) == (1, 300, 1)
    assert torch.allclose(initial, desaturation(data["desaturation"][:, 0]))
    assert model.correction_heads["desaturation"].weight.grad is not None
    assert torch.count_nonzero(model.correction_heads["desaturation"].weight.grad) > 0
    assert all(parameter.grad is None for expert in (sleep, desaturation) for parameter in expert.parameters())


def test_paired_dataset_prepares_shared_signal_contract_once():
    start = pd.Timestamp("2024-01-01")
    base = UnlabelledDataset(channels=[], sample_frequency=1, total_input="4s", stride="1s")
    source = UnlabelledDataset(channels=[ChannelConfig("signal", ["EEG"])], sample_frequency=10, total_input="4s", stride="1s")
    target = UnlabelledDataset(channels=[ChannelConfig("signal", ["EEG"])], sample_frequency=10, total_input="4s", stride="2s", prepare_sample=lambda **kwargs: kwargs)
    calls = []

    def initialize(dataset, name, end):
        def run(patients, num_workers=4, *, strict=False, multiprocessing_start_method=None):
            calls.append(name)
            dataset.edf_files = [EDFFile(channels=["EEG"], path="patient.edf", start_date=start, end_date=end, length=10)]
            dataset.initialized = True
        dataset.initialize = run

    initialize(base, "base", start + pd.Timedelta("10s"))
    initialize(source, "source", start + pd.Timedelta("20s"))
    initialize(target, "target", start + pd.Timedelta("20s"))
    paired = PairedDataset({"source": source, "target": target}, base=base, input_offsets={"source": [0], "target": [0]})

    paired.initialize(["patient.edf"], num_workers=0)

    assert calls == ["base", "source"]
    assert paired.sample_keys["source"] != paired.sample_keys["target"]
    assert paired.component_files["source"]["patient.edf"] is paired.component_files["target"]["patient.edf"]


def test_paired_dataset_loads_overlapping_identical_samples_once():
    start = pd.Timestamp("2024-01-01")
    base = UnlabelledDataset(channels=[], sample_frequency=1, total_input="4s", stride="1s")
    source = UnlabelledDataset(channels=[ChannelConfig("signal", ["EEG"])], sample_frequency=10, total_input="4s", stride="1s")
    target = UnlabelledDataset(channels=[ChannelConfig("signal", ["EEG"])], sample_frequency=10, total_input="4s", stride="2s")
    file = EDFFile(channels=["EEG"], path="patient.edf", start_date=start, end_date=start + pd.Timedelta("20s"), length=10)
    calls = []

    base.get_target_item = lambda file, timestamp: {"patient": file.path, "time": timestamp}
    source.get_items = lambda file, timestamps: calls.append(list(timestamps)) or [{"patient": file.path, "time": timestamp, "data": torch.tensor([float(index)])} for index, timestamp in enumerate(timestamps)]
    target.get_items = lambda file, timestamps: (_ for _ in ()).throw(AssertionError("duplicate input contract was loaded twice"))
    paired = PairedDataset({"source": source, "target": target}, base=base, input_offsets={"source": [0, int(pd.Timedelta("1s").value)], "target": [int(pd.Timedelta("1s").value)]})
    paired.component_files = {"source": {"patient.edf": file}, "target": {"patient.edf": file}}

    item = paired.get_item(file, start)

    assert calls == [[start, start + pd.Timedelta("1s")]]
    assert torch.equal(item["data"]["source"], torch.tensor([[0.0], [1.0]]))
    assert torch.equal(item["data"]["target"], torch.tensor([[1.0]]))


def test_paired_dataset_batch_fetch_shares_patient_envelope():
    start = pd.Timestamp("2024-01-01")
    base = UnlabelledDataset(channels=[], sample_frequency=1, total_input="4s", stride="1s")
    source = UnlabelledDataset(channels=[ChannelConfig("signal", ["EEG"])], sample_frequency=10, total_input="4s", stride="1s")
    target = UnlabelledDataset(channels=[ChannelConfig("signal", ["EEG"])], sample_frequency=10, total_input="4s", stride="2s")
    file = EDFFile(channels=["EEG"], path="patient.edf", start_date=start, end_date=start + pd.Timedelta("20s"), start_offsets=torch.tensor([0, 1]).numpy(), length=2)
    calls = []

    base.get_target_item = lambda file, timestamp: {"patient": file.path, "time": timestamp}
    source.get_items = lambda file, timestamps: calls.append(list(timestamps)) or [{"patient": file.path, "time": timestamp, "data": torch.tensor([float(index)])} for index, timestamp in enumerate(timestamps)]
    target.get_items = lambda file, timestamps: (_ for _ in ()).throw(AssertionError("duplicate input contract was loaded twice"))
    paired = PairedDataset({"source": source, "target": target}, base=base, input_offsets={"source": [0, int(pd.Timedelta("1s").value)], "target": [int(pd.Timedelta("1s").value)]})
    paired.component_files = {"source": {"patient.edf": file}, "target": {"patient.edf": file}}
    paired.edf_files = [file]
    paired.lower_bounds = [0]
    paired.upper_bounds = [2]
    paired.rejection_strategy = "none"
    paired.initialized = True

    items = paired.__getitems__([0, 1])

    assert calls == [[start, start + pd.Timedelta("1s"), start + pd.Timedelta("2s")]]
    assert torch.equal(items[0]["data"]["source"], torch.tensor([[0.0], [1.0]]))
    assert torch.equal(items[0]["data"]["target"], torch.tensor([[1.0]]))
    assert torch.equal(items[1]["data"]["source"], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(items[1]["data"]["target"], torch.tensor([[2.0]]))


def test_stacked_package_round_trip(tmp_path):
    model, _, _ = two_expert_stack("frozen", context_init_std=0)
    dataset = model.pair_dataset(UnlabelledDataset(channels=[], sample_frequency=1, total_input="4s", stride="4s"))
    contract = {"type": "multitask", "tasks": {name: {"classes": model.experts[name].output["classes"], "n_steps": 4, "target_resolution": "1s", "target_offset": "0s"} for name in model.expert_names}}
    path = PackagedModel(name="stack", task="multitask", model=model, dataset=dataset, classification_contract=contract).save(tmp_path / "stack")

    loaded = load_packaged_model(path)
    outputs = loaded.model({name: torch.randn(*shape) for name, shape in loaded.model.input_spec()[0].items()})

    assert set(outputs) == {"source", "target"}


