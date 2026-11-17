"""Synchronous probability stacking for heterogeneous classifier packages."""

import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from sleepwalker.datasets.PairedDataset import PairedDataset
from sleepwalker.deployment import PackagedModel, load_packaged_model
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel


def time_delta(value):
    return value if isinstance(value, pd.Timedelta) else pd.to_timedelta(value)


def interval_grid(start, count, step):
    start_ns = int(time_delta(start).value)
    step_ns = int(time_delta(step).value)
    starts = torch.arange(count, dtype=torch.int64) * step_ns + start_ns
    return starts, starts + step_ns


def timeline_slice(starts, ends, required_start, required_end, label):
    starts = torch.as_tensor(starts, dtype=torch.int64)
    ends = torch.as_tensor(ends, dtype=torch.int64)
    if starts.ndim != 1 or len(starts) == 0 or not torch.all(starts[1:] >= ends[:-1]) or not torch.all(ends > starts):
        raise ValueError(f"{label} prediction intervals must be non-empty, ordered, and non-overlapping.")
    selected = torch.nonzero((ends > required_start) & (starts < required_end), as_tuple=False).flatten()
    if len(selected) == 0 or starts[selected[0]] > required_start or ends[selected[-1]] < required_end:
        raise ValueError(f"{label} predictions do not cover the required receiver timeline.")
    return selected


class TimeSeriesResampler(torch.nn.Module):
    """Crop and resample one prediction series onto a receiver timeline."""

    def __init__(self, source_indices, target_indices, target_length, source_repeat, crop_start, crop_length, mode):
        super().__init__()
        self.register_buffer("source_indices", source_indices, persistent=False)
        self.register_buffer("target_indices", target_indices, persistent=False)
        self.target_length = int(target_length)
        self.source_repeat = int(source_repeat)
        self.crop_start = int(crop_start)
        self.crop_length = int(crop_length)
        self.mode = str(mode)

    def forward(self, values):
        source = values.index_select(1, self.source_indices).repeat_interleave(self.source_repeat, dim=1)
        source = source[:, self.crop_start:self.crop_start + self.crop_length].transpose(1, 2)
        if source.shape[-1] != len(self.target_indices):
            source = F.interpolate(source, size=len(self.target_indices), mode=self.mode)
        return values.new_zeros(values.shape[0], self.target_length, values.shape[2]).index_copy(1, self.target_indices, source.transpose(1, 2))


class RateResampler(torch.nn.Module):
    """Resize a prediction sequence without consulting its timestamps."""

    def __init__(self, source_length, target_length):
        super().__init__()
        self.source_length = int(source_length)
        self.target_length = int(target_length)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1] != self.source_length:
            raise ValueError(f"Rate resampling expected [batch, {self.source_length}, classes], got {tuple(values.shape)}.")
        if self.source_length == self.target_length:
            return values
        mode = "nearest-exact" if self.source_length < self.target_length else "area"
        return F.interpolate(values.transpose(1, 2), size=self.target_length, mode=mode).transpose(1, 2)


def timeline_resampler(source_starts, source_ends, target_starts, target_ends, required_start, required_end):
    source_starts = torch.as_tensor(source_starts, dtype=torch.int64)
    source_ends = torch.as_tensor(source_ends, dtype=torch.int64)
    target_starts = torch.as_tensor(target_starts, dtype=torch.int64)
    target_ends = torch.as_tensor(target_ends, dtype=torch.int64)
    required_start = int(required_start)
    required_end = int(required_end)
    source_indices = timeline_slice(source_starts, source_ends, required_start, required_end, "Source")
    target_indices = timeline_slice(target_starts, target_ends, required_start, required_end, "Receiver")
    selected_source_starts = source_starts.index_select(0, source_indices)
    selected_source_ends = source_ends.index_select(0, source_indices)
    selected_target_starts = target_starts.index_select(0, target_indices)
    selected_target_ends = target_ends.index_select(0, target_indices)
    source_steps = selected_source_ends - selected_source_starts
    target_steps = selected_target_ends - selected_target_starts
    if not torch.all(source_steps == source_steps[0]) or not torch.all(target_steps == target_steps[0]) or not torch.all(selected_source_starts[1:] == selected_source_ends[:-1]) or not torch.all(selected_target_starts[1:] == selected_target_ends[:-1]):
        raise ValueError("Prediction alignment requires regular, contiguous source and receiver timelines.")
    if selected_target_starts[0] != required_start or selected_target_ends[-1] != required_end:
        raise ValueError("The required receiver span must follow complete receiver prediction intervals.")
    source_start = int(selected_source_starts[0])
    source_step = int(source_steps[0])
    target_step = int(target_steps[0])
    common_step = math.gcd(math.gcd(source_step, target_step), abs(required_start - source_start))
    source_repeat = source_step // common_step
    crop_start = (required_start - source_start) // common_step
    crop_length = (required_end - required_start) // common_step
    if crop_length % len(target_indices):
        raise ValueError("The receiver timeline is not divisible into equal prediction intervals.")
    mode = "nearest-exact" if source_step >= target_step else "area"
    return TimeSeriesResampler(source_indices, target_indices, len(target_starts), source_repeat, crop_start, crop_length, mode)


def package_contract(package):
    if not isinstance(package, PackagedModel):
        raise TypeError("Stacking experts must be PackagedModel instances.")
    contract = package.classification_contract
    if contract is None or contract.get("type") != "single-head-multiclass":
        raise ValueError("Each stacking expert must expose exactly one single-head multiclass task.")
    sequence_length = int(contract["sequence_len"])
    span = time_delta(contract["target_resolution"])
    if span.value % sequence_length:
        raise ValueError(f"Expert target span {span} is not divisible by sequence length {sequence_length}.")
    return {
        "task": str(package.task),
        "classes": list(contract["classes"]),
        "sequence_length": sequence_length,
        "step": span / sequence_length,
        "offset": time_delta(contract.get("target_offset", "0s")),
    }


def output_contract(task, output):
    required = {"labels", "sequence_len", "target_resolution"}
    missing = required - set(output)
    if missing:
        raise ValueError(f"Stacked output for '{task}' is missing {sorted(missing)}.")
    return {
        "task": task,
        "classes": list(output["labels"]),
        "sequence_length": int(output["sequence_len"]),
        "step": time_delta(output["target_resolution"]),
        "offset": time_delta(output.get("target_offset", "0s")),
    }


class AlignedPackagedClassifier(torch.nn.Module):
    """Execute one classifier package over every native call required by the stacking timeline."""

    def __init__(self, package, output):
        super().__init__()
        package = load_packaged_model(package) if isinstance(package, (str, Path)) else package
        if not isinstance(package, PackagedModel):
            raise TypeError(f"package must be a path or PackagedModel, got {type(package).__name__}.")
        if not isinstance(package.model, ClassifierModel):
            raise TypeError(f"Stacking expert '{package.name}' does not expose classification logits.")
        self.name = str(package.task)
        self.expert = package.model
        self.dataset = package.dataset
        self.native = package_contract(package)
        self.output = output_contract(self.name, output)
        if self.native["classes"] != self.output["classes"]:
            raise ValueError(f"Expert '{self.name}' changes class order between its native and stacked contracts.")

        native_span = self.native["sequence_length"] * self.native["step"]
        output_span = self.output["sequence_length"] * self.output["step"]
        if self.native["step"] == self.output["step"] and self.native["sequence_length"] >= self.output["sequence_length"]:
            first_native_step = (self.native["sequence_length"] - self.output["sequence_length"]) // 2
            self.native_anchor = self.output["offset"] - first_native_step * self.native["step"]
        else:
            self.native_anchor = self.output["offset"] + output_span / 2 - native_span / 2

        self.original_trainability = [parameter.requires_grad for parameter in self.expert.parameters()]
        self.expert_trainable = False
        self.input_offsets = []
        self.timeline_starts = torch.empty(0, dtype=torch.int64)
        self.timeline_ends = torch.empty(0, dtype=torch.int64)
        self.output_starts, self.output_ends = interval_grid(self.output["offset"], self.output["sequence_length"], self.output["step"])
        self.output_resampler = None

    def set_expert_trainable(self, trainable):
        self.expert_trainable = bool(trainable)
        for parameter, originally_trainable in zip(self.expert.parameters(), self.original_trainability):
            parameter.requires_grad_(self.expert_trainable and originally_trainable)
        if not self.expert_trainable:
            self.expert.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.expert_trainable:
            self.expert.train(mode)
        else:
            self.expert.eval()
        return self

    def configure(self, required_start, required_end, expert_trainable):
        required_start = time_delta(required_start)
        required_end = time_delta(required_end)
        if required_end <= required_start:
            raise ValueError(f"Expert '{self.name}' received an empty required prediction envelope.")
        native_span = self.native["sequence_length"] * self.native["step"]
        first_call = math.floor((required_start - self.native_anchor) / native_span)
        last_call = math.ceil((required_end - self.native_anchor) / native_span) - 1
        output_starts = [self.native_anchor + call * native_span for call in range(first_call, last_call + 1)]
        self.input_offsets = [start - self.native["offset"] for start in output_starts]
        timeline_starts = []
        timeline_ends = []
        for start in output_starts:
            starts, ends = interval_grid(start, self.native["sequence_length"], self.native["step"])
            timeline_starts.extend(starts.tolist())
            timeline_ends.extend(ends.tolist())
        self.timeline_starts = torch.tensor(timeline_starts, dtype=torch.int64)
        self.timeline_ends = torch.tensor(timeline_ends, dtype=torch.int64)
        self.output_resampler = timeline_resampler(self.timeline_starts, self.timeline_ends, self.output_starts, self.output_ends, int(self.output_starts[0]), int(self.output_ends[-1]))
        self.set_expert_trainable(expert_trainable)

    def execute(self, data):
        if self.output_resampler is None:
            raise RuntimeError(f"Expert '{self.name}' has not been configured.")
        if data.ndim < 3 or data.shape[1] != len(self.input_offsets):
            raise ValueError(f"Expert '{self.name}' expected {len(self.input_offsets)} native calls, got input shaped {tuple(data.shape)}.")
        batch_size, call_count = data.shape[:2]
        native_input = data.reshape(batch_size * call_count, *data.shape[2:])
        logits = self.expert(native_input)
        expected = (batch_size * call_count, self.native["sequence_length"], len(self.native["classes"]))
        if tuple(logits.shape) != expected:
            raise ValueError(f"Expert '{self.name}' returned logits shaped {tuple(logits.shape)}, expected {expected}.")
        logits = logits.reshape(batch_size, call_count * self.native["sequence_length"], len(self.native["classes"]))
        return {
            "probability": torch.softmax(logits, dim=-1),
            "output_logits": self.output_resampler(logits),
        }

    def input_shape(self):
        return (1, len(self.input_offsets), *self.expert.input_spec()[0][1:])


def load_aligned_classifiers(packages, *, outputs):
    experts = {}
    for name, package in packages.items():
        package = load_packaged_model(package) if isinstance(package, (str, Path)) else package
        task = package_contract(package)["task"]
        if name != task:
            raise ValueError(f"Stacking key '{name}' does not match packaged task '{task}'.")
        if task not in outputs:
            raise ValueError(f"Missing output contract for stacking task '{task}'.")
        experts[name] = AlignedPackagedClassifier(package, outputs[task])
    if set(outputs) != set(experts):
        raise ValueError(f"Stacking outputs {sorted(outputs)} do not match experts {sorted(experts)}.")
    return experts


def connection_name(source, target):
    return f"{source}__{target}"


class StackedClassifierModel(BaseModel, ClassifierModel):
    """Compose native expert predictions through task-specific probability correction heads."""

    def __init__(self, *, experts, regime, alignment="contract", context_init_std=0.005, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        if regime not in {"frozen", "task-local", "joint"}:
            raise ValueError("regime must be frozen, task-local, or joint.")
        if len(experts) < 2:
            raise ValueError("Probability stacking requires at least two experts.")
        self.regime = regime
        if alignment not in {"contract", "rate-only", "naive"}:
            raise ValueError("alignment must be contract, rate-only, or naive.")
        self.alignment = alignment
        self.context_init_std = float(context_init_std)
        if self.context_init_std < 0:
            raise ValueError("context_init_std must be non-negative.")
        self.expert_names = list(experts)
        self.experts = torch.nn.ModuleDict(experts)
        tasks = [self.experts[name].output["task"] for name in self.expert_names]
        if tasks != self.expert_names or len(set(tasks)) != len(tasks):
            raise ValueError("Every stacking expert key must equal its unique task name.")

        output_starts = [int(expert.output_starts[0]) for expert in self.experts.values()]
        output_ends = [int(expert.output_ends[-1]) for expert in self.experts.values()]
        self.required_start = min(output_starts)
        self.required_end = max(output_ends)
        for expert in self.experts.values():
            if self.alignment == "contract":
                expert.configure(self.required_start, self.required_end, expert_trainable=self.regime != "frozen")
            else:
                expert.configure(int(expert.output_starts[0]), int(expert.output_ends[-1]), expert_trainable=self.regime != "frozen")
                if self.alignment == "naive":
                    expert.input_offsets = [time_delta("0s")]
                    expert.output_resampler = RateResampler(expert.native["sequence_length"], expert.output["sequence_length"])

        self.resamplers = torch.nn.ModuleDict()
        self.correction_heads = torch.nn.ModuleDict()
        for target_name in self.expert_names:
            target = self.experts[target_name]
            context_dim = 0
            for source_name in self.expert_names:
                if source_name == target_name:
                    continue
                source = self.experts[source_name]
                context_dim += len(source.native["classes"])
                if self.alignment == "contract":
                    resampler = timeline_resampler(source.timeline_starts, source.timeline_ends, target.output_starts, target.output_ends, int(target.output_starts[0]), int(target.output_ends[-1]))
                else:
                    resampler = RateResampler(source.output["sequence_length"], target.output["sequence_length"])
                self.resamplers[connection_name(source_name, target_name)] = resampler
            head = torch.nn.Linear(context_dim, len(target.output["classes"]))
            with torch.no_grad():
                head.weight.normal_(mean=0, std=self.context_init_std)
                head.bias.zero_()
            self.correction_heads[target_name] = head

    def input_spec(self):
        shapes = {name: expert.input_shape() for name, expert in self.experts.items()}
        offsets = {name: [int(offset.value) for offset in expert.input_offsets] for name, expert in self.experts.items()}
        return shapes, {"layout": "mapping", "inputs": list(self.expert_names), "input_offsets": offsets}

    def pair_dataset(self, dataset):
        datasets = {name: expert.dataset.clone() for name, expert in self.experts.items()}
        offsets = {name: [int(offset.value) for offset in expert.input_offsets] for name, expert in self.experts.items()}
        return PairedDataset(datasets, base=dataset, input_offsets=offsets)

    def compute(self, data):
        if set(data) != set(self.expert_names):
            raise ValueError(f"Stacked model expected inputs {self.expert_names}, got {sorted(data)}.")
        native = {name: self.experts[name].execute(data[name]) for name in self.expert_names}
        outputs = {}
        for target_name in self.expert_names:
            messages = []
            for source_name in self.expert_names:
                if source_name == target_name:
                    continue
                if self.alignment == "contract":
                    probability = native[source_name]["probability"]
                else:
                    probability = torch.softmax(native[source_name]["output_logits"], dim=-1)
                if self.regime != "joint":
                    probability = probability.detach()
                messages.append(self.resamplers[connection_name(source_name, target_name)](probability))
            context = torch.cat(messages, dim=-1)
            correction = self.correction_heads[target_name](context)
            outputs[target_name] = native[target_name]["output_logits"] + correction
        return outputs

    def communication_scalars(self):
        total = 0
        for target_name in self.expert_names:
            target_steps = self.experts[target_name].output["sequence_length"]
            total += target_steps * sum(len(self.experts[source_name].native["classes"]) for source_name in self.expert_names if source_name != target_name)
        return total
