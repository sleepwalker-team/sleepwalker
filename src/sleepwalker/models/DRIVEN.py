"""Independent temporal-image EfficientNetV2-S branches for DRIVEN.

Original repository: https://github.com/LCSB-SCG/DRIVEN
Topology: https://github.com/LCSB-SCG/DRIVEN/blob/dce8bfadd05d249ca7b55fa5d2d1974b52ba35ec/python_code/training/train_nn.py
Association: https://github.com/LCSB-SCG/DRIVEN/blob/dce8bfadd05d249ca7b55fa5d2d1974b52ba35ec/python_code/preliminary/label_ahi.py

This independent PyTorch implementation uses
torchvision's stage topology with a one-channel stem, dynamic SAME padding,
global pooling and two logits. It does not convert the author's Keras weights.
"""

import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_v2_s

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel
from sleepwalker.models.preprocessors.DelayedChannelSlices import DelayedChannelSlices, integral_samples
from sleepwalker.models.preprocessors.ScaleChannels import ScaleChannels
from sleepwalker.datasets.events import validate_event_table
from sleepwalker.training.callbacks import map_events
from sleepwalker.trainer.utils.targets import normalize_multitask_config, passes_filters, prepare_multitask_target, slice_target_interval


class SamePadConv2d(nn.Conv2d):
    """TensorFlow SAME geometry, including asymmetric strided padding."""

    def forward(self, x):
        padding = []
        for length, kernel, stride, dilation in zip(reversed(x.shape[-2:]), reversed(self.kernel_size), reversed(self.stride), reversed(self.dilation)):
            total = max((math.ceil(length / stride) - 1) * stride + dilation * (kernel - 1) + 1 - length, 0)
            padding.extend([total // 2, total - total // 2])
        return F.conv2d(F.pad(x, padding), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def use_same_padding(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
            replacement = SamePadConv2d(child.in_channels, child.out_channels, child.kernel_size, stride=child.stride, dilation=child.dilation, groups=child.groups, bias=child.bias is not None)
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            use_same_padding(child)


class EfficientNetV2Branch(nn.Module):
    def __init__(self, dropout=0.2, image_scale=1 / 128, image_offset=-1):
        super().__init__()
        network = efficientnet_v2_s(weights=None, dropout=dropout)
        network.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(network.features[0][0].weight, mode="fan_out")
        self.features = network.features
        use_same_padding(self.features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(1280, 2)
        self.image_scale = float(image_scale)
        self.image_offset = float(image_offset)

    def forward(self, signal):
        image = signal[:, None, :, None] * self.image_scale + self.image_offset
        pooled = self.pool(self.features(image)).flatten(1)
        return self.head(self.dropout(pooled)), pooled


class DRIVEN(BaseModel, ClassifierModel, EmbeddingModel):
    """Named branch logits and stable concatenated probability/pooled features.

    Defaults accept [B,3520,2] (ABD, physical-percent SpO2) at 64 Hz. An explicit
    preprocessor list replaces the default ScaleChannels/DelayedChannelSlices
    pipeline. With [], supply already-sliced [B,slice_samples,n_branches] input
    and declare ts_len/n_channels accordingly. No missing context is padded.
    """

    def __init__(self, ts_len=3520, n_channels=2, sampling_frequency=64, window_duration=30, delays=(0, 10, 15, 20, 25), task_config=None, preprocessors=None, feature_variant="probabilities", input_channels=None, dropout=0.2, image_scale=1 / 128, image_offset=-1):
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in [sampling_frequency, window_duration]):
            raise ValueError("Sampling frequency and window duration must be finite and positive.")
        if int(ts_len) != ts_len or int(n_channels) != n_channels:
            raise ValueError("Input dimensions must be integers.")
        if not math.isfinite(float(image_scale)) or not math.isfinite(float(image_offset)):
            raise ValueError("Image rescaling parameters must be finite.")
        delays = tuple(float(delay) for delay in delays)
        if any(not math.isfinite(delay) or delay < 0 for delay in delays):
            raise ValueError("Delays must be finite and nonnegative.")
        names = ["abd", *(f"spo2_{delay:g}".replace(".", "p") for delay in delays)]
        if len(names) != len(set(names)):
            raise ValueError("DRIVEN branch names require distinct SpO2 delays.")
        if not delays:
            raise ValueError("Configure at least one SpO2 delay.")
        if preprocessors is None:
            preprocessors = [ScaleChannels(scales={1: 0.01}, clip={1: [0, 1]}), DelayedChannelSlices(sampling_frequency=sampling_frequency, window_duration=window_duration, slices=[(0, 0), *((1, delay) for delay in delays)])]
            if n_channels != 2 or ts_len < integral_samples(window_duration + max(delays), sampling_frequency, name="input duration"):
                raise ValueError("Default DRIVEN preprocessing requires two channels and enough delayed context.")
        super().__init__(preprocessors)
        if not delays or feature_variant not in {"probabilities", "pooled"}:
            raise ValueError("Configure at least one SpO2 delay and feature_variant='probabilities' or 'pooled'.")
        self.ts_len, self.n_channels = int(ts_len), int(n_channels)
        self.sampling_frequency = float(sampling_frequency)
        self.slice_samples = integral_samples(window_duration, sampling_frequency, name="window_duration")
        if self.slice_samples < 32 or min(self.ts_len, self.n_channels) < 1:
            raise ValueError("Temporal images need at least 32 samples and positive input dimensions.")
        self.input_channels = None if input_channels is None else list(input_channels)
        self.branch_names = names if task_config is None else list(task_config)
        if len(self.branch_names) != len(names):
            raise ValueError("Declare exactly one task per configured branch, in slice order.")
        self.task_config = normalize_multitask_config(task_config or {name: {"labels": ["regular", "respiratory_event"], "default": "regular", "target_resolution": f"{window_duration}s", "sequence_len": 1, "percentage": 1 / self.slice_samples, "loss_function": F.cross_entropy} for name in names})
        labels = [list(spec["labels"]) for spec in self.task_config.values()]
        if any(len(value) != 2 or value != labels[0] for value in labels) or any(spec["n_steps"] != 1 for spec in self.task_config.values()):
            raise ValueError("Every DRIVEN branch needs the same two labels and one target per window.")
        self.classes = labels[0]
        self.feature_variant = feature_variant
        self.branches = nn.ModuleDict({name: EfficientNetV2Branch(dropout, image_scale, image_offset) for name in self.branch_names})

    def branch_outputs(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.slice_samples, len(self.branch_names)) or not torch.isfinite(x).all():
            raise ValueError(f"DRIVEN's processed input must be finite [B,{self.slice_samples},{len(self.branch_names)}].")
        logits, pooled = {}, []
        for index, name in enumerate(self.branch_names):
            branch_logits, branch_features = self.branches[name](x[:, :, index])
            logits[name] = branch_logits.unsqueeze(1)
            pooled.append(branch_features)
        return logits, pooled

    def compute(self, x):
        return self.branch_outputs(x)[0]

    def encode(self, x):
        logits, pooled = self.branch_outputs(x)
        if self.feature_variant == "probabilities":
            return torch.cat([logits[name][:, 0].softmax(dim=-1) for name in self.branch_names], dim=1)
        return torch.cat(pooled, dim=1)

    def feature_dim(self):
        return len(self.branch_names) * (2 if self.feature_variant == "probabilities" else 1280)

    def input_spec(self):
        meta = {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels, "sampling_frequency": self.sampling_frequency, "feature_variant": self.feature_variant, "feature_order": list(self.branch_names), "feature_classes": list(self.classes), "feature_dim": self.feature_dim()}
        if self.input_channels is not None:
            meta["input_channels"] = self.input_channels
        return (1, self.ts_len, self.n_channels), meta


def associate_hypopneas(events, *, desaturation_min_drop_pp=3.0, desaturation_after_seconds=45, arousal_after_seconds=5):
    """Apply DRIVEN's desaturation-or-arousal rule to each hypopnea.

    Process events in onset order. Remember the latest hypopnea until an
    arousal or sufficiently large desaturation confirms it, or another hypopnea
    replaces it. Desaturations must start by hypopnea end +45 s and arousals by
    end +5 s (configurable, inclusive). Evidence is used once. A missing drop
    within that window makes the hypopnea unknown unless positive evidence later
    confirms it. Equal onsets retain source order; other events are preserved.
    """
    validate_event_table(events)
    if "desaturation_drop_pp" not in events:
        raise ValueError("Hypopnea association requires desaturation_drop_pp; the source provided no drop field.")
    for value in [desaturation_min_drop_pp, desaturation_after_seconds, arousal_after_seconds]:
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("Association thresholds and delays must be finite and nonnegative.")
    result = events.copy()
    drops = pd.to_numeric(events["desaturation_drop_pp"], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    finite = drops[~np.isnan(drops)]
    if not np.isfinite(finite).all() or np.any(finite < 0):
        raise ValueError("Desaturation drops must be finite nonnegative percentage-point measurements or missing.")
    labels = events["Label"].to_numpy(copy=True)
    starts = pd.DatetimeIndex(events["Starttime"])
    ends = pd.DatetimeIndex(events["Endtime"])
    pending = None
    for index in np.argsort(starts.asi8, kind="stable"):
        label = labels[index]
        if label == "hypopnea":
            result.iloc[index, result.columns.get_loc("Label")] = "hypopnea_unassociated"
            pending = index
        elif pending is not None and label == "desaturation":
            if starts[pending] <= starts[index] <= ends[pending] + pd.Timedelta(seconds=desaturation_after_seconds):
                if np.isnan(drops[index]):
                    result.iloc[pending, result.columns.get_loc("Label")] = "hypopnea_unknown"
                elif drops[index] >= desaturation_min_drop_pp:
                    result.iloc[pending, result.columns.get_loc("Label")] = "hypopnea_associated"
                    pending = None
        elif pending is not None and label == "arousal":
            if starts[pending] <= starts[index] <= ends[pending] + pd.Timedelta(seconds=arousal_after_seconds):
                result.iloc[pending, result.columns.get_loc("Label")] = "hypopnea_associated"
                pending = None
    return result


def prepare_events(label_df, label_extra_df, *, mapping, remove_unmapped=True, on_missing_field="error", **association_options):
    """Associate and map each annotator independently, retaining unknown intervals."""
    if on_missing_field not in {"error", "reject"}:
        raise ValueError("on_missing_field must be 'error' or 'reject'.")
    tables = [label_df] + ([] if label_extra_df is None else [label_extra_df])
    if on_missing_field == "reject" and any("desaturation_drop_pp" not in table for table in tables):
        return None
    transformed = [associate_hypopneas(table, **association_options) for table in tables]
    return map_events(transformed[0], transformed[1] if len(transformed) == 2 else None, mapping=mapping, remove_unmapped=remove_unmapped)


def prepare_target(target, target_extra=None, patient=None, time=None, *, task_config, class_cnts=None, annotation_labels=None, allow_partial=False):
    """Supervise DRIVEN's first 30 s only during sleep without unknown hypopneas.

    The remaining 25 s supplies future SpO2 context. This eligibility rule is
    applied before the generic multitask target builder, identically for all
    six heads; it is independent of tensor/sample formatting.
    """
    selected = slice_target_interval(target, target_resolution="30s", target_offset="0s")
    filters = [{"columns": ["sleep"], "percentage": 1.0, "mode": "min"},
               {"columns": ["hypopnea_unknown"], "percentage": 0.0, "mode": "max"}]
    if not passes_filters(selected, filters):
        return None
    if target_extra is not None:
        extra = slice_target_interval(target_extra, target_resolution="30s", target_offset="0s")
        if not passes_filters(extra, filters):
            target_extra = None
    return prepare_multitask_target(target, target_extra, patient, time, class_cnts=class_cnts, task_config=task_config, annotation_labels=annotation_labels, allow_partial=allow_partial)
