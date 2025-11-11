import os

from cosy.specification_builder import SpecificationBuilder
from cosy.types import Constructor, Group, DataGroup, Literal, Type, Var
from cosy.synthesizer import Synthesizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleepwalker.models.preprocessors.ChannelSampler import ChannelSampler
from sleepwalker.models.preprocessors.Crop import Crop
from sleepwalker.models.preprocessors.EmpiricalClipScaler import EmpiricalClipScaler
from sleepwalker.models.preprocessors.FIR import FIR
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.models.preprocessors.Spectogram import Spectogram
from sleepwalker.models.preprocessors.ZNormalize import ZNormalize
from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets.Basedataset import ChannelConfig

from torchinfo import summary
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split, random_split
from sleepwalker.datasets import Ruhrlandklinik

from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger, MlflowSink

DSL = SpecificationBuilder # because Andrej and Conny need to rename everything every few weeks and make implementing against CoSy really annoying -.-


class ChannelWiseNormalization(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ChannelWiseNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x has shape (batch_size, num_channels, time_steps)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_scaled = self.gamma.view(1, -1, 1) * x_normalized + self.beta.view(1, -1, 1)
        return x_scaled


class Conv1dLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(Conv1dLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Permute to [batch_size, length, channels] for LayerNorm
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        # Permute back to [batch_size, channels, length]
        x = x.permute(0, 2, 1)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )

        # Pointwise (1x1) convolution: mixes channels
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class UTime(BaseModel):
    def __init__(self, forward_function, classes, n_channels, modules, preprocessors=None):
        super(UTime, self).__init__(preprocessors=preprocessors)
        self.forward_function = forward_function
        self.classes = list(classes)
        self.nchannel = n_channels
        self.nclass = len(classes)
        name = "module"
        for i, m in enumerate(modules):
            self.add_module(name + str(i), m)

    def _forward(self, x):
        return self.forward_function(x)



class UtimeRepository:
    def __init__(self, dimension_choices, normalization_eps_choices, #normalization_momentum_choices,
                 dropout_p_choices,
                 convolution_kernel_size_choices, convolution_stride_choices, convolution_padding_choices,
                 convolution_dilations_choices,
                 maxpool_size_choices, maxpool_stride_choices, maxpool_padding_choices, maxpool_dilation_choices,
                 preprocessor_channel_sampler_n_choices,
                 preprocessor_crop_total_input_choices, preprocessor_crop_sampling_rate_choices,
                 preprocessor_empirical_clip_scaler_q_choices, preprocessor_empirical_clip_scaler_scale_choices,
                 preprocessor_fir_sampling_rate_choices, preprocessor_fir_channels_choices, preprocessor_fir_filter_params_choices,
                 preprocessor_robust_scaler_lower_quantile_choices, preprocessor_robust_scaler_upper_quantile_choices,
                 preprocessor_spectogram_n_fft_choices, preprocessor_spectogram_hop_length_choices,
                 preprocessor_spectogram_win_length_choices, preprocessor_spectogram_epoch_len_samples_choices,
                 n_samples,
                 abc_channel_choices, abc_event_mapping, abc_num_workers, abc_sample_frequency, abc_total_input,
                 abc_target_resolution,
                 batch_size,
                 optimizer_learning_rate, optimizer_learning_rate_decay, optimizer_weight_decay, optimizer_eps,
                 optimizer_beta, optimizer_initial_accumulator_value, optimizer_momentum, optimizer_dampening,
                 lr_scheduler_start_factor_choices, lr_scheduler_end_factor_choices, lr_scheduler_total_iters_choices,
                 lr_scheduler_step_size_choices, lr_scheduler_gamma_choices, lr_scheduler_last_epoch_choices
                 ):
        self.dimension_choices = dimension_choices + [x for x in map(lambda x: x * 2, dimension_choices) if x not in dimension_choices]
        self.normalization_eps_choices = normalization_eps_choices
        #self.normalization_momentum_choices = normalization_momentum_choices
        self.dropout_p_choices = dropout_p_choices
        self.convolution_kernel_size_choices = convolution_kernel_size_choices
        self.convolution_stride_choices = convolution_stride_choices
        self.convolution_padding_choices = convolution_padding_choices
        self.convolution_dilations_choices = convolution_dilations_choices
        self.maxpool_size_choices = maxpool_size_choices
        self.maxpool_stride_choices = maxpool_stride_choices
        self.maxpool_padding_choices = maxpool_padding_choices
        self.maxpool_dilation_choices = maxpool_dilation_choices
        self.preprocessor_channel_sampler_n_choices = preprocessor_channel_sampler_n_choices
        self.preprocessor_crop_total_input_choices = preprocessor_crop_total_input_choices
        self.preprocessor_crop_sampling_rate_choices = preprocessor_crop_sampling_rate_choices
        self.preprocessor_empirical_clip_scaler_q_choices = preprocessor_empirical_clip_scaler_q_choices
        self.preprocessor_empirical_clip_scaler_scale_choices = preprocessor_empirical_clip_scaler_scale_choices
        self.preprocessor_fir_sampling_rate_choices = preprocessor_fir_sampling_rate_choices
        self.preprocessor_fir_channels_choices = preprocessor_fir_channels_choices
        self.preprocessor_fir_filter_params_choices = preprocessor_fir_filter_params_choices
        self.preprocessor_robust_scaler_lower_quantile_choices = preprocessor_robust_scaler_lower_quantile_choices
        self.preprocessor_robust_scaler_upper_quantile_choices = preprocessor_robust_scaler_upper_quantile_choices
        self.preprocessor_spectogram_n_fft_choices = preprocessor_spectogram_n_fft_choices
        self.preprocessor_spectogram_hop_length_choices = preprocessor_spectogram_hop_length_choices
        self.preprocessor_spectogram_win_length_choices = preprocessor_spectogram_win_length_choices
        self.preprocessor_spectogram_epoch_len_samples_choices = preprocessor_spectogram_epoch_len_samples_choices
        self.n_samples = n_samples
        self.abc_channel_choices = abc_channel_choices
        self.abc_event_mapping = abc_event_mapping
        self.abc_num_workers = abc_num_workers
        self.abc_sample_frequency = abc_sample_frequency
        self.abc_total_input = abc_total_input
        self.abc_target_resolution = abc_target_resolution
        self.batch_size = batch_size
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_learning_rate_decay = optimizer_learning_rate_decay
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_eps = optimizer_eps
        self.optimizer_beta = optimizer_beta
        self.optimizer_initial_accumulator_value = optimizer_initial_accumulator_value
        self.optimizer_momentum = optimizer_momentum
        self.optimizer_dampening = optimizer_dampening
        self.lr_scheduler_start_factor_choices = lr_scheduler_start_factor_choices
        self.lr_scheduler_end_factor_choices = lr_scheduler_end_factor_choices
        self.lr_scheduler_total_iters_choices = lr_scheduler_total_iters_choices
        self.lr_scheduler_step_size_choices = lr_scheduler_step_size_choices
        self.lr_scheduler_gamma_choices = lr_scheduler_gamma_choices
        self.lr_scheduler_last_epoch_choices = lr_scheduler_last_epoch_choices

        # Is this really necessary? With our request language, the user has to ensure this himself...
        if 1 not in self.convolution_kernel_size_choices:
            self.convolution_kernel_size_choices.append(1)

        """
        # Parameters that are optional in request language need to have None as a choice
        self.normalization_eps_choices.append(None)
        #self.normalization_momentum_choices.append(None)
        self.dropout_p_choices.append(None)
        self.convolution_kernel_size_choices.append(None)
        self.convolution_stride_choices.append(None)
        self.convolution_padding_choices.append(None)
        self.convolution_dilations_choices.append(None)
        self.maxpool_size_choices.append(None)
        self.maxpool_stride_choices.append(None)
        self.maxpool_padding_choices.append(None)
        self.maxpool_dilation_choices.append(None)
        self.dimension_choices.append(None)
        """

        self.convs = ["simple_convolution", "depthwise_separable_convolution"] #, None]

        self.afs = ["ReLu", "ELU", "Tanh"] #, None]

        self.norms = ["batch_norm", "conv1d_layer_norm", "channel_wise_norm"] #, None]

        self.losses = ["BCE_with_logits", "CrossEntropy", "MAE", "MSE"]


    class Nat(Group):
        name = "Nat"

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    class Nat_Tuple(Group):
        name = "Nat_Tuple"

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(isinstance(v, int) and v >= 0 for v in value)

    class Conv_Tuple(Group):
        name = "Conv_Tuple"

        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(v in self.conv_choices for v in value)

    class Conv_Tuple_Tuple(Group):
        name = "Conv_Tuple_Tuple"

        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and
                    all(isinstance(v, tuple) and
                        all(c in self.conv_choices for c in v) for v in value))

    class AF_Tuple(Group):
        name = "AF_Tuple"

        def __init__(self, af_choices):
            self.af_choices = af_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(v in self.af_choices for v in value)

    class Norm_Tuple(Group):
        name = "Norm_Tuple"

        def __init__(self, norm_choices):
            self.norm_choices = norm_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(v in self.norm_choices for v in value)

    class Dropout_Tuple(Group):
        name = "Dropout_Tuple"

        def __init__(self, dropout_p_choices):
            self.dropout_p_choices = dropout_p_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(v in self.dropout_p_choices for v in value)

    class Kernel_Size_Tuple(Group):
        name = "Kernel_Size_Tuple"

        def __init__(self, kernel_size_choices):
            self.kernel_size_choices = kernel_size_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(v in self.kernel_size_choices for v in value)

    class Maxpool_Size_Tuple(Group):
        name = "Maxpool_Size_Tuple"

        def __init__(self, maxpool_size_choices):
            self.maxpool_size_choices = maxpool_size_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(v in self.maxpool_size_choices for v in value)

    class Dimension_Tuple(Group):
        name = "Dimension_Tuple"

        def __init__(self, dimension_choices):
            self.dimension_choices = dimension_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and all(v in self.dimension_choices for v in value))

    class Maybe_Nat(Group):
        name = "Maybe_Nat"

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, int) and value >= 0)

    class Maybe_Nat_Tuple(Group):
        name = "Maybe_Nat_Tuple"

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else isinstance(v, int) and v >= 0 for v in value)

    class Maybe_Conv_Tuple(Group):
        name = "Maybe_Conv_Tuple"

        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.conv_choices for v in value)

    class Maybe_Conv_Tuple_Tuple(Group):
        name = "Maybe_Conv_Tuple_Tuple"

        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and
                    all(isinstance(v, tuple) and
                        all(True if c is None else c in self.conv_choices for c in v) for v in value))

    class Maybe_AF_Tuple(Group):
        name = "Maybe_AF_Tuple"

        def __init__(self, af_choices):
            self.af_choices = af_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.af_choices for v in value)

    class Maybe_Norm_Tuple(Group):
        name = "Maybe_Norm_Tuple"

        def __init__(self, norm_choices):
            self.norm_choices = norm_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.norm_choices for v in value)

    class Maybe_Dropout_Tuple(Group):
        name = "Maybe_Dropout_Tuple"

        def __init__(self, dropout_p_choices):
            self.dropout_p_choices = dropout_p_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.dropout_p_choices for v in value)

    class Maybe_Kernel_Size_Tuple(Group):
        name = "Maybe_Kernel_Size_Tuple"

        def __init__(self, kernel_size_choices):
            self.kernel_size_choices = kernel_size_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.kernel_size_choices for v in value)

    class Maybe_Maxpool_Size_Tuple(Group):
        name = "Maybe_Maxpool_Size_Tuple"

        def __init__(self, maxpool_size_choices):
            self.maxpool_size_choices = maxpool_size_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.maxpool_size_choices for v in value)

    class Maybe_Dimension_Tuple(Group):
        name = "Maybe_Dimension_Tuple"

        def __init__(self, dimension_choices):
            self.dimension_choices = dimension_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and all(True if v is None else v in self.dimension_choices for v in value))

    class Preprocessor(Group):
        name = "Preprocessor"

        """
("ChannelSampler", v["n"])
or
("Crop", v["total_input"], v["sampling_rate"], v["where"])
or
("EmpiricalClipScaler", v["q"], v["scale"])
or
("FIR", v["sampling_rate"], v["channels"], v["filter_params"], v["zero_phase"])
or
("Normalize",)
or
("RobustScaler", v["lower_quantile"], v["upper_quantile"])
or
("Spectogram", v["n_fft"], v["hop_length"], v["win_length"], v["epoch_len_samples"])
or
("ZNormalize", v["use_global_statistics"])
        """
        def __init__(self, channel_sampler_n_choices, crop_total_input_choices, crop_sampling_rate_choices,
                     empirical_clip_scaler_q_choices, empirical_clip_scaler_scale_choices,
                     fir_sampling_rate_choices, fir_channels_choices, fir_filter_params_choices,
                     robust_scaler_lower_quantile_choices, robust_scaler_upper_quantile_choices, spectogram_n_fft_choices,
                     spectogram_hop_length_choices, spectogram_win_length_choices, spectogram_epoch_len_samples_choices):
            self.channel_sampler_n_choices = channel_sampler_n_choices
            self.crop_total_input_choices = crop_total_input_choices
            self.crop_sampling_rate_choices = crop_sampling_rate_choices
            self.empirical_clip_scaler_q_choices = empirical_clip_scaler_q_choices
            self.empirical_clip_scaler_scale_choices = empirical_clip_scaler_scale_choices
            self.fir_sampling_rate_choices = fir_sampling_rate_choices
            self.fir_channels_choices = fir_channels_choices
            self.fir_filter_params_choices = fir_filter_params_choices
            self.robust_scaler_lower_quantile_choices = robust_scaler_lower_quantile_choices
            self.robust_scaler_upper_quantile_choices = robust_scaler_upper_quantile_choices
            self.spectogram_n_fft_choices = spectogram_n_fft_choices
            self.spectogram_hop_length_choices = spectogram_hop_length_choices
            self.spectogram_win_length_choices = spectogram_win_length_choices
            self.spectogram_epoch_len_samples_choices = spectogram_epoch_len_samples_choices

        def iter_channel_sampler(self):
            for n in self.channel_sampler_n_choices:
                yield ("ChannelSampler", n)

        def iter_crop(self):
            for total_input in self.crop_total_input_choices:
                for sampling_rate in self.crop_sampling_rate_choices:
                    for where in ["left", "middle", "right"]:
                        yield ("Crop", total_input, sampling_rate, where)

        def iter_empirical_clip_scaler(self):
            for q in self.empirical_clip_scaler_q_choices:
                for scale in self.empirical_clip_scaler_scale_choices:
                    yield ("EmpiricalClipScaler", q, scale)

        def iter_fir(self):
            for sampling_rate in self.fir_sampling_rate_choices:
                for channels in self.fir_channels_choices:
                    for filter_params in self.fir_filter_params_choices:
                        for zero_phase in [True, False]:
                            yield ("FIR", sampling_rate, channels, filter_params, zero_phase)

        def iter_normalize(self):
            yield ("Normalize",)

        def iter_robust_scaler(self):
            for lower_quantile in self.robust_scaler_lower_quantile_choices:
                for upper_quantile in self.robust_scaler_upper_quantile_choices:
                    yield ("RobustScaler", lower_quantile, upper_quantile)

        def iter_spectogram(self):
            for n_fft in self.spectogram_n_fft_choices:
                for hop_length in self.spectogram_hop_length_choices:
                    for win_length in self.spectogram_win_length_choices:
                        for epoch_len_samples in self.spectogram_epoch_len_samples_choices:
                            yield ("Spectogram", n_fft, hop_length, win_length, epoch_len_samples)

        def iter_znormalize(self):
            for use_global_statistics in [True, False]:
                yield ("ZNormalize", use_global_statistics)

        def __iter__(self):
            yield from self.iter_channel_sampler()
            yield from self.iter_crop()
            yield from self.iter_empirical_clip_scaler()
            yield from self.iter_fir()
            yield from self.iter_normalize()
            yield from self.iter_robust_scaler()
            yield from self.iter_spectogram()
            yield from self.iter_znormalize()

        def __contains__(self, value: object) -> bool:
            if (isinstance(value, tuple)):
                if value[0] == "ChannelSampler":
                    return len(value) == 2 and value[1] in self.channel_sampler_n_choices
                elif value[0] == "Crop":
                    return (len(value) == 4 and
                            value[1] in self.crop_total_input_choices and
                            value[2] in self.crop_sampling_rate_choices and
                            value[3] in ["left", "middle", "right"])
                elif value[0] == "EmpiricalClipScaler":
                    return (len(value) == 3 and
                            value[1] in self.empirical_clip_scaler_q_choices and
                            value[2] in self.empirical_clip_scaler_scale_choices)
                elif value[0] == "FIR":
                    return (len(value) == 5 and
                            value[1] in self.fir_sampling_rate_choices and
                            value[2] in self.fir_channels_choices and
                            value[3] in self.fir_filter_params_choices and
                            isinstance(value[4], bool))
                elif value[0] == "Normalize":
                    return len(value) == 1
                elif value[0] == "RobustScaler":
                    return (len(value) == 3 and
                            value[1] in self.robust_scaler_lower_quantile_choices and
                            value[2] in self.robust_scaler_upper_quantile_choices)
                elif value[0] == "Spectogram":
                    return (len(value) == 5 and
                            value[1] in self.spectogram_n_fft_choices and
                            value[2] in self.spectogram_hop_length_choices and
                            value[3] in self.spectogram_win_length_choices and
                            value[4] in self.spectogram_epoch_len_samples_choices)
                elif value[0] == "ZNormalize":
                    return len(value) == 2 and isinstance(value[1], bool)
                else:
                    return False
            else:
                return False

        def from_maybe_preprocessor(self, maybe_preprocessor):
            name = maybe_preprocessor[0]
            result = [(name,)]
            for i, p in enumerate(maybe_preprocessor[1:]):
                j = i + 1
                if p is not None:
                    old_result = result
                    result = []
                    for t in old_result:
                        result.append(t + (p,))
                else:
                    old_result = result
                    result = []
                    for t in old_result:
                        if name == "ChannelSampler":
                            if j == 1:
                                for n in self.channel_sampler_n_choices:
                                    result.append(t + (n,))
                            else:
                                raise ValueError("Unexpected index in ChannelSampler preprocessor")
                        elif name == "Crop":
                            if j == 1:
                                for total_input in self.crop_total_input_choices:
                                    result.append(t + (total_input,))
                            elif j == 2:
                                for sampling_rate in self.crop_sampling_rate_choices:
                                    result.append(t + (sampling_rate,))
                            elif j == 3:
                                for where in ["left", "middle", "right"]:
                                    result.append(t + (where,))
                            else:
                                raise ValueError("Unexpected index in Crop preprocessor")
                        elif name == "EmpiricalClipScaler":
                            if j == 1:
                                for q in self.empirical_clip_scaler_q_choices:
                                    result.append(t + (q,))
                            elif j == 2:
                                for scale in self.empirical_clip_scaler_scale_choices:
                                    result.append(t + (scale,))
                            else:
                                raise ValueError("Unexpected index in EmpiricalClipScaler preprocessor")
                        elif name == "FIR":
                            if j == 1:
                                for sampling_rate in self.fir_sampling_rate_choices:
                                    result.append(t + (sampling_rate,))
                            elif j == 2:
                                for channels in self.fir_channels_choices:
                                    result.append(t + (channels,))
                            elif j == 3:
                                for filter_params in self.fir_filter_params_choices:
                                    result.append(t + (filter_params,))
                            elif j == 4:
                                for zero_phase in [True, False]:
                                    result.append(t + (zero_phase,))
                            else:
                                raise ValueError("Unexpected index in FIR preprocessor")
                        elif name == "RobustScaler":
                            if j == 1:
                                for lower_quantile in self.robust_scaler_lower_quantile_choices:
                                    result.append(t + (lower_quantile,))
                            elif j == 2:
                                for upper_quantile in self.robust_scaler_upper_quantile_choices:
                                    result.append(t + (upper_quantile,))
                            else:
                                raise ValueError("Unexpected index in RobustScaler preprocessor")
                        elif name == "Spectogram":
                            if j == 1:
                                for n_fft in self.spectogram_n_fft_choices:
                                    result.append(t + (n_fft,))
                            elif j == 2:
                                for hop_length in self.spectogram_hop_length_choices:
                                    result.append(t + (hop_length,))
                            elif j == 3:
                                for win_length in self.spectogram_win_length_choices:
                                    result.append(t + (win_length,))
                            elif j == 4:
                                for epoch_len_samples in self.spectogram_epoch_len_samples_choices:
                                    result.append(t + (epoch_len_samples,))
                            else:
                                raise ValueError("Unexpected index in Spectogram preprocessor")
                        elif name == "ZNormalize":
                            if j == 1:
                                for use_global_statistics in [True, False]:
                                    result.append(t + (use_global_statistics,))
                            else:
                                raise ValueError("Unexpected index in ZNormalize preprocessor")

            return result






    class Maybe_Preprocessor(Group):
            name = "Maybe_Preprocessor"

            def __init__(self, channel_sampler_n_choices, crop_total_input_choices, crop_sampling_rate_choices,
                         empirical_clip_scaler_q_choices, empirical_clip_scaler_scale_choices,
                         fir_sampling_rate_choices, fir_channels_choices, fir_filter_params_choices,
                         robust_scaler_lower_quantile_choices, robust_scaler_upper_quantile_choices,
                         spectogram_n_fft_choices,
                         spectogram_hop_length_choices, spectogram_win_length_choices,
                         spectogram_epoch_len_samples_choices):
                self.channel_sampler_n_choices = channel_sampler_n_choices + [None]
                self.crop_total_input_choices = crop_total_input_choices + [None]
                self.crop_sampling_rate_choices = crop_sampling_rate_choices + [None]
                self.empirical_clip_scaler_q_choices = empirical_clip_scaler_q_choices + [None]
                self.empirical_clip_scaler_scale_choices = empirical_clip_scaler_scale_choices + [None]
                self.fir_sampling_rate_choices = fir_sampling_rate_choices + [None]
                self.fir_channels_choices = fir_channels_choices + [None]
                self.fir_filter_params_choices = fir_filter_params_choices + [None]
                self.robust_scaler_lower_quantile_choices = robust_scaler_lower_quantile_choices + [None]
                self.robust_scaler_upper_quantile_choices = robust_scaler_upper_quantile_choices + [None]
                self.spectogram_n_fft_choices = spectogram_n_fft_choices + [None]
                self.spectogram_hop_length_choices = spectogram_hop_length_choices + [None]
                self.spectogram_win_length_choices = spectogram_win_length_choices + [None]
                self.spectogram_epoch_len_samples_choices = spectogram_epoch_len_samples_choices + [None]

            def iter_channel_sampler(self):
                for n in self.channel_sampler_n_choices:
                    yield ("ChannelSampler", n)

            def iter_crop(self):
                for total_input in self.crop_total_input_choices:
                    for sampling_rate in self.crop_sampling_rate_choices:
                        for where in ["left", "middle", "right", None]:
                            yield ("Crop", total_input, sampling_rate, where)

            def iter_empirical_clip_scaler(self):
                for q in self.empirical_clip_scaler_q_choices:
                    for scale in self.empirical_clip_scaler_scale_choices:
                        yield ("EmpiricalClipScaler", q, scale)

            def iter_fir(self):
                for sampling_rate in self.fir_sampling_rate_choices:
                    for channels in self.fir_channels_choices:
                        for filter_params in self.fir_filter_params_choices:
                            for zero_phase in [True, False, None]:
                                yield ("FIR", sampling_rate, channels, filter_params, zero_phase)

            def iter_normalize(self):
                yield ("Normalize",)

            def iter_robust_scaler(self):
                for lower_quantile in self.robust_scaler_lower_quantile_choices:
                    for upper_quantile in self.robust_scaler_upper_quantile_choices:
                        yield ("RobustScaler", lower_quantile, upper_quantile)

            def iter_spectogram(self):
                for n_fft in self.spectogram_n_fft_choices:
                    for hop_length in self.spectogram_hop_length_choices:
                        for win_length in self.spectogram_win_length_choices:
                            for epoch_len_samples in self.spectogram_epoch_len_samples_choices:
                                yield ("Spectogram", n_fft, hop_length, win_length, epoch_len_samples)

            def iter_znormalize(self):
                for use_global_statistics in [True, False, None]:
                    yield ("ZNormalize", use_global_statistics)

            def __iter__(self):
                yield from self.iter_channel_sampler()
                yield from self.iter_crop()
                yield from self.iter_empirical_clip_scaler()
                yield from self.iter_fir()
                yield from self.iter_normalize()
                yield from self.iter_robust_scaler()
                yield from self.iter_spectogram()
                yield from self.iter_znormalize()

            def __contains__(self, value: object) -> bool:
                if (isinstance(value, tuple)):
                    if value[0] == "ChannelSampler":
                        return len(value) == 2 and value[1] in self.channel_sampler_n_choices
                    elif value[0] == "Crop":
                        return (len(value) == 4 and
                                value[1] in self.crop_total_input_choices and
                                value[2] in self.crop_sampling_rate_choices and
                                value[3] in ["left", "middle", "right", None])
                    elif value[0] == "EmpiricalClipScaler":
                        return (len(value) == 3 and
                                value[1] in self.empirical_clip_scaler_q_choices and
                                value[2] in self.empirical_clip_scaler_scale_choices)
                    elif value[0] == "FIR":
                        return (len(value) == 5 and
                                value[1] in self.fir_sampling_rate_choices and
                                value[2] in self.fir_channels_choices and
                                value[3] in self.fir_filter_params_choices and
                                value[4] in [True, False, None])
                    elif value[0] == "Normalize":
                        return len(value) == 1
                    elif value[0] == "RobustScaler":
                        return (len(value) == 3 and
                                value[1] in self.robust_scaler_lower_quantile_choices and
                                value[2] in self.robust_scaler_upper_quantile_choices)
                    elif value[0] == "Spectogram":
                        return (len(value) == 5 and
                                value[1] in self.spectogram_n_fft_choices and
                                value[2] in self.spectogram_hop_length_choices and
                                value[3] in self.spectogram_win_length_choices and
                                value[4] in self.spectogram_epoch_len_samples_choices)
                    elif value[0] == "ZNormalize":
                        return len(value) == 2 and value[1] in [True, False, None]
                    else:
                        return False
                else:
                    return False

    class Maybe_Preprocessor_Tuple(Group):
        name = "Maybe_Preprocessor_Tuple"

        def __init__(self, preprocessors):
            self.preprocessors = preprocessors

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and all(True if v is None else v in self.preprocessors for v in value))

    class Preprocessor_Tuple(Group):
        name = "Preprocessor_Tuple"

        def __init__(self, preprocessors):
            self.preprocessors = preprocessors

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and all(v in self.preprocessors for v in value))

        def from_maybe_preprocessor_tuple(self, maybe_preprocessor_tuple):
            result = []
            for pre in maybe_preprocessor_tuple:
                if pre is None:
                    if not result:
                        for p in self.preprocessors:
                            result.append((p,))
                    else:
                        old_result = result
                        result = []
                        for t in old_result:
                            for p in self.preprocessors:
                                result.append(t + (p,))
                else:
                    ps = self.preprocessors.from_maybe_preprocessor(pre)
                    if not result:
                        result = [(p,) for p in ps]
                    else:
                        old_result = result
                        result = []
                        for t in old_result:
                            for p in ps:
                                result.append(t + (p,))
            return result


    class Maybe_Optimizer(Group):
            name = "Maybe_Optimizer"

            """
            ("Adagrad", learning_rate, learning_rate_decay, weight_decay, initial_accumulator_value, eps)
            ("Adam", learning_rate, betas, eps, weight_decay, amsgrad = [True, False])
            ("AdamW", learning_rate, betas, eps, weight_decay, amsgrad = [True, False])
            ("Adamax", learning_rate, betas, eps, weight_decay)
            ("SGD", learning_rate, momentum, dampening, weight_decay, nesterov = [True, False])
            """

            def __init__(self, learning_rate, learning_rate_decay, weight_decay, eps, beta, initial_accumulator_value,
                         momentum, dampening):
                self.learning_rate = learning_rate + [None]
                self.learning_rate_decay = learning_rate_decay + [None]
                self.weight_decay = weight_decay + [None]
                self.eps = eps + [None]
                self.beta = beta + [None]
                self.initial_accumulator_value = initial_accumulator_value + [None]
                self.momentum = momentum + [None]
                self.dampening = dampening + [None]

            def iter_adagrad(self):
                for lr in self.learning_rate:
                    for lr_decay in self.learning_rate_decay:
                        for wd in self.weight_decay:
                            for init_acc in self.initial_accumulator_value:
                                for e in self.eps:
                                    yield ("Adagrad", lr, lr_decay, wd, init_acc, e)

            def iter_adam(self):
                for lr in self.learning_rate:
                    for b in self.beta:
                        for e in self.eps:
                            for wd in self.weight_decay:
                                for amsgrad in [True, False, None]:
                                    yield ("Adam", lr, b, e, wd, amsgrad)

            def iter_adamw(self):
                for lr in self.learning_rate:
                    for b in self.beta:
                        for e in self.eps:
                            for wd in self.weight_decay:
                                for amsgrad in [True, False, None]:
                                    yield ("AdamW", lr, b, e, wd, amsgrad)

            def iter_adamax(self):
                for lr in self.learning_rate:
                    for b in self.beta:
                        for e in self.eps:
                            for wd in self.weight_decay:
                                yield ("Adamax", lr, b, e, wd)

            def iter_sgd(self):
                for lr in self.learning_rate:
                    for m in self.momentum:
                        for d in self.dampening:
                            for wd in self.weight_decay:
                                for nesterov in [True, False, None]:
                                    yield ("SGD", lr, m, d, wd, nesterov)

            def __iter__(self):
                yield from self.iter_adagrad()
                yield from self.iter_adam()
                yield from self.iter_adamw()
                yield from self.iter_adamax()
                yield from self.iter_sgd()

            def __contains__(self, value: object) -> bool:
                if (isinstance(value, tuple)):
                    if value[0] == "Adagrad":
                        return (len(value) == 6 and
                                value[1] in self.learning_rate and
                                value[2] in self.learning_rate_decay and
                                value[3] in self.weight_decay and
                                value[4] in self.initial_accumulator_value and
                                value[5] in self.eps)
                    elif value[0] == "Adam":
                        return (len(value) == 6 and
                                value[1] in self.learning_rate and
                                value[2] in self.beta and
                                value[3] in self.eps and
                                value[4] in self.weight_decay and
                                value[5] in [True, False, None])
                    elif value[0] == "AdamW":
                        return (len(value) == 6 and
                                value[1] in self.learning_rate and
                                value[2] in self.beta and
                                value[3] in self.eps and
                                value[4] in self.weight_decay and
                                value[5] in [True, False, None])
                    elif value[0] == "Adamax":
                        return (len(value) == 5 and
                                value[1] in self.learning_rate and
                                value[2] in self.beta and
                                value[3] in self.eps and
                                value[4] in self.weight_decay)
                    elif value[0] == "SGD":
                        return (len(value) == 6 and
                                value[1] in self.learning_rate and
                                value[2] in self.momentum and
                                value[3] in self.dampening and
                                value[4] in self.weight_decay and
                                value[5] in [True, False, None])
                    else:
                        return False
                else:
                    return value is None

            def unfold_none(self, maybe_optimizer):
                if maybe_optimizer is None:
                    old_result = self.__iter__()
                    result = []
                    for t in old_result:
                        if None not in t:
                            result.append(t)
                    return result
                name = maybe_optimizer[0]
                result = [(name,)]
                for i, p in enumerate(maybe_optimizer[1:]):
                    j = i + 1
                    if p is not None:
                        old_result = result
                        result = []
                        for t in old_result:
                            result.append(t + (p,))
                    else:
                        old_result = result
                        result = []
                        for t in old_result:
                            if name == "Adagrad":
                                if j == 1:
                                    for lr in self.learning_rate:
                                        if lr is not None:
                                            result.append(t + (lr,))
                                elif j == 2:
                                    for lr_decay in self.learning_rate_decay:
                                        if lr_decay is not None:
                                            result.append(t + (lr_decay,))
                                elif j == 3:
                                    for wd in self.weight_decay:
                                        if wd is not None:
                                            result.append(t + (wd,))
                                elif j == 4:
                                    for init_acc in self.initial_accumulator_value:
                                        if init_acc is not None:
                                            result.append(t + (init_acc,))
                                elif j == 5:
                                    for e in self.eps:
                                        if e is not None:
                                            result.append(t + (e,))
                                else:
                                    raise ValueError("Unexpected index in Adagrad optimizer")
                            if name == "Adam":
                                if j == 1:
                                    for lr in self.learning_rate:
                                        if lr is not None:
                                            result.append(t + (lr,))
                                elif j == 2:
                                    for b in self.beta:
                                        if b is not None:
                                            result.append(t + (b,))
                                elif j == 3:
                                    for e in self.eps:
                                        if e is not None:
                                            result.append(t + (e,))
                                elif j == 4:
                                    for wd in self.weight_decay:
                                        if wd is not None:
                                            result.append(t + (wd,))
                                elif j == 5:
                                    for amsgrad in [True, False]:
                                        result.append(t + (amsgrad,))
                                else:
                                    raise ValueError("Unexpected index in Adam optimizer")
                            if name == "AdamW":
                                if j == 1:
                                    for lr in self.learning_rate:
                                        if lr is not None:
                                            result.append(t + (lr,))
                                elif j == 2:
                                    for b in self.beta:
                                        if b is not None:
                                            result.append(t + (b,))
                                elif j == 3:
                                    for e in self.eps:
                                        if e is not None:
                                            result.append(t + (e,))
                                elif j == 4:
                                    for wd in self.weight_decay:
                                        if wd is not None:
                                            result.append(t + (wd,))
                                elif j == 5:
                                    for amsgrad in [True, False]:
                                        result.append(t + (amsgrad,))
                                else:
                                    raise ValueError("Unexpected index in AdamW optimizer")
                            if name == "Adamax":
                                if j == 1:
                                    for lr in self.learning_rate:
                                        if lr is not None:
                                            result.append(t + (lr,))
                                elif j == 2:
                                    for b in self.beta:
                                        if b is not None:
                                            result.append(t + (b,))
                                elif j == 3:
                                    for e in self.eps:
                                        if e is not None:
                                            result.append(t + (e,))
                                elif j == 4:
                                    for wd in self.weight_decay:
                                        if wd is not None:
                                            result.append(t + (wd,))
                                else:
                                    raise ValueError("Unexpected index in Adamax optimizer")
                            if name == "SGD":
                                if j == 1:
                                    for lr in self.learning_rate:
                                        if lr is not None:
                                            result.append(t + (lr,))
                                elif j == 2:
                                    for m in self.momentum:
                                        if m is not None:
                                            result.append(t + (m,))
                                elif j == 3:
                                    for d in self.dampening:
                                        if d is not None:
                                            result.append(t + (d,))
                                elif j == 4:
                                    for wd in self.weight_decay:
                                        if wd is not None:
                                            result.append(t + (wd,))
                                elif j == 5:
                                    for nesterov in [True, False]:
                                        result.append(t + (nesterov,))
                                else:
                                    raise ValueError("Unexpected index in SGD optimizer")
                return result


    class Maybe_LR_Scheduler(Group):
            name = "Maybe_LR_Scheduler"

            """
            ("LinearLR", start_factor, end_factor, total_iters, last_epoch)
            ("StepLR", step_size, gamma, last_epoch)
            ("ExponentialLR", gamma, last_epoch)
            """

            def __init__(self, start_factor_choices, end_factor_choices, total_iters_choices,
                         step_size_choices, gamma_choices, last_epoch_choices):
                self.start_factor_choices = start_factor_choices + [None]
                self.end_factor_choices = end_factor_choices + [None]
                self.total_iters_choices = total_iters_choices + [None]
                self.step_size_choices = step_size_choices + [None]
                self.gamma_choices = gamma_choices + [None]
                self.last_epoch_choices = last_epoch_choices + [None]


            def iter_linear_lr(self):
                for start_factor in self.start_factor_choices:
                    for end_factor in self.end_factor_choices:
                        for total_iters in self.total_iters_choices:
                            for last_epoch in self.last_epoch_choices:
                                yield ("LinearLR", start_factor, end_factor, total_iters, last_epoch)

            def iter_step_lr(self):
                for step_size in self.step_size_choices:
                    for gamma in self.gamma_choices:
                        for last_epoch in self.last_epoch_choices:
                            yield ("StepLR", step_size, gamma, last_epoch)

            def iter_exponential_lr(self):
                for gamma in self.gamma_choices:
                    for last_epoch in self.last_epoch_choices:
                        yield ("ExponentialLR", gamma, last_epoch)

            def __iter__(self):
                yield from self.iter_linear_lr()
                yield from self.iter_step_lr()
                yield from self.iter_exponential_lr()

            def __contains__(self, value: object) -> bool:
                if (isinstance(value, tuple)):
                    if value[0] == "LinearLR":
                        return (len(value) == 5 and
                                value[1] in self.start_factor_choices and
                                value[2] in self.end_factor_choices and
                                value[3] in self.total_iters_choices and
                                value[4] in self.last_epoch_choices)
                    elif value[0] == "StepLR":
                        return (len(value) == 4 and
                                value[1] in self.step_size_choices and
                                value[2] in self.gamma_choices and
                                value[3] in self.last_epoch_choices)
                    elif value[0] == "ExponentialLR":
                        return (len(value) == 3 and
                                value[1] in self.gamma_choices and
                                value[2] in self.last_epoch_choices)
                    else:
                        return False
                else:
                    return value is None

            def unfold_none(self, maybe_lr_scheduler):
                if maybe_lr_scheduler is None:
                    old_result = self.__iter__()
                    result = []
                    for t in old_result:
                        if None not in t:
                            result.append(t)
                    return result
                name = maybe_lr_scheduler[0]
                result = [(name,)]
                for i, p in enumerate(maybe_lr_scheduler[1:]):
                    j = i + 1
                    if p is not None:
                        old_result = result
                        result = []
                        for t in old_result:
                            result.append(t + (p,))
                    else:
                        old_result = result
                        result = []
                        for t in old_result:
                            if name == "LinearLR":
                                if j == 1:
                                    for start_factor in self.start_factor_choices:
                                        if start_factor is not None:
                                            result.append(t + (start_factor,))
                                elif j == 2:
                                    for end_factor in self.end_factor_choices:
                                        if end_factor is not None:
                                            result.append(t + (end_factor,))
                                elif j == 3:
                                    for total_iters in self.total_iters_choices:
                                        if total_iters is not None:
                                            result.append(t + (total_iters,))
                                elif j == 4:
                                    for last_epoch in self.last_epoch_choices:
                                        if last_epoch is not None:
                                            result.append(t + (last_epoch,))
                                else:
                                    raise ValueError("Unexpected index in LinearLR scheduler")
                            if name == "StepLR":
                                if j == 1:
                                    for step_size in self.step_size_choices:
                                        if step_size is not None:
                                            result.append(t + (step_size,))
                                elif j == 2:
                                    for gamma in self.gamma_choices:
                                        if gamma is not None:
                                            result.append(t + (gamma,))
                                elif j == 3:
                                    for last_epoch in self.last_epoch_choices:
                                        if last_epoch is not None:
                                            result.append(t + (last_epoch,))
                                else:
                                    raise ValueError("Unexpected index in StepLR scheduler")
                            if name == "ExponentialLR":
                                if j == 1:
                                    for gamma in self.gamma_choices:
                                        if gamma is not None:
                                            result.append(t + (gamma,))
                                elif j == 2:
                                    for last_epoch in self.last_epoch_choices:
                                        if last_epoch is not None:
                                            result.append(t + (last_epoch,))
                                else:
                                    raise ValueError("Unexpected index in ExponentialLR scheduler")
                return result

    def specification(self):
        dimension = DataGroup("dimension", self.dimension_choices)
        maybe_dimension = DataGroup("dimension", self.dimension_choices + [None])
        normalization_eps = DataGroup("normalization_eps", self.normalization_eps_choices)
        dropout_p = DataGroup("dropout_p", self.dropout_p_choices)
        maxpool_size = DataGroup("maxpool_size", self.maxpool_size_choices)
        kernel_size = DataGroup("kernel_size", self.convolution_kernel_size_choices)
        maybe_kernel_size = DataGroup("kernel_size", self.convolution_kernel_size_choices + [None])
        activation_function = DataGroup("activation_function", self.afs)
        convolution = DataGroup("convolution", self.convs)
        normalization = DataGroup("normalization", self.norms)
        dimension_list = self.Maybe_Dimension_Tuple(self.dimension_choices)
        kernel_size_list = self.Maybe_Kernel_Size_Tuple(self.convolution_kernel_size_choices)
        maxpool_size_list = self.Maybe_Maxpool_Size_Tuple(self.maxpool_size_choices)
        length = self.Nat()
        maxpool_stride = DataGroup("maxpool_stride", self.maxpool_stride_choices)
        maxpool_padding = DataGroup("maxpool_padding", self.maxpool_padding_choices)
        maxpool_dilation = DataGroup("maxpool_dilation", self.maxpool_dilation_choices)
        convolution_stride = DataGroup("convolution_stride", self.convolution_stride_choices)
        convolution_padding = DataGroup("convolution_padding", self.convolution_padding_choices)
        convolution_dilation = DataGroup("convolution_dilation", self.convolution_dilations_choices)
        bias = DataGroup("bias", [True, False])
        loss = DataGroup("loss", self.losses)
        preprocessor = self.Preprocessor(self.preprocessor_channel_sampler_n_choices,
                                               self.preprocessor_crop_total_input_choices,
                                               self.preprocessor_crop_sampling_rate_choices,
                                               self.preprocessor_empirical_clip_scaler_q_choices,
                                               self.preprocessor_empirical_clip_scaler_scale_choices,
                                               self.preprocessor_fir_sampling_rate_choices,
                                               self.preprocessor_fir_channels_choices,
                                               self.preprocessor_fir_filter_params_choices,
                                               self.preprocessor_robust_scaler_lower_quantile_choices,
                                               self.preprocessor_robust_scaler_upper_quantile_choices,
                                               self.preprocessor_spectogram_n_fft_choices,
                                               self.preprocessor_spectogram_hop_length_choices,
                                               self.preprocessor_spectogram_win_length_choices,
                                               self.preprocessor_spectogram_epoch_len_samples_choices
                                               )
        maybe_preprocessor = self.Maybe_Preprocessor(self.preprocessor_channel_sampler_n_choices,
                                               self.preprocessor_crop_total_input_choices,
                                               self.preprocessor_crop_sampling_rate_choices,
                                               self.preprocessor_empirical_clip_scaler_q_choices,
                                               self.preprocessor_empirical_clip_scaler_scale_choices,
                                               self.preprocessor_fir_sampling_rate_choices,
                                               self.preprocessor_fir_channels_choices,
                                               self.preprocessor_fir_filter_params_choices,
                                               self.preprocessor_robust_scaler_lower_quantile_choices,
                                               self.preprocessor_robust_scaler_upper_quantile_choices,
                                               self.preprocessor_spectogram_n_fft_choices,
                                               self.preprocessor_spectogram_hop_length_choices,
                                               self.preprocessor_spectogram_win_length_choices,
                                               self.preprocessor_spectogram_epoch_len_samples_choices
                                               )
        maybe_preprocessor_tuple = self.Maybe_Preprocessor_Tuple(maybe_preprocessor)
        preprocessor_tuple = self.Preprocessor_Tuple(preprocessor)
        n_samples = DataGroup("n_samples", self.n_samples)
        """
        abc_channels = DataGroup("Channel", 
                                            ["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2", "E1", "E2", "ECG1", 
                                             "ECG2", "LLeg1", "LLeg2", "RLeg1", "RLeg2", "Chin1", "Chin2", "Chin3", 
                                             "Airflow", "Abdo", "Thor", "Snore", "Sum", "PosSensor", "Ox Status", 
                                             "Pulse", "SpO2", "Nasal Pressure", "CPAP Flow", "CPAP Press", "Pleth",
                                             "Derived HR", "Light", "Manual Pos"]) # ohne Respiration Rate, weil keine 100%
        """
        abc_channels = DataGroup("abc_channels", self.abc_channel_choices)
        optimizer_learning_rate = DataGroup("optimizer_learning_rate", self.optimizer_learning_rate)
        optimizer_learning_rate_decay = DataGroup("optimizer_learning_rate_decay", self.optimizer_learning_rate_decay)
        optimizer_weight_decay = DataGroup("optimizer_weight_decay", self.optimizer_weight_decay)
        optimizer_eps = DataGroup("optimizer_eps", self.optimizer_eps)
        optimizer_beta = DataGroup("optimizer_beta", self.optimizer_beta)
        optimizer_initial_accumulator_value = DataGroup("optimizer_initial_accumulator_value", self.optimizer_initial_accumulator_value)
        optimizer_momentum = DataGroup("optimizer_momentum", self.optimizer_momentum)
        optimizer_dampening = DataGroup("optimizer_dampening", self.optimizer_dampening)
        maybe_optimizer = self.Maybe_Optimizer(self.optimizer_learning_rate,
                                         self.optimizer_learning_rate_decay,
                                         self.optimizer_weight_decay,
                                         self.optimizer_eps,
                                         self.optimizer_beta,
                                         self.optimizer_initial_accumulator_value,
                                         self.optimizer_momentum,
                                         self.optimizer_dampening)
        lr_scheduler_start_factor = DataGroup("lr_scheduler_start_factor", self.lr_scheduler_start_factor_choices)
        lr_scheduler_end_factor = DataGroup("lr_scheduler_end_factor", self.lr_scheduler_end_factor_choices)
        lr_scheduler_total_iters = DataGroup("lr_scheduler_total_iters", self.lr_scheduler_total_iters_choices)
        lr_scheduler_step_size = DataGroup("lr_scheduler_step_size", self.lr_scheduler_step_size_choices)
        lr_scheduler_gamma = DataGroup("lr_scheduler_gamma", self.lr_scheduler_gamma_choices)
        lr_scheduler_last_epoch = DataGroup("lr_scheduler_last_epoch", self.lr_scheduler_last_epoch_choices)
        maybe_lr_scheduler = self.Maybe_LR_Scheduler(self.lr_scheduler_start_factor_choices,
                                               self.lr_scheduler_end_factor_choices,
                                               self.lr_scheduler_total_iters_choices,
                                               self.lr_scheduler_step_size_choices,
                                               self.lr_scheduler_gamma_choices,
                                               self.lr_scheduler_last_epoch_choices)

        return {
            "ReLu": Constructor("activation_function") & Literal("ReLu") & Literal(None),

            "ELU": Constructor("activation_function") & Literal("ELU") & Literal(None),

            "Tanh": Constructor("activation_function") & Literal("Tanh") & Literal(None),

            "BatchNorm1d": DSL()
            .parameter("n", dimension)
            .parameter("e", normalization_eps)
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n"))
                                & Constructor("output", Literal(None))
                                )
                    & Constructor("normalization_epsilon", Var("e"))
                    & Constructor("normalization_epsilon", Literal(None))
                    & Literal("batch_norm")
                    ),

            "ChannelWiseNorm": DSL()
            .parameter("n", dimension)
            .parameter("e", normalization_eps)
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n"))
                                & Constructor("output", Literal(None))
                                )
                    & Constructor("normalization_epsilon", Var("e"))
                    & Constructor("normalization_epsilon", Literal(None))
                    & Literal("channel_wise_norm")
                    ),

            "Conv1dLayerNorm": DSL()
            .parameter("n", dimension)
            .parameter("e", normalization_eps)
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n"))
                                & Constructor("output", Literal(None))
                                )
                    & Constructor("normalization_epsilon", Var("e"))
                    & Constructor("normalization_epsilon", Literal(None))
                    & Constructor("conv1d_layer_norm")
                    ),

            "Dropout1d": DSL()
            .parameter("d", dropout_p)
            .suffix(Constructor("dropout")
                    & Constructor("dropout_probability", Var("d"))
                    & Constructor("dropout_probability", Literal(None))
                    ),

            "Maxpool1d": DSL()
            .parameter("n", maxpool_size)
            .parameter("s", maxpool_stride)
            .parameter("p", maxpool_padding)
            .parameter("d", maxpool_dilation)
            .suffix(Constructor("maxpool1d",
                                Constructor("size", Var("n"))
                                & Constructor("size", Literal(None))
                                )
                    & Constructor("maxpool_stride", Var("s"))
                    & Constructor("maxpool_stride", Literal(None))
                    & Constructor("maxpool_padding", Var("p"))
                    & Constructor("maxpool_padding", Literal(None))
                    & Constructor("maxpool_dilation", Var("d"))
                    & Constructor("maxpool_dilation", Literal(None))
                    ),

            "Upsample1d": DSL()
            .parameter("n", maxpool_size)
            .suffix(Constructor("upsample1d",
                                Constructor("scale_factor", Var("n"))
                                & Constructor("scale_factor", Literal(None))
                                )
                    ),

            "Conv1d": DSL()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("k", kernel_size)
            .parameter("s", convolution_stride)
            .parameter("p", convolution_padding)
            .parameter("d", convolution_dilation)
            .parameter("b", bias)
            .suffix(Constructor("1d_conv_layer",
                                Constructor("input", Var("in"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("out"))
                                & Constructor("output", Literal(None))
                                & Constructor("kernel_size", Var("k"))
                                & Constructor("kernel_size", Literal(None))
                                )
                    & Literal("simple_convolution")
                    & Literal(None)
                    & Constructor("convolution_stride", Var("s"))
                    & Constructor("convolution_stride", Literal(None))
                    & Constructor("convolution_padding", Var("p"))
                    & Constructor("convolution_padding", Literal(None))
                    & Constructor("convolution_dilation", Var("d"))
                    & Constructor("convolution_dilation", Literal(None))
                    & Constructor("bias", Var("b"))
                    & Constructor("bias", Literal(None))
                    ),

            "DepthwiseSeparableConv1d": DSL()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("k", kernel_size)
            .parameter("s", convolution_stride)
            .parameter("p", convolution_padding)
            .parameter("d", convolution_dilation)
            .parameter("b", bias)
            .suffix(Constructor("1d_conv_layer",
                                Constructor("input", Var("in"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("out"))
                                & Constructor("output", Literal(None))
                                & Constructor("kernel_size", Var("k"))
                                & Constructor("kernel_size", Literal(None))
                                )
                    & Literal("depthwise_separable_convolution")
                    & Literal(None)
                    & Constructor("convolution_stride", Var("s"))
                    & Constructor("convolution_stride", Literal(None))
                    & Constructor("convolution_padding", Var("p"))
                    & Constructor("convolution_padding", Literal(None))
                    & Constructor("convolution_dilation", Var("d"))
                    & Constructor("convolution_dilation", Literal(None))
                    & Constructor("bias", Var("b"))
                    & Constructor("bias", Literal(None))
                    ),

            "ConvBlock": DSL()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("k", kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("stride", convolution_stride)
            .parameter("padding", convolution_padding)
            .parameter("dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("norm", normalization)
            .parameter("norm_e", normalization_eps)
            .argument("activation", Constructor("activation_function") & Var("af"))
            .argument("dropout", Constructor("dropout") & Constructor("dropout_probability", Var("d")))
            .argument("c1", Constructor("1d_conv_layer",
                                        Constructor("input", Var("in"))
                                        & Constructor("output", Var("out"))
                                        & Constructor("kernel_size", Var("k"))
                                        )
                      & Var("conv")
                      & Constructor("convolution_stride", Var("stride"))
                      & Constructor("convolution_padding", Var("padding"))
                      & Constructor("convolution_dilation", Var("dilation"))
                      & Constructor("bias", Var("b"))
                      )
            .argument("c2", Constructor("1d_conv_layer",
                                        Constructor("input", Var("in"))
                                        & Constructor("output", Var("out"))
                                        & Constructor("kernel_size", Var("k"))
                                        )
                      & Var("conv")
                      & Constructor("convolution_stride", Var("stride"))
                      & Constructor("convolution_padding", Var("padding"))
                      & Constructor("convolution_dilation", Var("dilation"))
                      & Constructor("bias", Var("b"))
                      )
            .argument("n1",
                      Constructor("normalization", Constructor("output", Var("out")))
                      & Var("norm")
                      & Constructor("normalization_epsilon", Var("norm_e")))
            .suffix(Constructor("conv_block",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out"))
                                & Constructor("kernel_size", Var("k"))
                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution_stride", Var("stride"))
                                  & Constructor("convolution_padding", Var("padding"))
                                  & Constructor("convolution_dilation", Var("dilation"))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization_epsilon", Var("norm_e"))
                                  )
                    ),

            "Encoder": DSL()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("k", kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("c_stride", convolution_stride)
            .parameter("c_padding", convolution_padding)
            .parameter("c_dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("e", normalization_eps)
            .parameter("norm", normalization)
            .parameter("m", maxpool_size)
            .parameter("m_stride", convolution_stride)
            .parameter("m_padding", convolution_padding)
            .parameter("m_dilation", convolution_dilation)
            .argument("mp",
                      Constructor("maxpool1d",Constructor("size", Var("m")))
                      & Constructor("maxpool_stride", Var("m_stride"))
                      & Constructor("maxpool_padding", Var("m_padding"))
                      & Constructor("maxpool_dilation", Var("m_dilation"))
                      )
            .argument("cb",
                      Constructor("conv_block",
                                  Constructor("input", Var("in"))
                                  & Constructor("output", Var("out"))
                                  & Constructor("kernel_size", Var("k")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("conv"))
                                    & Constructor("convolution_stride", Var("c_stride"))
                                    & Constructor("convolution_padding", Var("c_padding"))
                                    & Constructor("convolution_dilation", Var("c_dilation"))
                                    & Constructor("bias", Var("b"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("norm"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .suffix(Constructor("encoder",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out"))
                                & Constructor("kernel_size", Var("k"))
                                & Constructor("maxpool_size", Var("m"))
                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("maxpool_stride", Var("m_stride"))
                                  & Constructor("maxpool_padding", Var("m_padding"))
                                  & Constructor("maxpool_dilation", Var("m_dilation"))
                                  )
                    ),

            "Decoder": DSL()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("k", kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("c_stride", convolution_stride)
            .parameter("c_padding", convolution_padding)
            .parameter("c_dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("e", normalization_eps)
            .parameter("norm", normalization)
            .parameter("sf", maxpool_size)
            .argument("up", Constructor("upsample1d", Constructor("scale_factor", Var("sf"))))
            .argument("cb",
                      Constructor("conv_block",
                                  Constructor("input", Var("in"))
                                  & Constructor("output", Var("out"))
                                  & Constructor("kernel_size", Var("k")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("conv"))
                                    & Constructor("convolution_stride", Var("c_stride"))
                                    & Constructor("convolution_padding", Var("c_padding"))
                                    & Constructor("convolution_dilation", Var("c_dilation"))
                                    & Constructor("bias", Var("b"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("norm"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .suffix(Constructor("decoder",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out"))
                                & Constructor("kernel_size", Var("k"))
                                & Constructor("upsample_size", Var("sf"))
                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  )
                    ),

            "UStructure": DSL()
            .parameter("in", dimension)
            .parameter("out_enc", dimension)
            .parameter("in_dec", dimension, lambda v: [2 * v["out_enc"]])
            .parameter("k1", kernel_size)
            .parameter("k2", kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("c_stride", convolution_stride)
            .parameter("c_padding", convolution_padding)
            .parameter("c_dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("e", normalization_eps)
            .parameter("norm", normalization)
            .parameter("m", maxpool_size)
            .parameter("m_stride", maxpool_stride)
            .parameter("m_padding", maxpool_padding)
            .parameter("m_dilation", maxpool_dilation)
            .parameter("ds", dimension_list, lambda v: [(v["in"],), (None,)])
            .parameter("ks", kernel_size_list, lambda v: [(v["k1"],), (None,)])
            .parameter("ms", maxpool_size_list, lambda v: [(v["m"],), (None,)])
            .argument("enc", Constructor("encoder",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out_enc"))
                                & Constructor("kernel_size", Var("k1"))
                                & Constructor("maxpool_size", Var("m"))
                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("maxpool_stride", Var("m_stride"))
                                  & Constructor("maxpool_padding", Var("m_padding"))
                                  & Constructor("maxpool_dilation", Var("m_dilation"))
                                  )
                      )
            .argument("dec", Constructor("decoder",
                                Constructor("input", Var("in_dec"))
                                & Constructor("output", Var("in"))
                                & Constructor("kernel_size", Var("k1"))
                                & Constructor("upsample_size", Var("m"))
                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  )
                      )
            .argument("cb",
                      Constructor("conv_block",
                                  Constructor("input", Var("out_enc"))
                                  & Constructor("output", Var("out_enc"))
                                  & Constructor("kernel_size", Var("k2")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("conv"))
                                    & Constructor("convolution_stride", Var("c_stride"))
                                    & Constructor("convolution_padding", Var("c_padding"))
                                    & Constructor("convolution_dilation", Var("c_dilation"))
                                    & Constructor("bias", Var("b"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("norm"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .suffix(Constructor("u_structure",
                                Constructor("dimensions", Var("ds"))
                                & Constructor("kernel_sizes", Var("ks"))
                                & Constructor("maxpool_sizes", Var("ms"))
                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("out_enc"))
                                  & Constructor("in_and_out", Literal(None))
                                  & Constructor("kernel_size", Var("k2"))
                                  & Constructor("kernel_size", Literal(None))
                                  )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution", Literal(None))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_stride", Literal(None))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_padding", Literal(None))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("convolution_dilation", Literal(None))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("bias", Literal(None))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("activation", Literal(None))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("dropout_p", Literal(None))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization", Literal(None))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("normalization_epsilon", Literal(None))
                                  & Constructor("maxpool_stride", Var("m_stride"))
                                  & Constructor("maxpool_stride", Literal(None))
                                  & Constructor("maxpool_padding", Var("m_padding"))
                                  & Constructor("maxpool_padding", Literal(None))
                                  & Constructor("maxpool_dilation", Var("m_dilation"))
                                  & Constructor("maxpool_dilation", Literal(None))
                                  )
                    ),

            "UStructure_Cons": DSL()
            .parameter("in_u", dimension)  # in_u == out_enc
            .parameter("in_enc", dimension)
            .parameter("in_dec", dimension, lambda v: [2 * v["in_u"]])
            .parameter("bd", maybe_dimension)
            .parameter("k", kernel_size)
            .parameter("bk", maybe_kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("c_stride", convolution_stride)
            .parameter("c_padding", convolution_padding)
            .parameter("c_dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("e", normalization_eps)
            .parameter("norm", normalization)
            .parameter("m", maxpool_size)
            .parameter("m_stride", maxpool_stride)
            .parameter("m_padding", maxpool_padding)
            .parameter("m_dilation", maxpool_dilation)
            .parameter("dds", dimension_list)
            .parameter_constraint(lambda v: len(v["dds"]) > 1 and (v["dds"][0] == v["in_enc"] or v["dds"][0] is None))
            .parameter("ds", dimension_list, lambda v: [v["dds"][1:]])
            .parameter_constraint(lambda v: len(v["ds"]) > 0 and (v["ds"][0] == v["in_u"] or v["ds"][0] is None))
            .parameter("kks", kernel_size_list)
            .parameter_constraint(lambda v: len(v["kks"]) > 1 and (v["kks"][0] == v["k"] or v["kks"][0] is None))
            .parameter("ks", kernel_size_list, lambda v: [v["kks"][1:]])
            .parameter_constraint(lambda v: len(v["ks"]) > 0)
            .parameter("mms", maxpool_size_list)
            .parameter_constraint(lambda v: len(v["mms"]) > 1 and (v["mms"][0] == v["m"] or v["mms"][0] is None))
            .parameter("ms", maxpool_size_list, lambda v: [v["mms"][1:]])
            .parameter_constraint(lambda v: len(v["ms"]) > 0)
            .argument("enc", Constructor("encoder",
                                         Constructor("input", Var("in_enc"))
                                         & Constructor("output", Var("in_u"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("maxpool_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("conv"))
                                    & Constructor("convolution_stride", Var("c_stride"))
                                    & Constructor("convolution_padding", Var("c_padding"))
                                    & Constructor("convolution_dilation", Var("c_dilation"))
                                    & Constructor("bias", Var("b"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("norm"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    & Constructor("maxpool_stride", Var("m_stride"))
                                    & Constructor("maxpool_padding", Var("m_padding"))
                                    & Constructor("maxpool_dilation", Var("m_dilation"))
                                    )
                      )
            .argument("dec", Constructor("decoder",
                                         Constructor("input", Var("in_dec"))
                                         & Constructor("output", Var("in_enc"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("upsample_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("conv"))
                                    & Constructor("convolution_stride", Var("c_stride"))
                                    & Constructor("convolution_padding", Var("c_padding"))
                                    & Constructor("convolution_dilation", Var("c_dilation"))
                                    & Constructor("bias", Var("b"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("norm"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .argument("u", Constructor("u_structure",
                                Constructor("dimensions", Var("ds"))
                                & Constructor("kernel_sizes", Var("ks"))
                                & Constructor("maxpool_sizes", Var("ms"))
                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("bd"))
                                  & Constructor("kernel_size", Var("bk"))
                                  )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("maxpool_stride", Var("m_stride"))
                                  & Constructor("maxpool_padding", Var("m_padding"))
                                  & Constructor("maxpool_dilation", Var("m_dilation"))
                                  )
                      )
            .suffix(Constructor("u_structure",
                                Constructor("dimensions", Var("dds"))
                                & Constructor("kernel_sizes", Var("kks"))
                                & Constructor("maxpool_sizes", Var("mms"))
                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("bd"))
                                  & Constructor("kernel_size", Var("bk")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution", Literal(None))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_stride", Literal(None))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_padding", Literal(None))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("convolution_dilation", Literal(None))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("bias", Literal(None))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("activation", Literal(None))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("dropout_p", Literal(None))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization", Literal(None))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("normalization_epsilon", Literal(None))
                                  & Constructor("maxpool_stride", Var("m_stride"))
                                  & Constructor("maxpool_stride", Literal(None))
                                  & Constructor("maxpool_padding", Var("m_padding"))
                                  & Constructor("maxpool_padding", Literal(None))
                                  & Constructor("maxpool_dilation", Var("m_dilation"))
                                  & Constructor("maxpool_dilation", Literal(None))
                                  )
                    ),

            "LinearLayer": DSL()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("b", bias)
            .suffix(Constructor("linear_layer",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out"))
                                & Constructor("bias", Var("b"))
                                )
                    ),

            "UClassifier": DSL()
            .parameter("in_u", dimension)  # in_u == out_enc
            .parameter("in_enc", dimension)
            .parameter("in_dec", dimension, lambda v: [2 * v["in_u"]])
            .parameter("bd", maybe_dimension)
            .parameter("k", kernel_size)
            .parameter("bk", maybe_kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("c_stride", convolution_stride)
            .parameter("c_padding", convolution_padding)
            .parameter("c_dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("e", normalization_eps)
            .parameter("norm", normalization)
            .parameter("m", maxpool_size)
            .parameter("m_stride", maxpool_stride)
            .parameter("m_padding", maxpool_padding)
            .parameter("m_dilation", maxpool_dilation)
            .parameter("first_d", dropout_p)
            .parameter("first_af", activation_function)
            .parameter("first_conv", convolution)
            .parameter("first_c_stride", convolution_stride)
            .parameter("first_c_padding", convolution_padding)
            .parameter("first_c_dilation", convolution_dilation)
            .parameter("first_b", bias)
            .parameter("first_e", normalization_eps)
            .parameter("first_norm", normalization)
            .parameter("first_m_stride", maxpool_stride)
            .parameter("first_m_padding", maxpool_padding)
            .parameter("first_m_dilation", maxpool_dilation)
            .parameter("fc_k", kernel_size)
            .parameter("fc_conv", convolution)
            .parameter("fc_stride", convolution_stride)
            .parameter("fc_padding", convolution_padding)
            .parameter("fc_dilation", convolution_dilation)
            .parameter("fc_b", bias)
            .parameter("mlp_in", dimension)
            .parameter("mlp_out", dimension)
            .parameter("mlp_b", bias)
            .parameter("dds", dimension_list)
            .parameter_constraint(lambda v: len(v["dds"]) > 1 and (v["dds"][0] == v["in_enc"] or v["dds"][0] is None))
            .parameter("ds", dimension_list, lambda v: [v["dds"][1:]])
            .parameter_constraint(lambda v: len(v["ds"]) > 0 and (v["ds"][0] == v["in_u"] or v["ds"][0] is None))
            .parameter("kks", kernel_size_list)
            .parameter_constraint(lambda v: len(v["kks"]) > 1 and (v["kks"][0] == v["k"] or v["kks"][0] is None))
            .parameter("ks", kernel_size_list, lambda v: [v["kks"][1:]])
            .parameter_constraint(lambda v: len(v["ks"]) > 0)
            .parameter("mms", maxpool_size_list)
            .parameter_constraint(lambda v: len(v["mms"]) > 1 and (v["mms"][0] == v["m"] or v["mms"][0] is None))
            .parameter("ms", maxpool_size_list, lambda v: [v["mms"][1:]])
            .parameter_constraint(lambda v: len(v["ms"]) > 0)
            .argument("enc", Constructor("encoder",
                                         Constructor("input", Var("in_enc"))
                                         & Constructor("output", Var("in_u"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("maxpool_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("first_conv"))
                                    & Constructor("convolution_stride", Var("first_c_stride"))
                                    & Constructor("convolution_padding", Var("first_c_padding"))
                                    & Constructor("convolution_dilation", Var("first_c_dilation"))
                                    & Constructor("bias", Var("first_b"))
                                    & Constructor("activation", Var("first_af"))
                                    & Constructor("dropout_p", Var("first_d"))
                                    & Constructor("normalization", Var("first_norm"))
                                    & Constructor("normalization_epsilon", Var("first_e"))
                                    & Constructor("maxpool_stride", Var("first_m_stride"))
                                    & Constructor("maxpool_padding", Var("first_m_padding"))
                                    & Constructor("maxpool_dilation", Var("first_m_dilation"))
                                    )
                      )
            .argument("dec", Constructor("decoder",
                                         Constructor("input", Var("in_dec"))
                                         & Constructor("output", Var("in_enc"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("upsample_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("first_conv"))
                                    & Constructor("convolution_stride", Var("first_c_stride"))
                                    & Constructor("convolution_padding", Var("first_c_padding"))
                                    & Constructor("convolution_dilation", Var("first_c_dilation"))
                                    & Constructor("bias", Var("first_b"))
                                    & Constructor("activation", Var("first_af"))
                                    & Constructor("dropout_p", Var("first_d"))
                                    & Constructor("normalization", Var("first_norm"))
                                    & Constructor("normalization_epsilon", Var("first_e"))
                                    )
                      )
            .argument("u", Constructor("u_structure",
                                       Constructor("dimensions", Var("ds"))
                                       & Constructor("kernel_sizes", Var("ks"))
                                       & Constructor("maxpool_sizes", Var("ms"))
                                       )
                      & Constructor("bottleneck",
                                    Constructor("in_and_out", Var("bd"))
                                    & Constructor("kernel_size", Var("bk"))
                                    )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("conv"))
                                    & Constructor("convolution_stride", Var("c_stride"))
                                    & Constructor("convolution_padding", Var("c_padding"))
                                    & Constructor("convolution_dilation", Var("c_dilation"))
                                    & Constructor("bias", Var("b"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("norm"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    & Constructor("maxpool_stride", Var("m_stride"))
                                    & Constructor("maxpool_padding", Var("m_padding"))
                                    & Constructor("maxpool_dilation", Var("m_dilation"))
                                    )
                      )
            .argument("fc", Constructor("1d_conv_layer",
                                        Constructor("input", Var("in_enc"))
                                        & Constructor("output", Var("mlp_in"))
                                        & Constructor("kernel_size", Literal(1))
                                        )
                      & Var("fc_conv")
                      & Constructor("convolution_stride", Var("fc_stride"))
                      & Constructor("convolution_padding", Var("fc_padding"))
                      & Constructor("convolution_dilation", Var("fc_dilation"))
                      & Constructor("bias", Var("fc_b"))
                      )
            .argument("mlp", Constructor("linear_layer",
                                         Constructor("input", Var("mlp_in"))
                                         & Constructor("output", Var("mlp_out"))
                                         & Constructor("bias", Var("mlp_b"))
                                         )
                      )
            .suffix(Constructor("u_classifier",
                                Constructor("dimensions", Var("dds"))
                                & Constructor("kernel_sizes", Var("kks"))
                                & Constructor("maxpool_sizes", Var("mms"))
                                )
                    & Constructor("u_first_level", Constructor("convolution", Var("first_conv"))
                                              & Constructor("convolution", Literal(None))
                                              & Constructor("convolution_stride", Var("first_c_stride"))
                                              & Constructor("convolution_stride", Literal(None))
                                              & Constructor("convolution_padding", Var("first_c_padding"))
                                              & Constructor("convolution_padding", Literal(None))
                                              & Constructor("convolution_dilation", Var("first_c_dilation"))
                                              & Constructor("convolution_dilation", Literal(None))
                                              & Constructor("bias", Var("first_b"))
                                              & Constructor("bias", Literal(None))
                                              & Constructor("activation", Var("first_af"))
                                              & Constructor("activation", Literal(None))
                                              & Constructor("dropout_p", Var("first_d"))
                                              & Constructor("dropout_p", Literal(None))
                                              & Constructor("normalization", Var("first_norm"))
                                              & Constructor("normalization", Literal(None))
                                              & Constructor("normalization_epsilon", Var("first_e"))
                                              & Constructor("normalization_epsilon", Literal(None))
                                              & Constructor("maxpool_stride", Var("first_m_stride"))
                                              & Constructor("maxpool_stride", Literal(None))
                                              & Constructor("maxpool_padding", Var("first_m_padding"))
                                              & Constructor("maxpool_padding", Literal(None))
                                              & Constructor("maxpool_dilation", Var("first_m_dilation"))
                                              & Constructor("maxpool_dilation", Literal(None))
                                  )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("bd"))
                                  & Constructor("kernel_size", Var("bk"))
                                  )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("conv"))
                                  & Constructor("convolution", Literal(None))
                                  & Constructor("convolution_stride", Var("c_stride"))
                                  & Constructor("convolution_stride", Literal(None))
                                  & Constructor("convolution_padding", Var("c_padding"))
                                  & Constructor("convolution_padding", Literal(None))
                                  & Constructor("convolution_dilation", Var("c_dilation"))
                                  & Constructor("convolution_dilation", Literal(None))
                                  & Constructor("bias", Var("b"))
                                  & Constructor("bias", Literal(None))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("activation", Literal(None))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("dropout_p", Literal(None))
                                  & Constructor("normalization", Var("norm"))
                                  & Constructor("normalization", Literal(None))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("normalization_epsilon", Literal(None))
                                  & Constructor("maxpool_stride", Var("m_stride"))
                                  & Constructor("maxpool_stride", Literal(None))
                                  & Constructor("maxpool_padding", Var("m_padding"))
                                  & Constructor("maxpool_padding", Literal(None))
                                  & Constructor("maxpool_dilation", Var("m_dilation"))
                                  & Constructor("maxpool_dilation", Literal(None))
                                  )
                    & Constructor("u_final_conv",
                                  Constructor("kernel_size", Var("fc_k"))
                                  & Constructor("kernel_size", Literal(None))
                                  & Constructor("convolution", Var("fc_conv"))
                                  & Constructor("convolution", Literal(None))
                                  & Constructor("convolution_stride", Var("fc_stride"))
                                  & Constructor("convolution_stride", Literal(None))
                                  & Constructor("convolution_padding", Var("fc_padding"))
                                  & Constructor("convolution_padding", Literal(None))
                                  & Constructor("convolution_dilation", Var("fc_dilation"))
                                  & Constructor("convolution_dilation", Literal(None))
                                  & Constructor("bias", Var("fc_b"))
                                  & Constructor("bias", Literal(None))
                                  )
                    & Constructor("u_linear_classifier",
                                  Constructor("linear_layer",
                                              Constructor("input", Var("mlp_in"))
                                              & Constructor("input", Literal(None))
                                              & Constructor("output", Var("mlp_out"))
                                              & Constructor("output", Literal(None))
                                              & Constructor("bias", Var("mlp_b"))
                                              & Constructor("bias", Literal(None))
                                              )
                                  )
                    ),

            # For simplicity, we only consider loss functions that are not parameterized

            "BCEwithLogits": Constructor("loss_function") & Literal("BCE_with_logits") & Literal(None),

            "CrossEntropy": Constructor("loss_function") & Literal("CrossEntropy") & Literal(None),

            "MAE": Constructor("loss_function") & Literal("MAE") & Literal(None),

            "MSE": Constructor("loss_function") & Literal("MSE") & Literal(None),

            "ChannelSampler": DSL()
            .parameter("n", DataGroup("ChannelSampler_n", self.preprocessor_channel_sampler_n_choices))
            .parameter("p", preprocessor, lambda v: [("ChannelSampler", v["n"])])
            .suffix(Constructor("preprocessor", Var("p")) &
                    Constructor("preprocessor", Literal(("ChannelSampler", None)))),

            "Crop": DSL()
            .parameter("total_input", DataGroup("Crop_total_input", self.preprocessor_crop_total_input_choices))
            .parameter("sampling_rate", DataGroup("Crop_sampling_rate", self.preprocessor_crop_sampling_rate_choices))
            .parameter("where", DataGroup("Crop_where", ["left", "middle", "right"]))
            .parameter("p", preprocessor, lambda v: [("Crop", v["total_input"], v["sampling_rate"], v["where"])])
            #.parameter("p_none", maybe_preprocessor)
            #.parameter_constraint(lambda v: v["p_none"][0] == "Crop"
            #                                and (v["p_none"][1] == v["total_input"] or v["p_none"][1] is None)
            #                                and (v["p_none"][2] == v["sampling_rate"] or v["p_none"][2] is None)
            #                                and (v["p_none"][3] == v["where"] or v["p_none"][3] is None))
            #.suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            # .suffix(Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p"))),

            "EmpiricalClipScaler": DSL()
            .parameter("q", DataGroup("EmpiricalClipScaler_q", self.preprocessor_empirical_clip_scaler_q_choices))
            .parameter("scale", DataGroup("EmpiricalClipScaler_scale", self.preprocessor_empirical_clip_scaler_scale_choices))
            .parameter("p", preprocessor, lambda v: [("EmpiricalClipScaler", v["q"], v["scale"])])
            #.parameter("p_none", maybe_preprocessor)
            #.parameter_constraint(lambda v: v["p_none"][0] == "EmpiricalClipScaler"
            #                                and (v["p_none"][1] == v["q"] or v["p_none"][1] is None)
            #                                and (v["p_none"][2] == v["scale"] or v["p_none"][2] is None))
            #.suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            # .suffix(Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p"))),

            "FIR": DSL()
            .parameter("sampling_rate", DataGroup("FIR_sampling_rate", self.preprocessor_fir_sampling_rate_choices))
            .parameter("channels", DataGroup("FIR_channels", self.preprocessor_fir_channels_choices))
            .parameter("filter_params", DataGroup("FIR_filter_params", self.preprocessor_fir_filter_params_choices))
            .parameter("zero_phase", DataGroup("FIR_zero_phase", [True, False]))
            .parameter("p", preprocessor, lambda v: [("FIR", v["sampling_rate"], v["channels"], v["filter_params"], v["zero_phase"])])
            #.parameter("p_none", maybe_preprocessor)
            #.parameter_constraint(lambda v: v["p_none"][0] == "FIR"
            #                                and (v["p_none"][1] == v["sampling_rate"] or v["p_none"][1] is None)
            #                                and (v["p_none"][2] == v["channels"] or v["p_none"][2] is None)
            #                                and (v["p_none"][3] == v["filter_params"] or v["p_none"][3] is None)
            #                                and (v["p_none"][4] == v["zero_phase"] or v["p_none"][4] is None))
            #.suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            # .suffix(Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p"))),

            "Normalize": DSL()
            #.parameter("p", preprocessor, lambda v: [("Normalize",)])
            .suffix(Constructor("preprocessor", Literal(("Normalize",)))),

            "RobustScaler": DSL()
            .parameter("lower_quantile", DataGroup("RobustScaler_lower_quantile", self.preprocessor_robust_scaler_lower_quantile_choices))
            .parameter("upper_quantile", DataGroup("RobustScaler_upper_quantile", self.preprocessor_robust_scaler_upper_quantile_choices))
            .parameter("p", preprocessor, lambda v: [("RobustScaler", v["lower_quantile"], v["upper_quantile"])])
            #.parameter("p_none", maybe_preprocessor)
            #.parameter_constraint(lambda v: v["p_none"][0] == "RobustScaler"
            #                                and (v["p_none"][1] == v["lower_quantile"] or v["p_none"][1] is None)
            #                                and (v["p_none"][2] == v["upper_quantile"] or v["p_none"][2] is None))
            # .suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            # .suffix(Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p"))),

            "Spectogram": DSL()
            .parameter("n_fft", DataGroup("Spectogram_n_fft", self.preprocessor_spectogram_n_fft_choices))
            .parameter("hop_length", DataGroup("Spectogram_hop_length", self.preprocessor_spectogram_hop_length_choices))
            .parameter("win_length", DataGroup("Spectogram_win_length", self.preprocessor_spectogram_win_length_choices))
            .parameter("epoch_len_samples", DataGroup("Spectogram_epoch_len_samples", self.preprocessor_spectogram_epoch_len_samples_choices))
            .parameter("p", preprocessor,
                       lambda v: [("Spectogram", v["n_fft"], v["hop_length"], v["win_length"], v["epoch_len_samples"])])
            #.parameter("p_none", maybe_preprocessor)
            #.parameter_constraint(lambda v: v["p_none"][0] == "Spectogram"
            #                                and (v["p_none"][1] == v["n_fft"] or v["p_none"][1] is None)
            #                                and (v["p_none"][2] == v["hop_length"] or v["p_none"][2] is None)
            #                                and (v["p_none"][3] is None or torch.equal(v["p_none"][3], v["win_length"]))
            #                                and (v["p_none"][4] == v["epoch_len_samples"] or v["p_none"][4] is None))
            # .suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            #.suffix(Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p"))),

            "ZNormalize": DSL()
            .parameter("use_global_statistics", DataGroup("ZNormalize_use_global_statistics", [True, False]))
            .parameter("p", preprocessor, lambda v: [("ZNormalize", v["use_global_statistics"])])
            .suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Literal(("ZNormalize", None)))),

            "Preprocessor_Sequence": DSL()  # we might want to consider to have no preprocessor at all
            # .parameter("p", preprocessor)
            .parameter("ps", preprocessor_tuple, lambda v: [()])  # [(v["p"],), (None,)])
            # .argument("x", Constructor("preprocessor", Var("p")))
            .suffix(Constructor("preprocessor_sequence", Var("ps"))),

            "Preprocessor_Sequence_Cons": DSL()
            .parameter("p", preprocessor)
            .parameter("pps", preprocessor_tuple)
            .parameter_constraint(lambda v: len(v["pps"]) > 0 and (v["pps"][0] == v["p"] or v["pps"][0] is None))
            .parameter("ps", preprocessor_tuple, lambda v: [v["pps"][1:]])
            .argument("x", Constructor("preprocessor", Var("p")))
            .argument("xs", Constructor("preprocessor_sequence", Var("ps")))
            .suffix(Constructor("preprocessor_sequence", Var("pps"))),

            "UModel": DSL()
            .parameter("bd", maybe_dimension)
            .parameter("bk", maybe_kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("c_stride", convolution_stride)
            .parameter("c_padding", convolution_padding)
            .parameter("c_dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("e", normalization_eps)
            .parameter("norm", normalization)
            .parameter("m_stride", maxpool_stride)
            .parameter("m_padding", maxpool_padding)
            .parameter("m_dilation", maxpool_dilation)
            .parameter("first_d", dropout_p)
            .parameter("first_af", activation_function)
            .parameter("first_conv", convolution)
            .parameter("first_c_stride", convolution_stride)
            .parameter("first_c_padding", convolution_padding)
            .parameter("first_c_dilation", convolution_dilation)
            .parameter("first_b", bias)
            .parameter("first_e", normalization_eps)
            .parameter("first_norm", normalization)
            .parameter("first_m_stride", maxpool_stride)
            .parameter("first_m_padding", maxpool_padding)
            .parameter("first_m_dilation", maxpool_dilation)
            .parameter("fc_k", kernel_size)
            .parameter("fc_conv", convolution)
            .parameter("fc_stride", convolution_stride)
            .parameter("fc_padding", convolution_padding)
            .parameter("fc_dilation", convolution_dilation)
            .parameter("fc_b", bias)
            .parameter("mlp_in", dimension)
            .parameter("mlp_out", dimension)
            .parameter("mlp_b", bias)
            .parameter("dds", dimension_list)
            .parameter("kks", kernel_size_list)
            .parameter("mms", maxpool_size_list)
            .parameter_constraint(lambda v: len(v["dds"]) > 1 and (len(v["dds"]) == len(v["kks"]) == len(v["mms"]))) # since this should be always instantiated by suffix, this predicate should not be necessary
            .parameter("loss", loss)
            .parameter("preps", preprocessor_tuple)
            .argument("loss_f", Constructor("loss_function") & Var("loss"))
            .argument("preprocessors", Constructor("preprocessor_sequence", Var("preps")))
            .argument("u",
                      Constructor("u_classifier",
                                              Constructor("dimensions", Var("dds"))
                                              & Constructor("kernel_sizes", Var("kks"))
                                              & Constructor("maxpool_sizes", Var("mms"))
                                              )
                                  & Constructor("u_first_level", Constructor("convolution", Var("first_conv"))
                                                & Constructor("convolution_stride", Var("first_c_stride"))
                                                & Constructor("convolution_padding", Var("first_c_padding"))
                                                & Constructor("convolution_dilation", Var("first_c_dilation"))
                                                & Constructor("bias", Var("first_b"))
                                                & Constructor("activation", Var("first_af"))
                                                & Constructor("dropout_p", Var("first_d"))
                                                & Constructor("normalization", Var("first_norm"))
                                                & Constructor("normalization_epsilon", Var("first_e"))
                                                & Constructor("maxpool_stride", Var("first_m_stride"))
                                                & Constructor("maxpool_padding", Var("first_m_padding"))
                                                & Constructor("maxpool_dilation", Var("first_m_dilation"))
                                                )
                                  & Constructor("bottleneck",
                                                Constructor("in_and_out", Var("bd"))
                                                & Constructor("kernel_size", Var("bk"))
                                                )
                                  & Constructor("homogeneous",
                                                Constructor("convolution", Var("conv"))
                                                & Constructor("convolution_stride", Var("c_stride"))
                                                & Constructor("convolution_padding", Var("c_padding"))
                                                & Constructor("convolution_dilation", Var("c_dilation"))
                                                & Constructor("bias", Var("b"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("normalization", Var("norm"))
                                                & Constructor("normalization_epsilon", Var("e"))
                                                & Constructor("maxpool_stride", Var("m_stride"))
                                                & Constructor("maxpool_padding", Var("m_padding"))
                                                & Constructor("maxpool_dilation", Var("m_dilation"))
                                                )
                                  & Constructor("u_final_conv",
                                                Constructor("kernel_size", Var("fc_k"))
                                                & Constructor("convolution", Var("fc_conv"))
                                                & Constructor("convolution_stride", Var("fc_stride"))
                                                & Constructor("convolution_padding", Var("fc_padding"))
                                                & Constructor("convolution_dilation", Var("fc_dilation"))
                                                & Constructor("bias", Var("fc_b"))
                                                )
                                  & Constructor("u_linear_classifier",
                                                Constructor("linear_layer",
                                                            Constructor("input", Var("mlp_in"))
                                                            & Constructor("output", Var("mlp_out"))
                                                            & Constructor("bias", Var("mlp_b"))
                                                            )
                                                )
                      )
            .suffix(Constructor("u_model",
                                Constructor("u_classifier",
                                            Constructor("dimensions", Var("dds"))
                                            & Constructor("kernel_sizes", Var("kks"))
                                            & Constructor("maxpool_sizes", Var("mms"))
                                            )
                                & Constructor("u_first_level", Constructor("convolution", Var("first_conv"))
                                              & Constructor("convolution", Literal(None))
                                              & Constructor("convolution_stride", Var("first_c_stride"))
                                              & Constructor("convolution_stride", Literal(None))
                                              & Constructor("convolution_padding", Var("first_c_padding"))
                                              & Constructor("convolution_padding", Literal(None))
                                              & Constructor("convolution_dilation", Var("first_c_dilation"))
                                              & Constructor("convolution_dilation", Literal(None))
                                              & Constructor("bias", Var("first_b"))
                                              & Constructor("bias", Literal(None))
                                              & Constructor("activation", Var("first_af"))
                                              & Constructor("activation", Literal(None))
                                              & Constructor("dropout_p", Var("first_d"))
                                              & Constructor("dropout_p", Literal(None))
                                              & Constructor("normalization", Var("first_norm"))
                                              & Constructor("normalization", Literal(None))
                                              & Constructor("normalization_epsilon", Var("first_e"))
                                              & Constructor("normalization_epsilon", Literal(None))
                                              & Constructor("maxpool_stride", Var("first_m_stride"))
                                              & Constructor("maxpool_stride", Literal(None))
                                              & Constructor("maxpool_padding", Var("first_m_padding"))
                                              & Constructor("maxpool_padding", Literal(None))
                                              & Constructor("maxpool_dilation", Var("first_m_dilation"))
                                              & Constructor("maxpool_dilation", Literal(None))
                                              )
                                & Constructor("bottleneck",
                                              Constructor("in_and_out", Var("bd"))
                                              & Constructor("kernel_size", Var("bk"))
                                              )
                                & Constructor("homogeneous",
                                              Constructor("convolution", Var("conv"))
                                              & Constructor("convolution", Literal(None))
                                              & Constructor("convolution_stride", Var("c_stride"))
                                              & Constructor("convolution_stride", Literal(None))
                                              & Constructor("convolution_padding", Var("c_padding"))
                                              & Constructor("convolution_padding", Literal(None))
                                              & Constructor("convolution_dilation", Var("c_dilation"))
                                              & Constructor("convolution_dilation", Literal(None))
                                              & Constructor("bias", Var("b"))
                                              & Constructor("bias", Literal(None))
                                              & Constructor("activation", Var("af"))
                                              & Constructor("activation", Literal(None))
                                              & Constructor("dropout_p", Var("d"))
                                              & Constructor("dropout_p", Literal(None))
                                              & Constructor("normalization", Var("norm"))
                                              & Constructor("normalization", Literal(None))
                                              & Constructor("normalization_epsilon", Var("e"))
                                              & Constructor("normalization_epsilon", Literal(None))
                                              & Constructor("maxpool_stride", Var("m_stride"))
                                              & Constructor("maxpool_stride", Literal(None))
                                              & Constructor("maxpool_padding", Var("m_padding"))
                                              & Constructor("maxpool_padding", Literal(None))
                                              & Constructor("maxpool_dilation", Var("m_dilation"))
                                              & Constructor("maxpool_dilation", Literal(None))
                                              )
                                & Constructor("u_final_conv",
                                              Constructor("kernel_size", Var("fc_k"))
                                              & Constructor("kernel_size", Literal(None))
                                              & Constructor("convolution", Var("fc_conv"))
                                              & Constructor("convolution", Literal(None))
                                              & Constructor("convolution_stride", Var("fc_stride"))
                                              & Constructor("convolution_stride", Literal(None))
                                              & Constructor("convolution_padding", Var("fc_padding"))
                                              & Constructor("convolution_padding", Literal(None))
                                              & Constructor("convolution_dilation", Var("fc_dilation"))
                                              & Constructor("convolution_dilation", Literal(None))
                                              & Constructor("bias", Var("fc_b"))
                                              & Constructor("bias", Literal(None))
                                              )
                                & Constructor("u_linear_classifier",
                                              Constructor("linear_layer",
                                                          Constructor("input", Var("mlp_in"))
                                                          & Constructor("input", Literal(None))
                                                          & Constructor("output", Var("mlp_out"))
                                                          & Constructor("output", Literal(None))
                                                          & Constructor("bias", Var("mlp_b"))
                                                          & Constructor("bias", Literal(None))
                                                          )
                                              )
                                & Constructor("loss_function", Var("loss"))
                                & Constructor("loss_function", Literal(None))
                                & Constructor("preprocessors", Var("preps"))
                                )
                    ),

            "NoSampler": DSL()
            .parameter("replacement", DataGroup("Sampler_Replacement", [True, False]))
            .parameter("num_samples", n_samples)
            .suffix(Constructor("Sampler", Literal("NoSampler")
                                & Constructor("replacement", Var("replacement"))
                                & Constructor("replacement", Literal(None))
                                & Constructor("num_samples", Var("num_samples"))
                                & Constructor("num_samples", Literal(None))
                                )
                    ),

            "RandomSampler": DSL()
            .parameter("replacement", DataGroup("Sampler_Replacement", [True, False]))
            .parameter("num_samples", n_samples)
            .suffix(Constructor("Sampler", Literal("RandomSampler")
                                & Constructor("replacement", Var("replacement"))
                                & Constructor("replacement", Literal(None))
                                & Constructor("num_samples", Var("num_samples"))
                                & Constructor("num_samples", Literal(None))
                                )
                    ),

            "ABC_Dataset": DSL()
            .parameter("annotator", DataGroup("ABC_Dataset_Annotator", ["nsrr", "profusion"]))
            .parameter("channels", abc_channels)
            .parameter("num_workers", DataGroup("ABC_Dataset_Num_Workers", self.abc_num_workers))
            .parameter("sample_frequency", DataGroup("ABC_Dataset_Sample_Frequency", self.abc_sample_frequency))
            .parameter("event_mapping", DataGroup("ABC_Dataset_Event_Mapping", self.abc_event_mapping))
            .parameter("online_filtering", DataGroup("ABC_Dataset_Online_Filtering", [True, False]))
            .parameter("total_input", DataGroup("ABC_Dataset_Total_Input", self.abc_total_input))
            .parameter("target_resolution", DataGroup("ABC_Dataset_Target_Resolution", self.abc_target_resolution))
            .suffix(Constructor("Dataset", Literal("ABC_Dataset")
                                & Constructor("annotator", Var("annotator"))
                                & Constructor("channels", Var("channels"))
                                & Constructor("num_workers", Var("num_workers"))
                                & Constructor("sample_frequency", Var("sample_frequency"))
                                & Constructor("event_mapping", Var("event_mapping"))
                                & Constructor("online_filtering", Var("online_filtering"))
                                & Constructor("total_input", Var("total_input"))
                                & Constructor("target_resolution", Var("target_resolution"))
                                )
                    ),

            "DataLoader": DSL()
            .parameter("annotator", DataGroup("ABC_Dataset_Annotator", ["nsrr", "profusion"]))
            .parameter("channels", abc_channels)
            .parameter("num_workers", DataGroup("ABC_Dataset_Num_Workers", self.abc_num_workers))
            .parameter("sample_frequency", DataGroup("ABC_Dataset_Sample_Frequency", self.abc_sample_frequency))
            .parameter("event_mapping", DataGroup("ABC_Dataset_Event_Mapping", self.abc_event_mapping))
            .parameter("online_filtering", DataGroup("ABC_Dataset_Online_Filtering", [True, False]))
            .parameter("total_input", DataGroup("ABC_Dataset_Total_Input", self.abc_total_input))
            .parameter("target_resolution", DataGroup("ABC_Dataset_Target_Resolution", self.abc_target_resolution))
            .parameter("replacement", DataGroup("Sampler_Replacement", [True, False]))
            .parameter("num_samples", n_samples)
            .parameter("sampler", DataGroup("Sampler", ["NoSampler", "RandomSampler"]))
            .parameter("batch_size", DataGroup("DataLoader_Batch_Size", self.batch_size))
            .argument("s", Constructor("Sampler", Var("sampler")
                                       & Constructor("replacement", Var("replacement"))
                                       & Constructor("num_samples", Var("num_samples"))
                                       )
                      )
            .argument("d", Constructor("Dataset", Literal("ABC_Dataset")
                                & Constructor("annotator", Var("annotator"))
                                & Constructor("channels", Var("channels"))
                                & Constructor("num_workers", Var("num_workers"))
                                & Constructor("sample_frequency", Var("sample_frequency"))
                                & Constructor("event_mapping", Var("event_mapping"))
                                & Constructor("online_filtering", Var("online_filtering"))
                                & Constructor("total_input", Var("total_input"))
                                & Constructor("target_resolution", Var("target_resolution"))
                                )
                      )
            .suffix(Constructor("dataloader",
                                Constructor("Sampler",
                                            Var("sampler")
                                            & Constructor("replacement", Var("replacement"))
                                            & Constructor("num_samples", Var("num_samples"))
                                            )
                                & Constructor("Dataset",
                                              Literal("ABC_Dataset")
                                              & Constructor("annotator", Var("annotator"))
                                              & Constructor("channels", Var("channels"))
                                              & Constructor("num_workers", Var("num_workers"))
                                              & Constructor("sample_frequency", Var("sample_frequency"))
                                              & Constructor("event_mapping", Var("event_mapping"))
                                              & Constructor("online_filtering", Var("online_filtering"))
                                              & Constructor("total_input", Var("total_input"))
                                              & Constructor("target_resolution", Var("target_resolution"))
                                              )
                                & Constructor("batch_size", Var("batch_size"))
                                )
                    ),

            "Adagrad": DSL()
            .parameter("lr", optimizer_learning_rate)
            .parameter("lr_d", optimizer_learning_rate_decay)
            .parameter("w_d", optimizer_weight_decay)
            .parameter("i_a_v", optimizer_initial_accumulator_value)
            .parameter("eps", optimizer_eps)
            .parameter("p_none", maybe_optimizer)
            .parameter_constraint(lambda v: v["p_none"][0] == "Adagrad"
                                            and (v["p_none"][1] == v["lr"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["lr_d"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["w_d"] or v["p_none"][3] is None)
                                            and (v["p_none"][4] == v["i_a_v"] or v["p_none"][4] is None)
                                            and (v["p_none"][5] == v["eps"] or v["p_none"][5] is None)
                                  )
            .suffix(Constructor("optimizer", Var("p_none"))),

            "Adam": DSL()
            .parameter("lr", optimizer_learning_rate)
            .parameter("beta", optimizer_beta)
            .parameter("eps", optimizer_eps)
            .parameter("w_d", optimizer_weight_decay)
            .parameter("amsgrad", DataGroup("Optimizer_Adam_Amsgrad", [True, False]))
            .parameter("p_none", maybe_optimizer)
            .parameter_constraint(lambda v: v["p_none"][0] == "Adam"
                                            and (v["p_none"][1] == v["lr"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["beta"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["eps"] or v["p_none"][3] is None)
                                            and (v["p_none"][4] == v["w_d"] or v["p_none"][4] is None)
                                            and (v["p_none"][5] == v["amsgrad"] or v["p_none"][5] is None)
                                  )
            .suffix(Constructor("optimizer", Var("p_none"))),

            "AdamW": DSL()
            .parameter("lr", optimizer_learning_rate)
            .parameter("beta", optimizer_beta)
            .parameter("eps", optimizer_eps)
            .parameter("w_d", optimizer_weight_decay)
            .parameter("amsgrad", DataGroup("Optimizer_AdamW_Amsgrad", [True, False]))
            .parameter("p_none", maybe_optimizer)
            .parameter_constraint(lambda v: v["p_none"][0] == "AdamW"
                                            and (v["p_none"][1] == v["lr"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["beta"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["eps"] or v["p_none"][3] is None)
                                            and (v["p_none"][4] == v["w_d"] or v["p_none"][4] is None)
                                            and (v["p_none"][5] == v["amsgrad"] or v["p_none"][5] is None)
                                  )
            .suffix(Constructor("optimizer", Var("p_none"))),

            "Adamax": DSL()
            .parameter("lr", optimizer_learning_rate)
            .parameter("beta", optimizer_beta)
            .parameter("eps", optimizer_eps)
            .parameter("w_d", optimizer_weight_decay)
            .parameter("p_none", maybe_optimizer)
            .parameter_constraint(lambda v: v["p_none"][0] == "Adamax"
                                            and (v["p_none"][1] == v["lr"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["beta"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["eps"] or v["p_none"][3] is None)
                                            and (v["p_none"][4] == v["w_d"] or v["p_none"][4] is None)
                                  )
            .suffix(Constructor("optimizer", Var("p_none"))),

            "SGD": DSL()
            .parameter("lr", optimizer_learning_rate)
            .parameter("momentum", optimizer_momentum)
            .parameter("dampening", optimizer_dampening)
            .parameter("w_d", optimizer_weight_decay)
            .parameter("nesterov", DataGroup("Optimizer_SGD_nesterov", [True, False]))
            .parameter("p_none", maybe_optimizer)
            .parameter_constraint(lambda v: v["p_none"][0] == "SGD"
                                            and (v["p_none"][1] == v["lr"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["momentum"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["dampening"] or v["p_none"][3] is None)
                                            and (v["p_none"][4] == v["w_d"] or v["p_none"][4] is None)
                                            and (v["p_none"][5] == v["nesterov"] or v["p_none"][5] is None)
                                  )
            .suffix(Constructor("optimizer", Var("p_none"))),

            "LinearLR": DSL()
            .parameter("start_factor", lr_scheduler_start_factor)
            .parameter("end_factor", lr_scheduler_end_factor)
            .parameter("total_iters", lr_scheduler_total_iters)
            .parameter("last_epoch", lr_scheduler_last_epoch)
            .parameter("p_none", maybe_lr_scheduler)
            .parameter_constraint(lambda v: v["p_none"][0] == "LinearLR"
                                            and (v["p_none"][1] == v["start_factor"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["end_factor"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["total_iters"] or v["p_none"][3] is None)
                                            and (v["p_none"][4] == v["last_epoch"] or v["p_none"][4] is None)
                                  )
            .suffix(Constructor("lr_scheduler", Var("p_none"))),

            "StepLR": DSL()
            .parameter("step_size", lr_scheduler_step_size)
            .parameter("gamma", lr_scheduler_gamma)
            .parameter("last_epoch", lr_scheduler_last_epoch)
            .parameter("p_none", maybe_lr_scheduler)
            .parameter_constraint(lambda v: v["p_none"][0] == "StepLR"
                                            and (v["p_none"][1] == v["step_size"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["gamma"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["last_epoch"] or v["p_none"][3] is None)
                                  )
            .suffix(Constructor("lr_scheduler", Var("p_none"))),

            "ExponentialLR": DSL()
            .parameter("gamma", lr_scheduler_gamma)
            .parameter("last_epoch", lr_scheduler_last_epoch)
            .parameter("p_none", maybe_lr_scheduler)
            .parameter_constraint(lambda v: v["p_none"][0] == "ExponentialLR"
                                            and (v["p_none"][1] == v["gamma"] or v["p_none"][2] is None)
                                            and (v["p_none"][2] == v["last_epoch"] or v["p_none"][2] is None)
                                  )
            .suffix(Constructor("lr_scheduler", Var("p_none"))),

            "MulticlassTrainer": DSL()
            .parameter("bd", maybe_dimension)
            .parameter("bk", maybe_kernel_size)
            .parameter("d", dropout_p)
            .parameter("af", activation_function)
            .parameter("conv", convolution)
            .parameter("c_stride", convolution_stride)
            .parameter("c_padding", convolution_padding)
            .parameter("c_dilation", convolution_dilation)
            .parameter("b", bias)
            .parameter("e", normalization_eps)
            .parameter("norm", normalization)
            .parameter("m_stride", maxpool_stride)
            .parameter("m_padding", maxpool_padding)
            .parameter("m_dilation", maxpool_dilation)
            .parameter("first_d", dropout_p)
            .parameter("first_af", activation_function)
            .parameter("first_conv", convolution)
            .parameter("first_c_stride", convolution_stride)
            .parameter("first_c_padding", convolution_padding)
            .parameter("first_c_dilation", convolution_dilation)
            .parameter("first_b", bias)
            .parameter("first_e", normalization_eps)
            .parameter("first_norm", normalization)
            .parameter("first_m_stride", maxpool_stride)
            .parameter("first_m_padding", maxpool_padding)
            .parameter("first_m_dilation", maxpool_dilation)
            .parameter("fc_k", kernel_size)
            .parameter("fc_conv", convolution)
            .parameter("fc_stride", convolution_stride)
            .parameter("fc_padding", convolution_padding)
            .parameter("fc_dilation", convolution_dilation)
            .parameter("fc_b", bias)
            .parameter("mlp_in", dimension)
            .parameter("mlp_out", dimension)
            .parameter("mlp_b", bias)
            .parameter("dds", dimension_list)
            .parameter("kks", kernel_size_list)
            .parameter("mms", maxpool_size_list)
            .parameter_constraint(lambda v: len(v["dds"]) > 1 and (len(v["dds"]) == len(v["kks"]) == len(v["mms"])))  # since this should be always instantiated by suffix, this predicate should not be necessary
            .parameter("loss", loss)
            .parameter("preps_none", maybe_preprocessor_tuple)
            .parameter("preps", preprocessor_tuple, lambda v: preprocessor_tuple.from_maybe_preprocessor_tuple(v["preps_none"]))
            .parameter("annotator", DataGroup("ABC_Dataset_Annotator", ["nsrr", "profusion"]))
            .parameter("channels", abc_channels)
            .parameter("num_workers", DataGroup("ABC_Dataset_Num_Workers", self.abc_num_workers))
            .parameter("sample_frequency", DataGroup("ABC_Dataset_Sample_Frequency", self.abc_sample_frequency))
            .parameter("event_mapping", DataGroup("ABC_Dataset_Event_Mapping", self.abc_event_mapping))
            .parameter("online_filtering", DataGroup("ABC_Dataset_Online_Filtering", [True, False]))
            .parameter("total_input", DataGroup("ABC_Dataset_Total_Input", self.abc_total_input))
            .parameter("target_resolution", DataGroup("ABC_Dataset_Target_Resolution", self.abc_target_resolution))
            .parameter("replacement", DataGroup("Sampler_Replacement", [True, False]))
            .parameter("num_samples", n_samples)
            .parameter("sampler", DataGroup("Sampler", ["NoSampler", "RandomSampler"]))
            .parameter("batch_size", DataGroup("DataLoader_Batch_Size", self.batch_size))
            .parameter("opti_none", maybe_optimizer)
            .parameter("opti", maybe_optimizer, lambda v: maybe_optimizer.unfold_none(v["opti_none"]))
            .parameter("lr_sched_none", maybe_lr_scheduler)
            .parameter("lr_sched", maybe_lr_scheduler, lambda v: maybe_lr_scheduler.unfold_none(v["lr_sched_none"]))
            .argument("model",
                      Constructor("u_model",
                                Constructor("u_classifier",
                                            Constructor("dimensions", Var("dds"))
                                            & Constructor("kernel_sizes", Var("kks"))
                                            & Constructor("maxpool_sizes", Var("mms"))
                                            )
                                & Constructor("u_first_level", Constructor("convolution", Var("first_conv"))
                                              & Constructor("convolution", Literal(None))
                                              & Constructor("convolution_stride", Var("first_c_stride"))
                                              & Constructor("convolution_stride", Literal(None))
                                              & Constructor("convolution_padding", Var("first_c_padding"))
                                              & Constructor("convolution_padding", Literal(None))
                                              & Constructor("convolution_dilation", Var("first_c_dilation"))
                                              & Constructor("convolution_dilation", Literal(None))
                                              & Constructor("bias", Var("first_b"))
                                              & Constructor("bias", Literal(None))
                                              & Constructor("activation", Var("first_af"))
                                              & Constructor("activation", Literal(None))
                                              & Constructor("dropout_p", Var("first_d"))
                                              & Constructor("dropout_p", Literal(None))
                                              & Constructor("normalization", Var("first_norm"))
                                              & Constructor("normalization", Literal(None))
                                              & Constructor("normalization_epsilon", Var("first_e"))
                                              & Constructor("normalization_epsilon", Literal(None))
                                              & Constructor("maxpool_stride", Var("first_m_stride"))
                                              & Constructor("maxpool_stride", Literal(None))
                                              & Constructor("maxpool_padding", Var("first_m_padding"))
                                              & Constructor("maxpool_padding", Literal(None))
                                              & Constructor("maxpool_dilation", Var("first_m_dilation"))
                                              & Constructor("maxpool_dilation", Literal(None))
                                              )
                                & Constructor("bottleneck",
                                              Constructor("in_and_out", Var("bd"))
                                              & Constructor("kernel_size", Var("bk"))
                                              )
                                & Constructor("homogeneous",
                                              Constructor("convolution", Var("conv"))
                                              & Constructor("convolution", Literal(None))
                                              & Constructor("convolution_stride", Var("c_stride"))
                                              & Constructor("convolution_stride", Literal(None))
                                              & Constructor("convolution_padding", Var("c_padding"))
                                              & Constructor("convolution_padding", Literal(None))
                                              & Constructor("convolution_dilation", Var("c_dilation"))
                                              & Constructor("convolution_dilation", Literal(None))
                                              & Constructor("bias", Var("b"))
                                              & Constructor("bias", Literal(None))
                                              & Constructor("activation", Var("af"))
                                              & Constructor("activation", Literal(None))
                                              & Constructor("dropout_p", Var("d"))
                                              & Constructor("dropout_p", Literal(None))
                                              & Constructor("normalization", Var("norm"))
                                              & Constructor("normalization", Literal(None))
                                              & Constructor("normalization_epsilon", Var("e"))
                                              & Constructor("normalization_epsilon", Literal(None))
                                              & Constructor("maxpool_stride", Var("m_stride"))
                                              & Constructor("maxpool_stride", Literal(None))
                                              & Constructor("maxpool_padding", Var("m_padding"))
                                              & Constructor("maxpool_padding", Literal(None))
                                              & Constructor("maxpool_dilation", Var("m_dilation"))
                                              & Constructor("maxpool_dilation", Literal(None))
                                              )
                                & Constructor("u_final_conv",
                                              Constructor("kernel_size", Var("fc_k"))
                                              & Constructor("kernel_size", Literal(None))
                                              & Constructor("convolution", Var("fc_conv"))
                                              & Constructor("convolution", Literal(None))
                                              & Constructor("convolution_stride", Var("fc_stride"))
                                              & Constructor("convolution_stride", Literal(None))
                                              & Constructor("convolution_padding", Var("fc_padding"))
                                              & Constructor("convolution_padding", Literal(None))
                                              & Constructor("convolution_dilation", Var("fc_dilation"))
                                              & Constructor("convolution_dilation", Literal(None))
                                              & Constructor("bias", Var("fc_b"))
                                              & Constructor("bias", Literal(None))
                                              )
                                & Constructor("u_linear_classifier",
                                              Constructor("linear_layer",
                                                          Constructor("input", Var("mlp_in"))
                                                          & Constructor("input", Literal(None))
                                                          & Constructor("output", Var("mlp_out"))
                                                          & Constructor("output", Literal(None))
                                                          & Constructor("bias", Var("mlp_b"))
                                                          & Constructor("bias", Literal(None))
                                                          )
                                              )
                                & Constructor("loss_function", Var("loss"))
                                & Constructor("loss_function", Literal(None))
                                & Constructor("preprocessors", Var("preps"))
                                )
                      )
            .argument("dataloader",
                      Constructor("dataloader",
                                  Constructor("Sampler",
                                              Var("sampler")
                                              & Constructor("replacement", Var("replacement"))
                                              & Constructor("num_samples", Var("num_samples"))
                                              )
                                  & Constructor("Dataset",
                                                Literal("ABC_Dataset")
                                                & Constructor("annotator", Var("annotator"))
                                                & Constructor("channels", Var("channels"))
                                                & Constructor("num_workers", Var("num_workers"))
                                                & Constructor("sample_frequency", Var("sample_frequency"))
                                                & Constructor("event_mapping", Var("event_mapping"))
                                                & Constructor("online_filtering", Var("online_filtering"))
                                                & Constructor("total_input", Var("total_input"))
                                                & Constructor("target_resolution", Var("target_resolution"))
                                                )
                                  & Constructor("batch_size", Var("batch_size"))
                                  )
                      )
            .argument("optimizer", Constructor("optimizer", Var("opti")))
            .argument("lr_scheduler", Constructor("lr_scheduler", Var("lr_sched")))
            .suffix(Constructor("trainer",
                                Constructor("u_model",
                                            Constructor("u_classifier",
                                                        Constructor("dimensions", Var("dds"))
                                                        & Constructor("kernel_sizes", Var("kks"))
                                                        & Constructor("maxpool_sizes", Var("mms"))
                                                        )
                                            & Constructor("u_first_level", Constructor("convolution", Var("first_conv"))
                                                          & Constructor("convolution", Literal(None))
                                                          & Constructor("convolution_stride", Var("first_c_stride"))
                                                          & Constructor("convolution_stride", Literal(None))
                                                          & Constructor("convolution_padding", Var("first_c_padding"))
                                                          & Constructor("convolution_padding", Literal(None))
                                                          & Constructor("convolution_dilation", Var("first_c_dilation"))
                                                          & Constructor("convolution_dilation", Literal(None))
                                                          & Constructor("bias", Var("first_b"))
                                                          & Constructor("bias", Literal(None))
                                                          & Constructor("activation", Var("first_af"))
                                                          & Constructor("activation", Literal(None))
                                                          & Constructor("dropout_p", Var("first_d"))
                                                          & Constructor("dropout_p", Literal(None))
                                                          & Constructor("normalization", Var("first_norm"))
                                                          & Constructor("normalization", Literal(None))
                                                          & Constructor("normalization_epsilon", Var("first_e"))
                                                          & Constructor("normalization_epsilon", Literal(None))
                                                          & Constructor("maxpool_stride", Var("first_m_stride"))
                                                          & Constructor("maxpool_stride", Literal(None))
                                                          & Constructor("maxpool_padding", Var("first_m_padding"))
                                                          & Constructor("maxpool_padding", Literal(None))
                                                          & Constructor("maxpool_dilation", Var("first_m_dilation"))
                                                          & Constructor("maxpool_dilation", Literal(None))
                                                          )
                                            & Constructor("bottleneck",
                                                          Constructor("in_and_out", Var("bd"))
                                                          & Constructor("kernel_size", Var("bk"))
                                                          )
                                            & Constructor("homogeneous",
                                                          Constructor("convolution", Var("conv"))
                                                          & Constructor("convolution", Literal(None))
                                                          & Constructor("convolution_stride", Var("c_stride"))
                                                          & Constructor("convolution_stride", Literal(None))
                                                          & Constructor("convolution_padding", Var("c_padding"))
                                                          & Constructor("convolution_padding", Literal(None))
                                                          & Constructor("convolution_dilation", Var("c_dilation"))
                                                          & Constructor("convolution_dilation", Literal(None))
                                                          & Constructor("bias", Var("b"))
                                                          & Constructor("bias", Literal(None))
                                                          & Constructor("activation", Var("af"))
                                                          & Constructor("activation", Literal(None))
                                                          & Constructor("dropout_p", Var("d"))
                                                          & Constructor("dropout_p", Literal(None))
                                                          & Constructor("normalization", Var("norm"))
                                                          & Constructor("normalization", Literal(None))
                                                          & Constructor("normalization_epsilon", Var("e"))
                                                          & Constructor("normalization_epsilon", Literal(None))
                                                          & Constructor("maxpool_stride", Var("m_stride"))
                                                          & Constructor("maxpool_stride", Literal(None))
                                                          & Constructor("maxpool_padding", Var("m_padding"))
                                                          & Constructor("maxpool_padding", Literal(None))
                                                          & Constructor("maxpool_dilation", Var("m_dilation"))
                                                          & Constructor("maxpool_dilation", Literal(None))
                                                          )
                                            & Constructor("u_final_conv",
                                                          Constructor("kernel_size", Var("fc_k"))
                                                          & Constructor("kernel_size", Literal(None))
                                                          & Constructor("convolution", Var("fc_conv"))
                                                          & Constructor("convolution", Literal(None))
                                                          & Constructor("convolution_stride", Var("fc_stride"))
                                                          & Constructor("convolution_stride", Literal(None))
                                                          & Constructor("convolution_padding", Var("fc_padding"))
                                                          & Constructor("convolution_padding", Literal(None))
                                                          & Constructor("convolution_dilation", Var("fc_dilation"))
                                                          & Constructor("convolution_dilation", Literal(None))
                                                          & Constructor("bias", Var("fc_b"))
                                                          & Constructor("bias", Literal(None))
                                                          )
                                            & Constructor("u_linear_classifier",
                                                          Constructor("linear_layer",
                                                                      Constructor("input", Var("mlp_in"))
                                                                      & Constructor("input", Literal(None))
                                                                      & Constructor("output", Var("mlp_out"))
                                                                      & Constructor("output", Literal(None))
                                                                      & Constructor("bias", Var("mlp_b"))
                                                                      & Constructor("bias", Literal(None))
                                                                      )
                                                          )
                                            & Constructor("loss_function", Var("loss"))
                                            & Constructor("loss_function", Literal(None))
                                            & Constructor("preprocessors", Var("preps_none"))
                                            )
                                & Constructor("dataloader",
                                              Constructor("Sampler",
                                                          Var("sampler")
                                                          & Constructor("replacement", Var("replacement"))
                                                          & Constructor("replacement", Literal(None))
                                                          & Constructor("num_samples", Var("num_samples"))
                                                          & Constructor("num_samples", Literal(None))
                                                          )
                                              & Constructor("Dataset",
                                                            Literal("ABC_Dataset")
                                                            & Constructor("annotator", Var("annotator"))
                                                            & Constructor("annotator", Literal(None))
                                                            & Constructor("channels", Var("channels"))
                                                            & Constructor("channels", Literal(None))
                                                            & Constructor("num_workers", Var("num_workers"))
                                                            & Constructor("num_workers",  Literal(None))
                                                            & Constructor("sample_frequency", Var("sample_frequency"))
                                                            & Constructor("sample_frequency", Literal(None))
                                                            & Constructor("event_mapping", Var("event_mapping"))
                                                            & Constructor("event_mapping", Literal(None))
                                                            & Constructor("online_filtering", Var("online_filtering"))
                                                            & Constructor("online_filtering", Literal(None))
                                                            & Constructor("total_input", Var("total_input"))
                                                            & Constructor("total_input", Literal(None))
                                                            & Constructor("target_resolution", Var("target_resolution"))
                                                            & Constructor("target_resolution", Literal(None))
                                                            )
                                              & Constructor("batch_size", Var("batch_size"))
                                              & Constructor("batch_size", Literal(None))
                                              )
                                & Constructor("optimizer", Var("opti_none"))
                                & Constructor("lr_scheduler", Var("lr_sched_none"))
                                )
                    ),


        }

    def pretty_term_algebra(self):
        return {
            "ReLu": "ReLu()",
            "ELU": "ELU()",
            "Tanh": "Tanh()",
            "BatchNorm1d": (lambda n, e: f"BatchNorm1d({n}, {e})"),
            "ChannelWiseNorm": (lambda n, e: f"ChannelWiseNormalization({n}, {e})"),
            "Conv1dLayerNorm": (lambda n, e: f"Conv1dLayerNorm({n}, {e})"),
            "Dropout1d": (lambda d: f"Dropout1d({d})"),
            "Maxpool1d": (lambda n, s, p, d: f"MaxPool1d({n}, {s}, {p}, {d})"),
            "Upsample1d": (lambda n: f"Upsample({n})"),
            "Conv1d": (lambda i, o, k, s, p, d, b: f"Conv1d({i}, {o}, {k}, {s}, {p}, {d}, {b})"),
            "DepthwiseSeparableConv1d": (lambda i, o, k, s, p, d, b:
                                         f"DepthwiseSeparableConv1d({i}, {o}, {k}, {s}, {p}, {d}, {b})"),
            "ConvBlock": (lambda i, o, k, d, af, c, s, p, di, b, n, e, activation, dropout, c1, c2, norm:
                          f"Conv_Block({activation}, {dropout}, {c1}, {c2}, {norm})"),
            "Encoder": (lambda i, o, k, d, af, c, s, p, di, b, e, n, m, ms, mpa, md, mp, cb: f"Encoder({cb}, {mp})"),
            "Decoder": (lambda i, o, k, d, af, c, s, p, di, b, e, n, m, mp, cb: f"Decoder({cb}, {mp})"),
            "UStructure": (lambda i, out_enc, in_dec, k1, k2, d, af, c, s, p, di, b, e, n, m, mst, mpa, md,
                              ds, ks, ms, enc, dec, cb: f"U_Structure({enc}, {dec}, {cb})"),

            "UStructure_Cons": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, s, p, di, b, e, n, m, mst, mpa, md,
                                            dds, ds, kks, ks, mms, ms, enc, dec, u_model:
                                f"U_Model_Structure({enc}, {dec}, {u_model})"),
            "LinearLayer": (lambda i, o, b: f"LinearLayer({i}, {o}, {b})"),
            "UClassifier": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, conv, c_stride, c_padding, c_dilation, b, e, norm, m, m_stride, m_padding, m_dilation,
                              first_d, first_af, first_conv, first_c_stride, first_c_padding, first_c_dilation, first_b, first_e, first_norm, first_m_stride, first_m_padding, first_m_dilation,
                              fc_k, fc_conv, fc_stride, fc_padding, fc_dilation, fc_b, mlp_in, mlp_out, mlp_b,
                              dds, ds, kks, ks, mms, ms, enc, dec, u, dc, mlp:
                       f"U_Classifier({enc}, {dec}, {u}, {dc}, {mlp})"),

            "BCEwithLogits": "BCE_with_logits()",

            "CrossEntropy": "CrossEntropy()",

            "MAE": "MAE()",

            "MSE": "MSE()",

            "ChannelSampler": lambda n, p: f"ChannelSampler({n})",

            "Crop": lambda ti, sr, w, p_none: f"Crop({ti}, {sr}, {w})",

            "EmpiricalClipScaler": lambda q, s, p_none: f"EmpiricalClipScaler({q}, {s})",

            "FIR": lambda sr, ch, fp, zp, p_none: f"FIR({sr}, {ch}, {fp}, {zp})",

            "Normalize": "Normalize()",

            "RobustScaler": lambda lq, uq, p_none: f"RobustScaler({lq}, {uq})",

            "Spectogram": lambda n_fft, hl, wl, els, p_none: f"Spectogram({n_fft}, {hl}, {wl}, {els})",

            "ZNormalize": lambda ugs, p: f"ZNormalize({ugs})",

            "Preprocessor_Sequence": lambda ps: f"[]",

            "Preprocessor_Sequence_Cons": lambda p, pps, ps, x, xs: f"({x} :: {xs})",

            "UModel": (lambda bd, bk, d, af, conv, c_s, c_p, c_d, b, e, norm,
                               m_s, m_p, m_d, f_d, f_af, f_c, f_c_s, f_c_p, f_c_d, f_b, f_e, f_norm, f_m_s, f_m_p,
                               f_m_d, fc_k, fc_c, fc_s, fc_p, fc_d, fc_b, mlp_in, mlp_out, mlp_b, dds, kks, mms,
                               loss, preps, loss_f, preprocessors, u: f"UModel({loss_f}, {preprocessors}, {u})"),

            "NoSampler": lambda r, n: f"NoSampler",

            "RandomSampler": lambda r, n: f"RandomSampler({r}, {n})",

            "ABC_Dataset": lambda a, c, nw, sf, em, of, ti, tr: f"ABC_Dataset({a}, {c}, {nw}, {sf}, {em}, {of}, {ti}, {tr})",

            "DataLoader": lambda a, ch, nw, sf, em, of, ti, tr, r, ns, sam, bs, s, d: f"DataLoader({s}, {d}, {bs})",

            "Adagrad": lambda lr, lr_d, w_d, i_a_v, eps, p_none: f"Adagrad({lr}, {lr_d}, {w_d}, {i_a_v}, {eps})",

            "Adam": lambda lr, beta, eps, w_d, amsgrad, p_none: f"Adam({lr}, {beta}, {eps}, {w_d}, {amsgrad})",

            "AdamW": lambda lr, beta, eps, w_d, amsgrad, p_none: f"AdamW({lr}, {beta}, {eps}, {w_d}, {amsgrad})",

            "Adamax": lambda lr, beta, eps, w_d, p_none: f"Adamax({lr}, {beta}, {eps}, {w_d})",

            "SGD": lambda lr, m, d, w_d, n, p_none: f"SGD({lr}, {m}, {d}, {w_d}, {n})",

            "LinearLR": lambda sf, ef, ti, le, p_none: f"LinearLR({sf}, {ef}, {ti}, {le})",

            "StepLR": lambda s, g, l, p_none: f"StepLR({s}, {g}, {l})",

            "ExponentialLR": lambda g, l, p_none: f"ExponentialLR({g}, {l})",

            "MulticlassTrainer": (lambda bd, bk, d, af, conv, c_s, c_p, c_d, b, e, norm, m_s, m_p, m_d, f_d, f_af, f_c,
                                         f_c_s, f_c_p, f_c_d, f_b, f_e, f_norm, f_m_s, f_m_p, f_m_d, fc_k, fc_c, fc_s,
                                         fc_p, fc_d, fc_b, mlp_in, mlp_out, mlp_b, dds, kks, mms, loss, preps_none, preps,
                                         annotator, channels, num_w, sample_f, event_map, online_f, total_in,
                                         target_res, replacement, num_s, sampler, batch_size, opti_none, opti,
                                         lr_sched_none, lr_sched,
                                         model, dataloader, optimizer, lr_scheduler:
                                  f"MulticlassTrainer({model}, {dataloader}, {optimizer}, {lr_scheduler})"),
        }

    @staticmethod
    def _conv_block(activation, dropout, c1, c2, norm, x):
        x = c1(x)
        x = norm(x)
        x = activation(x)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        x = dropout(x)
        return x

    @staticmethod
    def _encoder(cb, mp, x):
        y = cb(x)
        x = mp(y)
        if x.shape[-1] < 1:
            raise ValueError("Encoder output is empty after pooling. Reduce the pooling size or number of layers.")
        return x, y

    @staticmethod
    def _decoder(cb, up, x, skip):
        x = up(x)
        output_size = skip.size(2)

        if x.size(2) != output_size:
            diff = output_size - x.size(2)
            x = F.pad(x, (0, diff))  # Apply zero padding to the end of the dimension

        x = torch.cat([x, skip], dim=1)
        x = cb(x)
        return x

    @staticmethod
    def _ustructure(enc, dec, cb, x, return_intermediate=False):
        x, y = enc(x)
        z = cb(x)
        x = dec(z, y)
        if return_intermediate:
            return x, z.flatten(start_dim=1)
        else:
            return x

    @staticmethod
    def _ustructure_cons(enc, dec, u_model, x, return_intermediate=False):
        x, y = enc(x)
        if return_intermediate:
            z, u_intermediate = u_model(x, return_intermediate=return_intermediate)
            x = dec(z, y)
            return x, u_intermediate
        else:
            z = u_model(x, return_intermediate=return_intermediate)
            x = dec(z, y)
            return x

    def _uclassifier(self, enc, dec, u_model, fc, mlp, x, return_intermediate=False):
        x = x.swapaxes(1, 2)
        T = x.shape[-1]
        if return_intermediate:
            x, u_intermediate = self._ustructure_cons(enc, dec, u_model, x, return_intermediate=return_intermediate)
            x = fc(x)
            x = x.mean(dim=2)
            feature_embeddings = x
            x = mlp(x)
            return x, (u_intermediate, feature_embeddings.flatten(start_dim=1))
        else:
            x = self._ustructure_cons(enc, dec, u_model, x, return_intermediate=return_intermediate)
            x = fc(x)
            x = x.mean(dim=2)
            x = mlp(x)
            return x

    def _bce_with_logits(self, pred, target, additional=None):
        (batch_size, _, _) = pred.shape
        return torch.nn.functional.binary_cross_entropy_with_logits(pred.reshape(batch_size, -1),
                                                                    target.reshape(batch_size, -1))

    def _mae(self, pred, target, additional=None):
        pred = pred.reshape(target.shape)
        return torch.nn.functional.l1_loss(pred, target)

    def _mse(self, pred, target, additional=None):
        pred = pred.reshape(target.shape)
        return torch.nn.functional.mse_loss(pred, target)

    def _dataloader(self, num_workers, batch_size, sampler, dataset, patients):
        data = dataset(patients)
        sample = sampler(data)
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=sample is None, sampler=sample,
                                             num_workers=num_workers, pin_memory=False, collate_fn=batch_collate,
                                             drop_last=False, persistent_workers=True)
        return loader, data

    def _train_multiclass(self, u_model, dataloader, optimizer, lr_scheduler):
        # TODO: make this right for ABC-dataset

        # Parameters for this run
        edf_folder = "/Users/felixlaarmann/Downloads/abc/polysomnography"
        epochs = 100

        all_patients = get_edf_files_in_repo(edf_folder, recursive=True)
        train_patients, test_patients = random_split(all_patients, test_frac=0.1)

        train_loader, dataset = dataloader(train_patients)

        model, loss = u_model(dataset.get_classes(), len(dataset.channels))  # n_channel = len(dataset.channels)?

        trainer = MulticlassTrainer(
                epochs=epochs,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                classes=dataset.get_classes(),
                save_every=10,
                loss_function=loss,
                device="cpu"
        )

        losses, cms = trainer.fit(model, train_loader)

        test_loader, _ = dataloader(test_patients)
        test_loss, test_cm = trainer.test(model, test_loader)

        record = {
                "test_loss": test_loss,
                "test_cm": test_cm,
                "train_loss": losses,
                "train_cm": cms,
                "classes": dataset.get_classes(),
        }

        print(record)




    def torch_algebra(self):
        return {
            "ReLu": nn.ReLU(),
            "ELU": nn.ELU(),
            "Tanh": nn.Tanh(),
            "BatchNorm1d": (lambda n, e: nn.BatchNorm1d(n, eps=e)),
            "ChannelWiseNorm": (lambda n, e: ChannelWiseNormalization(n, e)),
            "Conv1dLayerNorm": (lambda n, e: Conv1dLayerNorm(n, eps=e)),
            "Dropout1d": (lambda d: nn.Dropout1d(p=d)),
            "Maxpool1d": (lambda n, s, p, d: nn.MaxPool1d(n, stride=s, padding=p, dilation=d)),
            "Upsample1d": (lambda n: nn.Upsample(scale_factor=n, mode="nearest")),
            "Conv1d": (lambda i, o, k, s, p, d, b: nn.Conv1d(i, o, kernel_size=k, stride=s, padding=p, dilation=d, bias=b)),
            "DepthwiseSeparableConv1d": (lambda i, o, k, s, p, d, b: DepthwiseSeparableConv1d(i, o, kernel_size=k, stride=s, padding=p, dilation=d, bias=b)),
            "ConvBlock": (lambda i, o, k, d, af, c, s, p, di, b, n, e, activation, dropout, c1, c2, norm:
                          (lambda x: self._conv_block(activation, dropout, c1, c2, norm, x), [activation, dropout, c1, c2, norm])),
            "Encoder": (lambda i, o, k, d, af, c, s, p, di, b, e, n, m, ms, mpa, md, mp, cb: (lambda x: self._encoder(cb[0], mp, x), [mp] + cb[1])),
            "Decoder": (lambda i, o, k, d, af, c, s, p, di, b, e, n, m, mp, cb: (lambda x, y: self._decoder(cb[0], mp, x, y), [mp] + cb[1])),

            "UStructure": (lambda i, out_enc, in_dec, k1, k2, d, af, c, s, p, di, b, e, n, m, mst, mpa, md,
                              ds, ks, ms, enc, dec, cb: (lambda x: self._ustructure(enc[0], dec[0], cb[0], x), enc[1] + dec[1] + cb[1])),

            "UStructure_Cons": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, s, p, di, b, e, n, m, mst, mpa, md,
                                            dds, ds, kks, ks, mms, ms, enc, dec, u_model: (lambda x: self._ustructure_cons(enc[0], dec[0], u_model[0], x), enc[1] + dec[1] + u_model[1])),

            "LinearLayer": (lambda i, o, b: nn.Linear(i, o, bias=b)),

            "UClassifier": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, conv, c_stride, c_padding, c_dilation, b, e, norm, m, m_stride, m_padding, m_dilation,
                              first_d, first_af, first_conv, first_c_stride, first_c_padding, first_c_dilation, first_b, first_e, first_norm, first_m_stride, first_m_padding, first_m_dilation,
                              fc_k, fc_conv, fc_stride, fc_padding, fc_dilation, fc_b, mlp_in, mlp_out, mlp_b,
                              dds, ds, kks, ks, mms, ms, enc, dec, u, fc, mlp:
                            (lambda x: self._uclassifier(enc[0], dec[0], u[0], fc, mlp, x), [fc, mlp] + enc[1] + dec[1] + u[1])),

            # TODO: check type of loss functions and make it compatible with MultiClassTrainer   --  should be ok?!

            "BCEwithLogits": self._bce_with_logits,

            "CrossEntropy": lambda pred, target, additional=None: torch.nn.functional.cross_entropy(pred, target.argmax(1)),

            "MAE": self._mae,

            "MSE": self._mse,

            "ChannelSampler": lambda n, p: ChannelSampler(n),

            "Crop": lambda ti, sr, w, p_none: Crop(ti, sr, w),

            "EmpiricalClipScaler": lambda q, s, p_none: EmpiricalClipScaler(q, s),

            "FIR": lambda sr, ch, fp, zp, p_none: FIR(sr, ch, {k: v for (k, v) in fp}, zp),

            "Normalize": Normalize(),

            "RobustScaler": lambda lq, uq, p_none: RobustScaler(lq, uq),

            "Spectogram": lambda n_fft, hl, wl, els, p_none: Spectogram(n_fft, hl, wl, els),

            "ZNormalize": lambda ugs, p: ZNormalize(ugs),

            "Preprocessor_Sequence": lambda ps: (),

            "Preprocessor_Sequence_Cons": lambda p, pps, ps, x, xs: (x,) + xs,

            "UModel": (lambda bd, bk, d, af, conv, c_s, c_p, c_d, b, e, norm,
                               m_s, m_p, m_d, f_d, f_af, f_c, f_c_s, f_c_p, f_c_d, f_b, f_e, f_norm, f_m_s, f_m_p,
                               f_m_d, fc_k, fc_c, fc_s, fc_p, fc_d, fc_b, mlp_in, mlp_out, mlp_b, dds, kks, mms,
                               loss, preps, loss_f, preprocessors, u, classes, n_channels: (UTime(u[0], classes,
                                                                                                  n_channels,
                                                                                                  u[1],
                                                                                                  preprocessors=preprocessors),
                                                                                            loss_f)),

            "NoSampler": lambda r, n, d: None,

            "RandomSampler": lambda r, n, d: torch.utils.data.RandomSampler(d, replacement=r, num_samples=n),

            "ABC_Dataset": lambda a, c, nw, sf, em, of, ti, tr, patients: ABC(annotator=a,
                                                                              channels=[ChannelConfig(name=s) for s in c],
                                                                              patients=patients, num_workers=nw,
                                                                              sample_frequency=sf, event_mapping={k:v for (k,v) in em},
                                                                              online_filtering=of, total_input=ti,
                                                                              target_resolution=tr),

            "DataLoader": (lambda a, ch, nw, sf, em, of, ti, tr, r, ns, sam, bs, s, d, patients:
                           self._dataloader(nw, bs, s, d, patients)),

            "Adagrad": lambda lr, lr_d, w_d, i_a_v, eps, p_none, model: torch.optim.Adagrad(params=model.parameters(),
                                                                                             lr=lr,
                                                                                             lr_decay=lr_d,
                                                                                             weight_decay=w_d,
                                                                                             initial_accumulator_value=i_a_v,
                                                                                             eps=eps),

            "Adam": lambda lr, beta, eps, w_d, amsgrad, p_none, model: torch.optim.Adam(params=model.parameters(), lr=lr,
                                                                                         betas=beta, eps=eps,
                                                                                         weight_decay=w_d,
                                                                                         amsgrad=amsgrad),

            "AdamW": lambda lr, beta, eps, w_d, amsgrad, p_none, model: torch.optim.AdamW(params=model.parameters(), lr=lr,
                                                                                           betas=beta, eps=eps,
                                                                                           weight_decay=w_d,
                                                                                           amsgrad=amsgrad),

            "Adamax": lambda lr, beta, eps, w_d, p_none, model: torch.optim.Adamax(params=model.parameters(), lr=lr, betas=beta,
                                                                                    eps=eps, weight_decay=w_d),

            "SGD": lambda lr, m, d, w_d, n, p_none, model: torch.optim.SGD(params=model.parameters(), lr=lr, momentum=m,
                                                                            dampening=d, weight_decay=w_d, nesterov=n),

            "LinearLR": lambda sf, ef, ti, le, p_none, opti: torch.optim.lr_scheduler.LinearLR(optimizer=opti,
                                                                                               start_factor=sf,
                                                                                               end_factor=ef,
                                                                                               total_iters=ti,
                                                                                               last_epoch=le),

            "StepLR": lambda s, g, l, p_none, opti: torch.optim.lr_scheduler.StepLR(optimizer=opti, step_size=s, gamma=g, last_epoch=l),

            "ExponentialLR": lambda g, l, p_none, opti: torch.optim.lr_scheduler.ExponentialLR(optimizer=opti, gamma=g, last_epoch=l),

            "MulticlassTrainer": (lambda bd, bk, d, af, conv, c_s, c_p, c_d, b, e, norm, m_s, m_p, m_d, f_d, f_af, f_c,
                                         f_c_s, f_c_p, f_c_d, f_b, f_e, f_norm, f_m_s, f_m_p, f_m_d, fc_k, fc_c, fc_s,
                                         fc_p, fc_d, fc_b, mlp_in, mlp_out, mlp_b, dds, kks, mms, loss, preps_none, preps,
                                         annotator, channels, num_w, sample_f, event_map, online_f, total_in,
                                         target_res, replacement, num_s, sampler, batch_size, opti_none, opti,
                                         lr_sched_none, lr_sched,
                                         model, dataloader, optimizer, lr_scheduler: self._train_multiclass(model, dataloader, optimizer, lr_scheduler)),
        }

if __name__ == "__main__":
    repo = UtimeRepository(dimension_choices=[64, 128, 256], normalization_eps_choices=[1e-3], dropout_p_choices=[0.1],
                           convolution_kernel_size_choices=[5, 3, 2], convolution_stride_choices=[1, ],
                           convolution_padding_choices=[0, ], convolution_dilations_choices=[1, ],
                           maxpool_size_choices=[3, 5], maxpool_stride_choices=[1,], maxpool_padding_choices=[0, ],
                           maxpool_dilation_choices=[1, ],
                           preprocessor_channel_sampler_n_choices=[1],
                           preprocessor_crop_total_input_choices=[128], preprocessor_crop_sampling_rate_choices=[64],
                           preprocessor_empirical_clip_scaler_q_choices=[0.9], preprocessor_empirical_clip_scaler_scale_choices=[1],
                           preprocessor_fir_sampling_rate_choices=[64], preprocessor_fir_channels_choices=["channel"],
                           preprocessor_fir_filter_params_choices=[(("channel", "filter_param"),)],
                           preprocessor_robust_scaler_lower_quantile_choices=[0.25], preprocessor_robust_scaler_upper_quantile_choices=[0.75],
                           preprocessor_spectogram_n_fft_choices=[256], preprocessor_spectogram_hop_length_choices=[64],
                           preprocessor_spectogram_win_length_choices=[torch.hamming_window(256)], preprocessor_spectogram_epoch_len_samples_choices=[1],
                           n_samples=[10000],
                           abc_channel_choices=[("Sp02", "ECG1", "ECG2", "Thor")],
                           abc_event_mapping=[(("hypopnea|hypopnea", "hypopnea"), ("central apnea|central apnea", "apnea"), ("obstructive apnea|obstructive apnea", "apnea"),)],
                           abc_num_workers=[8], abc_sample_frequency=[10, 100],
                           abc_total_input=["30s"], abc_target_resolution=["1s"],
                           batch_size=[32], optimizer_learning_rate=[1e-3], optimizer_learning_rate_decay=[0], optimizer_weight_decay=[0, 1e-4],
                           optimizer_eps=[1e-10], optimizer_beta=[(0.9, 0.999)], optimizer_initial_accumulator_value=[0], optimizer_momentum=[0],
                           optimizer_dampening=[0], lr_scheduler_start_factor_choices=[1], lr_scheduler_end_factor_choices=[1e-2],
                           lr_scheduler_total_iters_choices=[50], lr_scheduler_step_size_choices=[30], lr_scheduler_gamma_choices=[0.1, 0.95],
                           lr_scheduler_last_epoch_choices=[-1])

    target0 = (
            Constructor("u_structure",
                        Constructor("dimensions", Literal((256, 128, 128)))
                        & Constructor("kernel_sizes", Literal((2, 3, 5)))
                        & Constructor("maxpool_sizes", Literal((5, 5, 3)))
                        )
            & Constructor("bottleneck",
                          Constructor("in_and_out", Literal(64))
                          & Constructor("kernel_size", Literal(1))
                          )
            & Constructor("homogeneous",
                          Constructor("convolution", Literal("simple_convolution"))
                          & Constructor("convolution_stride", Literal(1))
                          & Constructor("convolution_padding", Literal(0))
                          & Constructor("convolution_dilation", Literal(1))
                          & Constructor("bias", Literal(True))
                          & Constructor("activation", Literal("ReLu"))
                          & Constructor("dropout_p", Literal(0.1))
                          & Constructor("normalization", Literal("channel_wise_norm"))
                          & Constructor("normalization_epsilon", Literal(1e-3))
                          & Constructor("maxpool_stride", Literal(1))
                          & Constructor("maxpool_padding", Literal(0))
                          & Constructor("maxpool_dilation", Literal(1))
                          )
              )

    target1 = (
            Constructor("u_structure",
                        Constructor("dimensions", Literal((256, None, 128)))
                        & Constructor("kernel_sizes", Literal((2, 3, None)))
                        & Constructor("maxpool_sizes", Literal((None, 5, 3)))
                        )
            & Constructor("bottleneck",
                          Constructor("in_and_out", Literal(None))
                          & Constructor("kernel_size", Literal(None))
                          )
            & Constructor("homogeneous",
                          Constructor("convolution", Literal(None))
                          & Constructor("convolution_stride", Literal(None))
                          & Constructor("convolution_padding", Literal(None))
                          & Constructor("convolution_dilation", Literal(None))
                          & Constructor("bias", Literal(True))
                          & Constructor("activation", Literal(None))
                          & Constructor("dropout_p", Literal(0.1))
                          & Constructor("normalization", Literal("channel_wise_norm"))
                          & Constructor("normalization_epsilon", Literal(1e-3))
                          & Constructor("maxpool_stride", Literal(1))
                          & Constructor("maxpool_padding", Literal(0))
                          & Constructor("maxpool_dilation", Literal(1))
                          )
    )

    target2 = (Constructor("u_classifier",
                                Constructor("dimensions", Literal((256, 128, 128)))
                                & Constructor("kernel_sizes", Literal((2, 3, 5)))
                                & Constructor("maxpool_sizes", Literal((5, 5, 3)))
                                )
               & Constructor("u_first_level", Constructor("convolution", Literal("simple_convolution"))
                          & Constructor("convolution_stride", Literal(1))
                          & Constructor("convolution_padding", Literal(0))
                          & Constructor("convolution_dilation", Literal(1))
                          & Constructor("bias", Literal(True))
                          & Constructor("activation", Literal("ReLu"))
                          & Constructor("dropout_p", Literal(0.1))
                          & Constructor("normalization", Literal("channel_wise_norm"))
                          & Constructor("normalization_epsilon", Literal(1e-3))
                          & Constructor("maxpool_stride", Literal(1))
                          & Constructor("maxpool_padding", Literal(0))
                          & Constructor("maxpool_dilation", Literal(1))
                             )
               & Constructor("bottleneck",
                          Constructor("in_and_out", Literal(64))
                          & Constructor("kernel_size", Literal(1))
                          )
               & Constructor("homogeneous",
                          Constructor("convolution", Literal("simple_convolution"))
                          & Constructor("convolution_stride", Literal(1))
                          & Constructor("convolution_padding", Literal(0))
                          & Constructor("convolution_dilation", Literal(1))
                          & Constructor("bias", Literal(True))
                          & Constructor("activation", Literal("ReLu"))
                          & Constructor("dropout_p", Literal(0.1))
                          & Constructor("normalization", Literal("channel_wise_norm"))
                          & Constructor("normalization_epsilon", Literal(1e-3))
                          & Constructor("maxpool_stride", Literal(1))
                          & Constructor("maxpool_padding", Literal(0))
                          & Constructor("maxpool_dilation", Literal(1))
                          )
               & Constructor("u_final_conv",
                                  Constructor("kernel_size", Literal(1))
                                  & Constructor("convolution", Literal("simple_convolution"))
                                  & Constructor("convolution_stride", Literal(1))
                                  & Constructor("convolution_padding", Literal(0))
                                  & Constructor("convolution_dilation", Literal(1))
                                  & Constructor("bias", Literal(True))
                             )
               & Constructor("u_linear_classifier",
                                  Constructor("linear_layer",
                                              Constructor("input", Literal(256))
                                              & Constructor("output", Literal(64))
                                              & Constructor("bias", Literal(False))
                                              )
                                  )
               )

    target3 = (Constructor("u_classifier",
                           Constructor("dimensions", Literal((None, 128, 128)))
                           & Constructor("kernel_sizes", Literal((2, None, 5)))
                           & Constructor("maxpool_sizes", Literal((None, 5, None)))
                           )
               & Constructor("u_first_level", Constructor("convolution", Literal(None))
                             & Constructor("convolution_stride", Literal(None))
                             & Constructor("convolution_padding", Literal(None))
                             & Constructor("convolution_dilation", Literal(None))
                             & Constructor("bias", Literal(None))
                             & Constructor("activation", Literal(None))
                             & Constructor("dropout_p", Literal(None))
                             & Constructor("normalization", Literal(None))
                             & Constructor("normalization_epsilon", Literal(None))
                             & Constructor("maxpool_stride", Literal(None))
                             & Constructor("maxpool_padding", Literal(None))
                             & Constructor("maxpool_dilation", Literal(None))
                             )
               & Constructor("bottleneck",
                             Constructor("in_and_out", Literal(64))
                             & Constructor("kernel_size", Literal(1))
                             )
               & Constructor("homogeneous",
                             Constructor("convolution", Literal("simple_convolution"))
                             & Constructor("convolution_stride", Literal(1))
                             & Constructor("convolution_padding", Literal(0))
                             & Constructor("convolution_dilation", Literal(1))
                             & Constructor("bias", Literal(True))
                             & Constructor("activation", Literal("ReLu"))
                             & Constructor("dropout_p", Literal(0.1))
                             & Constructor("normalization", Literal("channel_wise_norm"))
                             & Constructor("normalization_epsilon", Literal(1e-3))
                             & Constructor("maxpool_stride", Literal(1))
                             & Constructor("maxpool_padding", Literal(0))
                             & Constructor("maxpool_dilation", Literal(1))
                             )
               & Constructor("u_final_conv",
                             Constructor("kernel_size", Literal(None))
                             & Constructor("convolution", Literal(None))
                             & Constructor("convolution_stride", Literal(None))
                             & Constructor("convolution_padding", Literal(None))
                             & Constructor("convolution_dilation", Literal(None))
                             & Constructor("bias", Literal(None))
                             )
               & Constructor("u_linear_classifier",
                             Constructor("linear_layer",
                                         Constructor("input", Literal(None))
                                         & Constructor("output", Literal(None))
                                         & Constructor("bias", Literal(None))
                                         )
                             )
               )

    target4 = Constructor("u_model",
                          target2
                          & Constructor("loss_function", Literal("MSE"))
                          & Constructor("preprocessors", Literal((
                                            ("ChannelSampler", 1),
                                            #("FIR", 64, "channel", (("channel", "filter_param"),), False),
                                            )))
                          )

    target5 = Constructor("u_model",
                          target2
                          & Constructor("loss_function", Literal("MSE"))
                          & Constructor("preprocessors", Literal((
                              ("ChannelSampler", 1),
                              ("FIR", 64, None, None, None),
                              None
                          )))
                          )

    target6 = Constructor("trainer",
                          target4
                          & Constructor("dataloader",
                                              Constructor("Sampler",
                                                          Literal("RandomSampler")
                                                          & Constructor("replacement", Literal(True))
                                                          & Constructor("num_samples", Literal(10000))
                                                          )
                                              & Constructor("Dataset",
                                                            Literal("ABC_Dataset")
                                                            & Constructor("annotator", Literal("nsrr"))
                                                            & Constructor("channels", Literal(("Sp02", "ECG1", "ECG2", "Thor")))
                                                            & Constructor("num_workers", Literal(8))
                                                            & Constructor("sample_frequency", Literal(100))
                                                            & Constructor("event_mapping", Literal((("hypopnea|hypopnea", "hypopnea"), ("central apnea|central apnea", "apnea"), ("obstructive apnea|obstructive apnea", "apnea"),)))
                                                            & Constructor("online_filtering", Literal(True))
                                                            & Constructor("total_input", Literal("30s"))
                                                            & Constructor("target_resolution", Literal("1s"))
                                                            )
                                              & Constructor("batch_size", Literal(32))
                                              )
                          & Constructor("optimizer", Literal(("Adam", 1e-3, (0.9, 0.999), 1e-10, 0, True)))
                          & Constructor("lr_scheduler", Literal(("LinearLR", 1, 1e-2, 50, -1)))
                          )

    target7 = Constructor("trainer",
                          target4
                          & Constructor("dataloader",
                                        Constructor("Sampler",
                                                    Literal("RandomSampler")
                                                    & Constructor("replacement", Literal(True))
                                                    & Constructor("num_samples", Literal(10000))
                                                    )
                                        & Constructor("Dataset",
                                                      Literal("ABC_Dataset")
                                                      & Constructor("annotator", Literal("nsrr"))
                                                      & Constructor("channels",
                                                                    Literal(("Sp02", "ECG1", "ECG2", "Thor")))
                                                      & Constructor("num_workers", Literal(8))
                                                      & Constructor("sample_frequency", Literal(100))
                                                      & Constructor("event_mapping", Literal(
                                                          (("hypopnea|hypopnea", "hypopnea"),
                                                           ("central apnea|central apnea", "apnea"),
                                                           ("obstructive apnea|obstructive apnea", "apnea"),)))
                                                      & Constructor("online_filtering", Literal(True))
                                                      & Constructor("total_input", Literal("30s"))
                                                      & Constructor("target_resolution", Literal("1s"))
                                                      )
                                        & Constructor("batch_size", Literal(32))
                                        )
                          & Constructor("optimizer", Literal(("Adam", 1e-3, (0.9, 0.999), 1e-10, 0, None)))
                          & Constructor("lr_scheduler", Literal(None))
                          )

    target = target6

    synthesizer = Synthesizer(repo.specification(), {})

    search_space = synthesizer.construct_solution_space(target).prune()

    trees = search_space.enumerate_trees(target, 10)

    for t in trees:
        #print(t)
        #print(t.interpret(repo.pretty_term_algebra()))
        t.interpret(repo.torch_algebra())


