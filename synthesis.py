from cosy.dsl import DSL
from cosy.types import Constructor, Group, DataGroup, Literal, Type, Var
from cosy.synthesizer import Synthesizer

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 optimizer_beta, optimizer_initial_accumulator_value, optimizer_momentum, optimizer_dampening
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

    class Maybe_Preprocessor_Tuple(Group):
        name = "Maybe_Preprocessor_Tuple"

        def __init__(self, preprocessors):
            self.preprocessors = preprocessors

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and all(True if v is None else v in self.preprocessors for v in value))


    class Optimizer(Group):
            name = "Optimizer"

            """
            ("Adagrad", learning_rate, learning_rate_decay, weight_decay, initial_accumulator_value, eps)
            ("Adam", learning_rate, betas, eps, weight_decay, amsgrad = [True, False])
            ("AdamW", learning_rate, betas, eps, weight_decay, amsgrad = [True, False])
            ("Adamax", learning_rate, betas, eps, weight_decay)
            ("SGD", learning_rate, momentum, dampening, weight_decay, nesterov = [True, False])
            """

            def __init__(self, learning_rate, learning_rate_decay, weight_decay, eps, beta, initial_accumulator_value,
                         momentum, dampening):
                self.learning_rate = learning_rate
                self.learning_rate_decay = learning_rate_decay
                self.weight_decay = weight_decay
                self.eps = eps
                self.beta = beta
                self.initial_accumulator_value = initial_accumulator_value
                self.momentum = momentum
                self.dampening = dampening

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
                                for amsgrad in [True, False]:
                                    yield ("Adam", lr, b, e, wd, amsgrad)

            def iter_adamw(self):
                for lr in self.learning_rate:
                    for b in self.beta:
                        for e in self.eps:
                            for wd in self.weight_decay:
                                for amsgrad in [True, False]:
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
                                for nesterov in [True, False]:
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
                                value[5] in [True, False])
                    elif value[0] == "AdamW":
                        return (len(value) == 6 and
                                value[1] in self.learning_rate and
                                value[2] in self.beta and
                                value[3] in self.eps and
                                value[4] in self.weight_decay and
                                value[5] in [True, False])
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
                                value[5] in [True, False])
                    else:
                        return False
                else:
                    return False

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
        maybe_preprocessor = self.Preprocessor(self.preprocessor_channel_sampler_n_choices + [None],
                                               self.preprocessor_crop_total_input_choices + [None],
                                               self.preprocessor_crop_sampling_rate_choices + [None],
                                               self.preprocessor_empirical_clip_scaler_q_choices + [None],
                                               self.preprocessor_empirical_clip_scaler_scale_choices + [None],
                                               self.preprocessor_fir_sampling_rate_choices + [None],
                                               self.preprocessor_fir_channels_choices + [None],
                                               self.preprocessor_fir_filter_params_choices + [None],
                                               self.preprocessor_robust_scaler_lower_quantile_choices + [None],
                                               self.preprocessor_robust_scaler_upper_quantile_choices + [None],
                                               self.preprocessor_spectogram_n_fft_choices + [None],
                                               self.preprocessor_spectogram_hop_length_choices + [None],
                                               self.preprocessor_spectogram_win_length_choices + [None],
                                               self.preprocessor_spectogram_epoch_len_samples_choices + [None]
                                               )
        preprocessor_tuple = self.Maybe_Preprocessor_Tuple(maybe_preprocessor)
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
        optimizer = self.Optimizer(self.optimizer_learning_rate,
                                   self.optimizer_learning_rate_decay,
                                   self.optimizer_weight_decay,
                                   self.optimizer_eps,
                                   self.optimizer_beta,
                                   self.optimizer_initial_accumulator_value,
                                   self.optimizer_momentum,
                                   self.optimizer_dampening)
        maybe_optimizer = self.Optimizer(self.optimizer_learning_rate + [None],
                                         self.optimizer_learning_rate_decay + [None],
                                         self.optimizer_weight_decay + [None],
                                         self.optimizer_eps + [None],
                                         self.optimizer_beta + [None],
                                         self.optimizer_initial_accumulator_value + [None],
                                         self.optimizer_momentum + [None],
                                         self.optimizer_dampening + [None])

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
            #.parameter("p", preprocessor, lambda v: [("Crop", v["total_input"], v["sampling_rate"], v["where"])])
            .parameter("p_none", maybe_preprocessor)
            .parameter_constraint(lambda v: v["p_none"][0] == "Crop"
                                            and (v["p_none"][1] == v["total_input"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["sampling_rate"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["where"] or v["p_none"][3] is None))
            #.suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p_none"))),

            "EmpiricalClipScaler": DSL()
            .parameter("q", DataGroup("EmpiricalClipScaler_q", self.preprocessor_empirical_clip_scaler_q_choices))
            .parameter("scale", DataGroup("EmpiricalClipScaler_scale", self.preprocessor_empirical_clip_scaler_scale_choices))
            #.parameter("p", preprocessor, lambda v: [("EmpiricalClipScaler", v["q"], v["scale"])])
            .parameter("p_none", maybe_preprocessor)
            .parameter_constraint(lambda v: v["p_none"][0] == "EmpiricalClipScaler"
                                            and (v["p_none"][1] == v["q"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["scale"] or v["p_none"][2] is None))
            #.suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p_none"))),

            "FIR": DSL()
            .parameter("sampling_rate", DataGroup("FIR_sampling_rate", self.preprocessor_fir_sampling_rate_choices))
            .parameter("channels", DataGroup("FIR_channels", self.preprocessor_fir_channels_choices))
            .parameter("filter_params", DataGroup("FIR_filter_params", self.preprocessor_fir_filter_params_choices))
            .parameter("zero_phase", DataGroup("FIR_zero_phase", [True, False]))
            #.parameter("p", preprocessor, lambda v: [("FIR", v["sampling_rate"], v["channels"], v["filter_params"], v["zero_phase"])])
            .parameter("p_none", maybe_preprocessor)
            .parameter_constraint(lambda v: v["p_none"][0] == "FIR"
                                            and (v["p_none"][1] == v["sampling_rate"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["channels"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] == v["filter_params"] or v["p_none"][3] is None)
                                            and (v["p_none"][4] == v["zero_phase"] or v["p_none"][4] is None))
            #.suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p_none"))),

            "Normalize": DSL()
            #.parameter("p", preprocessor, lambda v: [("Normalize",)])
            .suffix(Constructor("preprocessor", Literal(("Normalize",)))),

            "RobustScaler": DSL()
            .parameter("lower_quantile", DataGroup("RobustScaler_lower_quantile", self.preprocessor_robust_scaler_lower_quantile_choices))
            .parameter("upper_quantile", DataGroup("RobustScaler_upper_quantile", self.preprocessor_robust_scaler_upper_quantile_choices))
            #.parameter("p", preprocessor, lambda v: [("RobustScaler", v["lower_quantile"], v["upper_quantile"])])
            .parameter("p_none", maybe_preprocessor)
            .parameter_constraint(lambda v: v["p_none"][0] == "RobustScaler"
                                            and (v["p_none"][1] == v["lower_quantile"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["upper_quantile"] or v["p_none"][2] is None))
            # .suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p_none"))),

            "Spectogram": DSL()
            .parameter("n_fft", DataGroup("Spectogram_n_fft", self.preprocessor_spectogram_n_fft_choices))
            .parameter("hop_length", DataGroup("Spectogram_hop_length", self.preprocessor_spectogram_hop_length_choices))
            .parameter("win_length", DataGroup("Spectogram_win_length", self.preprocessor_spectogram_win_length_choices))
            .parameter("epoch_len_samples", DataGroup("Spectogram_epoch_len_samples", self.preprocessor_spectogram_epoch_len_samples_choices))
            #.parameter("p", preprocessor,
            #           lambda v: [("Spectogram", v["n_fft"], v["hop_length"], v["win_length"], v["epoch_len_samples"])])
            .parameter("p_none", maybe_preprocessor)
            .parameter_constraint(lambda v: v["p_none"][0] == "Spectogram"
                                            and (v["p_none"][1] == v["n_fft"] or v["p_none"][1] is None)
                                            and (v["p_none"][2] == v["hop_length"] or v["p_none"][2] is None)
                                            and (v["p_none"][3] is None or torch.equal(v["p_none"][3], v["win_length"]))
                                            and (v["p_none"][4] == v["epoch_len_samples"] or v["p_none"][4] is None))
            # .suffix(Constructor("preprocessor", Var("p")) & Constructor("preprocessor", Var("p_none"))),
            .suffix(Constructor("preprocessor", Var("p_none"))),

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
            .parameter("p", maybe_preprocessor)
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

            # TODO: make stuff below noneable

            "NoSampler": Constructor("Sampler", Literal("NoSampler")),

            "RandomSampler": DSL()
            .parameter("replacement", DataGroup("Sampler_Replacement", [True, False]))
            .parameter("num_samples", n_samples)
            .suffix(Constructor("Sampler", Literal("RandomSampler")
                                & Constructor("replacement", Var("replacement"))
                                & Constructor("num_samples", Var("num_samples"))
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

            """
                        ("Adagrad", learning_rate, learning_rate_decay, weight_decay, initial_accumulator_value, eps)
                        ("Adam", learning_rate, beta, eps, weight_decay, amsgrad = [True, False])
                        ("AdamW", learning_rate, beta, eps, weight_decay, amsgrad = [True, False])
                        ("Adamax", learning_rate, beta, eps, weight_decay)
                        ("SGD", learning_rate, momentum, dampening, weight_decay, nesterov = [True, False])
                        """ :(),

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

        }


    # TODO: Refactor from here

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
    def _umodel_length(enc, dec, cb, x, return_intermediate=False):
        x, y = enc(x)
        z = cb(x)
        x = dec(z, y)
        if return_intermediate:
            return x, z.flatten(start_dim=1)
        else:
            return x

    @staticmethod
    def _umodel_cons_length(enc, dec, u_model, x, return_intermediate=False):
        x, y = enc(x)
        if return_intermediate:
            z, u_intermediate = u_model(x, return_intermediate=return_intermediate)
            x = dec(z, y)
            return x, u_intermediate
        else:
            z = u_model(x, return_intermediate=return_intermediate)
            x = dec(z, y)
            return x

    def torch_algebra(self):
        return {
            "ReLu": nn.ReLU(),
            "ELU": nn.ELU(),
            "Tanh": nn.Tanh(),
            "BatchNorm1d": (lambda n: nn.BatchNorm1d(n)),
            "ChannelWiseNorm": (lambda n, e: ChannelWiseNormalization(n, e)),
            "Conv1dLayerNorm": (lambda n: Conv1dLayerNorm(n)),
            "Dropout1d": (lambda d: nn.Dropout1d(p=d)),
            "Maxpool1d": (lambda n: nn.MaxPool1d(n)),
            "Upsample1d": (lambda n: nn.Upsample(scale_factor=n, mode="nearest")),
            "Conv1d": (lambda i, o, k: nn.Conv1d(i, o, kernel_size=k, padding=k // 2)),
            "DepthwiseSeparableConv1d": (lambda i, o, k: DepthwiseSeparableConv1d(i, o, kernel_size=k, padding=k // 2)),
            "ConvBlock": (lambda i, o, k, d, af, c, e, activation, dropout, c1, c2, norm, x:
                          self._conv_block(activation, dropout, c1, c2, norm, x)),
            "Encoder": (lambda i, o, k, d, af, c, e, n, m, mp, cb, x: self._encoder(cb, mp, x)),
            "Decoder": (lambda i, o, k, d, af, c, e, n, m, mp, cb, x, y: self._decoder(cb, mp, x, y)),
            "Linear": (lambda i, o: nn.Linear(i, o)),
            "UModel": (lambda i, out_enc, in_dec, k1, k2, d, af, c, e, n, m, ds, ks, ms, enc, dec, cb, x:
                       self._umodel_length(enc, dec, cb, x)),

            "UModel_Cons": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, e, n, m,
                                   dds, ds, kks, ks, mms, ms, enc, dec, u_model, x:
                            self._umodel_cons_length(enc, dec, u_model, x)),
            "UModel_length": (lambda i, out_enc, in_dec, k1, k2, d, af, c, e, n, m, enc, dec, cb, x:
                              self._umodel_length(enc, dec, cb, x)),
            "UModel_Cons_length": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, e, n, m, l, l_u, enc, dec, u_model, x:
                                   self._umodel_cons_length(enc, dec, u_model, x)),
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
                           preprocessor_spectogram_win_length_choices=[torch.hamming_window(256)], preprocessor_spectogram_epoch_len_samples_choices=[1])

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
                                            ("ChannelSampler", None),
                                            ("FIR", 64, None, None, False),
                                            None
                                            )))
                          )

    target = target4

    synthesizer = Synthesizer(repo.specification(), {})

    search_space = synthesizer.construct_solution_space(target).prune()

    trees = search_space.enumerate_trees(target, 10)

    for t in trees:
        #print(t)
        print(t.interpret(repo.pretty_term_algebra()))


