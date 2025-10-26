from cosy.dsl import DSL
from cosy.types import Constructor, Group, DataGroup, Literal, Type, Var
from cosy.synthesizer import Synthesizer

#import torch
#import torch.nn as nn
#import torch.nn.functional as F

"""
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
"""
class UtimeRepository:
    def __init__(self, dimension_choices, normalization_eps_choices, #normalization_momentum_choices,
                 dropout_p_choices,
                 convolution_kernel_size_choices, convolution_stride_choices, convolution_padding_choices,
                 convolution_dilations_choices,
                 maxpool_size_choices, maxpool_stride_choices, maxpool_padding_choices, maxpool_dilation_choices,
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

        if 1 not in self.convolution_kernel_size_choices:
            self.convolution_kernel_size_choices.append(1)

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

        self.convs = ["simple_convolution", "depthwise_separable_convolution", None]

        self.afs = ["ReLu", "ELU", "Tanh", None]

        self.norms = ["batch_norm", "conv1d_layer_norm", "channel_wise_norm", None]

    class Maybe_Nat(Group):
        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, int) and value >= 0)

    class Nat(Group):
        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    class Maybe_Nat_Tuple(Group):
        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else isinstance(v, int) and v >= 0 for v in value)

    class Maybe_Conv_Tuple(Group):
        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.conv_choices for v in value)

    class Maybe_Conv_Tuple_Tuple(Group):
        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and
                    all(isinstance(v, tuple) and
                        all(True if c is None else c in self.conv_choices for c in v) for v in value))

    class Maybe_AF_Tuple(Group):
        def __init__(self, af_choices):
            self.af_choices = af_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.af_choices for v in value)

    class Maybe_Norm_Tuple(Group):
        def __init__(self, norm_choices):
            self.norm_choices = norm_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.norm_choices for v in value)

    class Maybe_Dropout_Tuple(Group):
        def __init__(self, dropout_p_choices):
            self.dropout_p_choices = dropout_p_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.dropout_p_choices for v in value)

    class Maybe_Kernel_Size_Tuple(Group):
        def __init__(self, kernel_size_choices):
            self.kernel_size_choices = kernel_size_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.kernel_size_choices for v in value)

    class Maybe_Maxpool_Size_Tuple(Group):
        def __init__(self, maxpool_size_choices):
            self.maxpool_size_choices = maxpool_size_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.maxpool_size_choices for v in value)

    class Maybe_Dimension_Tuple(Group):
        def __init__(self, dimension_choices):
            self.dimension_choices = dimension_choices

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and all(True if v is None else v in self.dimension_choices for v in value))


    def dimension(self):
        return DataGroup("dimension", self.dimension_choices)

    def normalization_eps(self):
        return DataGroup("normalization_eps", self.normalization_eps_choices)

    def dropout_p(self):
        return DataGroup("dropout_p", self.dropout_p_choices)

    def maxpool_size(self):
        return DataGroup("maxpool_size", self.maxpool_size_choices)

    def kernel_size(self):
        return DataGroup("kernel_size", self.convolution_kernel_size_choices)

    def activation_function(self):
        return DataGroup("activation_function", self.afs)

    def convolution(self):
        return DataGroup("convolution", self.convs)

    def normalization(self):
        return DataGroup("normalization", self.norms)

    def dimension_list(self):
        return DataGroup("dimension_list", self.Maybe_Dimension_Tuple(self.dimension_choices))

    def kernel_size_list(self):
        return DataGroup("kernel_size_list", self.Maybe_Kernel_Size_Tuple(self.convolution_kernel_size_choices))

    def maxpool_size_list(self):
        return DataGroup("maxpool_size_list", self.Maybe_Maxpool_Size_Tuple(self.maxpool_size_choices))

    def length(self):
        return DataGroup("length", self.Maybe_Nat())

    #def normalization_momentum(self):
    #    return DataGroup("normalization_momentum", self.normalization_momentum_choices)

    def maxpool_stride(self):
        return DataGroup("maxpool_stride", self.maxpool_stride_choices)

    def maxpool_padding(self):
        return DataGroup("maxpool_padding", self.maxpool_padding_choices)

    def maxpool_dilation(self):
        return DataGroup("maxpool_dilation", self.maxpool_dilation_choices)

    def convolution_stride(self):
        return DataGroup("convolution_stride", self.convolution_stride_choices)

    def convolution_padding(self):
        return DataGroup("convolution_padding", self.convolution_padding_choices)

    def convolution_dilation(self):
        return DataGroup("convolution_dilation", self.convolution_dilations_choices)

    def bias(self):
        return DataGroup("bias", [True, False])

    def specification(self):
        return {
            "ReLu": Constructor("activation_function") & Literal("ReLu") & Literal(None),

            "ELU": Constructor("activation_function") & Literal("ELU") & Literal(None),

            "Tanh": Constructor("activation_function") & Literal("Tanh") & Literal(None),

            "BatchNorm1d": DSL()
            .parameter("n", self.dimension())
            .parameter_constraint(lambda v: v["n"] is not None)
            .parameter("e", self.normalization_eps())
            .parameter_constraint(lambda v: v["e"] is not None)
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n"))
                                & Constructor("output", Literal(None))
                                )
                    & Constructor("normalization_epsilon", Var("e"))
                    & Constructor("normalization_epsilon", Literal(None))
                    & Literal("batch_norm")
                    ),

            "ChannelWiseNorm": DSL()
            .parameter("n", self.dimension())
            .parameter_constraint(lambda v: v["n"] is not None)
            .parameter("e", self.normalization_eps())
            .parameter_constraint(lambda v: v["e"] is not None)
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n"))
                                & Constructor("output", Literal(None))
                                )
                    & Constructor("normalization_epsilon", Var("e"))
                    & Constructor("normalization_epsilon", Literal(None))
                    & Literal("channel_wise_norm")
                    ),

            "Conv1dLayerNorm": DSL()
            .parameter("n", self.dimension())
            .parameter_constraint(lambda v: v["n"] is not None)
            .parameter("e", self.normalization_eps())
            .parameter_constraint(lambda v: v["e"] is not None)
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n"))
                                & Constructor("output", Literal(None))
                                )
                    & Constructor("normalization_epsilon", Var("e"))
                    & Constructor("normalization_epsilon", Literal(None))
                    & Constructor("conv1d_layer_norm")
                    ),

            "Dropout1d": DSL()
            .parameter("d", self.dropout_p())
            .parameter_constraint(lambda v: v["d"] is not None)
            .suffix(Constructor("dropout")
                    & Constructor("dropout_probability", Var("d"))
                    & Constructor("dropout_probability", Literal(None))
                    ),

            "Maxpool1d": DSL()
            .parameter("n", self.maxpool_size())
            .parameter_constraint(lambda v: v["n"] is not None)
            .parameter("s", self.maxpool_stride())
            .parameter_constraint(lambda v: v["s"] is not None)
            .parameter("p", self.maxpool_padding())
            .parameter_constraint(lambda v: v["p"] is not None)
            .parameter("d", self.maxpool_dilation())
            .parameter_constraint(lambda v: v["d"] is not None)
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
            .parameter("n", self.maxpool_size())
            .parameter_constraint(lambda v: v["n"] is not None)
            .suffix(Constructor("upsample1d",
                                Constructor("scale_factor", Var("n"))
                                & Constructor("scale_factor", Literal(None))
                                )
                    ),

            "Conv1d": DSL()
            .parameter("in", self.dimension())
            .parameter_constraint(lambda v: v["in"] is not None)
            .parameter("out", self.dimension())
            .parameter_constraint(lambda v: v["out"] is not None)
            .parameter("k", self.kernel_size())
            .parameter_constraint(lambda v: v["k"] is not None)
            .parameter("s", self.convolution_stride())
            .parameter_constraint(lambda v: v["s"] is not None)
            .parameter("p", self.convolution_padding())
            .parameter_constraint(lambda v: v["p"] is not None)
            .parameter("d", self.convolution_dilation())
            .parameter_constraint(lambda v: v["d"] is not None)
            .parameter("b", self.bias())
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
            .parameter("in", self.dimension())
            .parameter_constraint(lambda v: v["in"] is not None)
            .parameter("out", self.dimension())
            .parameter_constraint(lambda v: v["out"] is not None)
            .parameter("k", self.kernel_size())
            .parameter_constraint(lambda v: v["k"] is not None)
            .parameter("s", self.convolution_stride())
            .parameter_constraint(lambda v: v["s"] is not None)
            .parameter("p", self.convolution_padding())
            .parameter_constraint(lambda v: v["p"] is not None)
            .parameter("d", self.convolution_dilation())
            .parameter_constraint(lambda v: v["d"] is not None)
            .parameter("b", self.bias())
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
            .parameter("in", self.dimension())
            .parameter("out", self.dimension())
            .parameter("k", self.kernel_size())
            .parameter("d", self.dropout_p())
            .parameter("af", self.activation_function())
            .parameter("conv", self.convolution())
            .parameter("stride", self.convolution_stride())
            .parameter("padding", self.convolution_padding())
            .parameter("dilation", self.convolution_dilation())
            .parameter("b", self.bias())
            .parameter("norm", self.normalization())
            .parameter("norm_e", self.normalization_eps())
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
            .parameter("in", self.dimension())
            .parameter("out", self.dimension())
            .parameter("k", self.kernel_size())
            .parameter("d", self.dropout_p())
            .parameter("af", self.activation_function())
            .parameter("conv", self.convolution())
            .parameter("c_stride", self.convolution_stride())
            .parameter("c_padding", self.convolution_padding())
            .parameter("c_dilation", self.convolution_dilation())
            .parameter("b", self.bias())
            .parameter("e", self.normalization_eps())
            .parameter("norm", self.normalization())
            .parameter("m", self.maxpool_size())
            .parameter("m_stride", self.convolution_stride())
            .parameter("m_padding", self.convolution_padding())
            .parameter("m_dilation", self.convolution_dilation())
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
            .parameter("in", self.dimension())
            .parameter("out", self.dimension())
            .parameter("k", self.kernel_size())
            .parameter("d", self.dropout_p())
            .parameter("af", self.activation_function())
            .parameter("conv", self.convolution())
            .parameter("c_stride", self.convolution_stride())
            .parameter("c_padding", self.convolution_padding())
            .parameter("c_dilation", self.convolution_dilation())
            .parameter("b", self.bias())
            .parameter("e", self.normalization_eps())
            .parameter("norm", self.normalization())
            .parameter("sf", self.maxpool_size())
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

            "UModel": DSL()
            .parameter("in", self.dimension(), lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("out_enc", self.dimension(), lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("in_dec", self.dimension(), lambda v: [2 * v["out_enc"]])
            .parameter("k1", self.kernel_size(), lambda v: [x for x in self.convolution_kernel_size_choices if x is not None])
            .parameter("k2", self.kernel_size(), lambda v: [x for x in self.convolution_kernel_size_choices if x is not None])
            .parameter("d", self.dropout_p(), lambda v: [x for x in self.dropout_p_choices if x is not None])
            .parameter("af", self.activation_function(), lambda v: [x for x in self.afs if x is not None])
            .parameter("conv", self.convolution(), lambda v: [x for x in self.convs if x is not None])
            .parameter("c_stride", self.convolution_stride())
            .parameter_constraint(lambda v: v["c_stride"] is not None)
            .parameter("c_padding", self.convolution_padding())
            .parameter_constraint(lambda v: v["c_padding"] is not None)
            .parameter("c_dilation", self.convolution_dilation())
            .parameter_constraint(lambda v: v["c_dilation"] is not None)
            .parameter("b", self.bias())
            .parameter("e", self.normalization_eps(), lambda v: [x for x in self.normalization_eps_choices if x is not None])
            .parameter("norm", self.normalization(), lambda v: [x for x in self.norms if x is not None])
            .parameter("m", self.maxpool_size(), lambda v: [x for x in self.maxpool_size_choices if x is not None])
            .parameter("m_stride", self.maxpool_stride())
            .parameter_constraint(lambda v: v["m_stride"] is not None)
            .parameter("m_padding", self.maxpool_padding())
            .parameter_constraint(lambda v: v["m_padding"] is not None)
            .parameter("m_dilation", self.maxpool_dilation())
            .parameter_constraint(lambda v: v["m_dilation"] is not None)
            .parameter("ds", self.dimension_list(), lambda v: [(v["in"],), (None,)])
            .parameter("ks", self.kernel_size_list(), lambda v: [(v["k1"],), (None,)])
            .parameter("ms", self.maxpool_size_list(), lambda v: [(v["m"],), (None,)])
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
            .suffix(Constructor("u_model",
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

            "UModel_Cons": DSL()
            .parameter("in_u", self.dimension(), lambda v: [x for x in self.dimension_choices if x is not None])  # in_u == out_enc
            .parameter("in_enc", self.dimension(), lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("in_dec", self.dimension(), lambda v: [2 * v["in_u"]])
            .parameter("bd", self.dimension())
            .parameter("k", self.kernel_size(), lambda v: [x for x in self.convolution_kernel_size_choices if x is not None])
            .parameter("bk", self.kernel_size())
            .parameter("d", self.dropout_p(), lambda v: [x for x in self.dropout_p_choices if x is not None])
            .parameter("af", self.activation_function(), lambda v: [x for x in self.afs if x is not None])
            .parameter("conv", self.convolution(), lambda v: [x for x in self.convs if x is not None])
            .parameter("c_stride", self.convolution_stride())
            .parameter_constraint(lambda v: v["c_stride"] is not None)
            .parameter("c_padding", self.convolution_padding())
            .parameter_constraint(lambda v: v["c_padding"] is not None)
            .parameter("c_dilation", self.convolution_dilation())
            .parameter_constraint(lambda v: v["c_dilation"] is not None)
            .parameter("b", self.bias())
            .parameter("e", self.normalization_eps(), lambda v: [x for x in self.normalization_eps_choices if x is not None])
            .parameter("norm", self.normalization(), lambda v: [x for x in self.norms if x is not None])
            .parameter("m", self.maxpool_size(), lambda v: [x for x in self.maxpool_size_choices if x is not None])
            .parameter("m_stride", self.maxpool_stride())
            .parameter_constraint(lambda v: v["m_stride"] is not None)
            .parameter("m_padding", self.maxpool_padding())
            .parameter_constraint(lambda v: v["m_padding"] is not None)
            .parameter("m_dilation", self.maxpool_dilation())
            .parameter_constraint(lambda v: v["m_dilation"] is not None)
            .parameter("dds", self.dimension_list())
            .parameter_constraint(lambda v: len(v["dds"]) > 1 and (v["dds"][0] == v["in_enc"] or v["dds"][0] is None))
            .parameter("ds", self.dimension_list(), lambda v: [v["dds"][1:]])
            .parameter_constraint(lambda v: len(v["ds"]) > 0 and (v["ds"][0] == v["in_u"] or v["ds"][0] is None))
            .parameter("kks", self.kernel_size_list())
            .parameter_constraint(lambda v: len(v["kks"]) > 1 and (v["kks"][0] == v["k"] or v["kks"][0] is None))
            .parameter("ks", self.kernel_size_list(), lambda v: [v["kks"][1:]])
            .parameter_constraint(lambda v: len(v["ks"]) > 0)
            .parameter("mms", self.maxpool_size_list())
            .parameter_constraint(lambda v: len(v["mms"]) > 1 and (v["mms"][0] == v["m"] or v["mms"][0] is None))
            .parameter("ms", self.maxpool_size_list(), lambda v: [v["mms"][1:]])
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
            .argument("u_model", Constructor("u_model",
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
            .suffix(Constructor("u_model",
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
            "UModel": (lambda i, out_enc, in_dec, k1, k2, d, af, c, s, p, di, b, e, n, m, mst, mpa, md,
                              ds, ks, ms, enc, dec, cb: f"U_Model({enc}, {dec}, {cb})"),

            "UModel_Cons": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, s, p, di, b, e, n, m, mst, mpa, md,
                                            dds, ds, kks, ks, mms, ms, enc, dec, u_model:
                            f"U_Model_Cons({enc}, {dec}, {u_model})"),
        }


    # TODO: Refactor from here
"""
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
"""
if __name__ == "__main__":
    repo = UtimeRepository(dimension_choices=[64, 128, 256], normalization_eps_choices=[1e-3], dropout_p_choices=[0.1],
                           convolution_kernel_size_choices=[5, 3, 2], convolution_stride_choices=[1, ],
                           convolution_padding_choices=[0, ], convolution_dilations_choices=[1, ],
                           maxpool_size_choices=[3, 5], maxpool_stride_choices=[1,], maxpool_padding_choices=[0, ],
                           maxpool_dilation_choices=[1, ])

    target0 = (
            Constructor("u_model",
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
            Constructor("u_model",
                        Constructor("dimensions", Literal((256, None, 128)))
                        & Constructor("kernel_sizes", Literal((2, 3, None)))
                        & Constructor("maxpool_sizes", Literal((None, 5, 3)))
                        )
            & Constructor("bottleneck",
                          Constructor("in_and_out", Literal(None))
                          & Constructor("kernel_size", Literal(1))
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


    target = target1

    synthesizer = Synthesizer(repo.specification(), {})

    search_space = synthesizer.construct_solution_space(target).prune()

    trees = search_space.enumerate_trees(target, 10)

    for t in trees:
        print(t.interpret(repo.pretty_term_algebra()))


