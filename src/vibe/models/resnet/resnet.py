import math
from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

INPLACE = True


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """helper function for constructing 3x3 grouped convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """helper function for constructing 1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def is_pos_int(number: int) -> bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


def is_pos_int_tuple(t: Tuple) -> bool:
    """
    Returns True if a tuple contains positive integers
    """
    return type(t) == tuple and all(is_pos_int(n) for n in t)


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int

    Returns
    -------
    padding : int
        The size of the padding to be added
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


class _Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool
        Whether to add a bias term to convolution operation.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network
    default_padding: str or int
        This sets the default padding mode that will be used by the pytorch Conv1d backend.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=False,
        weight_norm=False,
        conv_init=None,
        default_padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=default_padding,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal":
            nn.init.normal_(self.conv.weight, std=1e-6)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        wx : torch.Tensor
            The convolved outputs.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.

        Returns
        -------
        x : torch.Tensor
            The padded outputs.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


class _BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.
    skip_transpose : bool
        Whether to skip the transposition.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.

        Returns
        -------
        x_n : torch.Tensor
            The normalized outputs.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    """1D convolution. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    """1D batch normalization. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the TDNN block.
    activation : torch class
        A class for constructing the activation layers.
    groups : int
        The groups size of the TDNN blocks.
    dropout : float
        Rate of channel dropout during training.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
        dropout=0.0,
    ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        return self.dropout(self.norm(self.activation(self.conv(x))))


class SqueezeAndExcitationLayer(nn.Module):
    """Squeeze and excitation layer, as per https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(
        self,
        in_planes,
        reduction_ratio: Optional[int] = 16,
        reduced_planes: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Either reduction_ratio is defined, or out_planes is defined,
        # neither both nor none of them
        assert bool(reduction_ratio) != bool(reduced_planes)

        if activation is None:
            activation = nn.ReLU()

        reduced_planes = (
            in_planes // reduction_ratio if reduced_planes is None else reduced_planes
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=True),
            activation,
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled


class GenericLayer(nn.Module):
    """
    Parent class for 2-layer (BasicLayer) and 3-layer (BottleneckLayer)
    bottleneck layer class
    """

    def __init__(
        self,
        convolutional_block,
        in_planes,
        out_planes,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=4,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):

        # assertions on inputs:
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(
            reduction
        )

        # set object fields:
        super(GenericLayer, self).__init__()
        self.convolutional_block = convolutional_block
        self.final_bn_relu = final_bn_relu

        # final batchnorm and relu layer:
        if final_bn_relu:
            self.bn = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=INPLACE)

        # define down-sampling layer (if direct residual impossible):
        self.downsample = None
        if (stride != 1 and stride != (1, 1)) or in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=stride),
                nn.BatchNorm2d(out_planes),
            )

        self.se = (
            SqueezeAndExcitationLayer(out_planes, reduction_ratio=se_reduction_ratio)
            if use_se
            else None
        )

    def forward(self, x):

        # if required, perform downsampling along shortcut connection:
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)

        # forward pass through convolutional block:
        out = self.convolutional_block(x)

        if self.final_bn_relu:
            out = self.bn(out)

        if self.se is not None:
            out = self.se(out)

        # add residual connection, perform rely + batchnorm, and return result:
        out += residual
        if self.final_bn_relu:
            out = self.relu(out)
        return out


class BasicLayer(GenericLayer):
    """
    ResNeXt layer with `in_planes` input planes and `out_planes`
    output planes.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=1,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):

        # assertions on inputs:
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(
            reduction
        )

        # define convolutional block:
        convolutional_block = nn.Sequential(
            conv3x3(in_planes, out_planes, stride=stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=INPLACE),
            conv3x3(out_planes, out_planes),
        )

        # call constructor of generic layer:
        super().__init__(
            convolutional_block,
            in_planes,
            out_planes,
            stride=stride,
            reduction=reduction,
            final_bn_relu=final_bn_relu,
            use_se=use_se,
            se_reduction_ratio=se_reduction_ratio,
        )


class BottleneckLayer(GenericLayer):
    """
    ResNeXt bottleneck layer with `in_planes` input planes, `out_planes`
    output planes, and a bottleneck `reduction`.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=4,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):

        # assertions on inputs:
        assert is_pos_int(in_planes) and is_pos_int(out_planes)
        assert (is_pos_int(stride) or is_pos_int_tuple(stride)) and is_pos_int(
            reduction
        )

        # define convolutional layers:
        bottleneck_planes = int(math.ceil(out_planes / reduction))
        cardinality = 1
        if mid_planes_and_cardinality is not None:
            mid_planes, cardinality = mid_planes_and_cardinality
            bottleneck_planes = mid_planes * cardinality

        convolutional_block = nn.Sequential(
            conv1x1(in_planes, bottleneck_planes),
            nn.BatchNorm2d(bottleneck_planes),
            nn.ReLU(inplace=INPLACE),
            conv3x3(
                bottleneck_planes, bottleneck_planes, stride=stride, groups=cardinality
            ),
            nn.BatchNorm2d(bottleneck_planes),
            nn.ReLU(inplace=INPLACE),
            conv1x1(bottleneck_planes, out_planes),
        )

        # call constructor of generic layer:
        super(BottleneckLayer, self).__init__(
            convolutional_block,
            in_planes,
            out_planes,
            stride=stride,
            reduction=reduction,
            final_bn_relu=final_bn_relu,
            use_se=use_se,
            se_reduction_ratio=se_reduction_ratio,
        )


class SmallInputInitialBlock(nn.Module):
    """
    ResNeXt initial block for small input with `in_planes` input planes
    """

    def __init__(self, init_planes):
        super().__init__()
        self._module = nn.Sequential(
            conv3x3(3, init_planes, stride=1),
            nn.BatchNorm2d(init_planes),
            nn.ReLU(inplace=INPLACE),
        )

    def forward(self, x):
        return self._module(x)


class InitialBlock(nn.Module):
    """
    ResNeXt initial block with `in_planes` input planes
    """

    def __init__(self, init_planes):
        super().__init__()
        self._module = nn.Sequential(
            nn.Conv2d(3, init_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_planes),
            nn.ReLU(inplace=INPLACE),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self._module(x)


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    global_context: bool
        Whether to use global context.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        lengths : torch.Tensor
            The corresponding relative lengths of the inputs.

        Returns
        -------
        pooled_stats : torch.Tensor
            mean and std of batch
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class ResNeXt(nn.Module):

    def __init__(
        self,
        num_blocks,
        embedding_size: int = 256, 
        init_planes: int = 64,
        reduction: int = 4,
        small_input: bool = False,
        zero_init_bn_residuals: bool = False,
        base_width_and_cardinality: Optional[Sequence] = None,
        basic_layer: bool = False,
        final_bn_relu: bool = True,
        use_se: bool = False,
        se_reduction_ratio: int = 16,
    ):
        """
        Implementation of `ResNeXt <https://arxiv.org/pdf/1611.05431.pdf>`_.

        Args:
            small_input: set to `True` for 32x32 sized image inputs.
            final_bn_relu: set to `False` to exclude the final batchnorm and
                ReLU layers. These settings are useful when training Siamese
                networks.
            use_se: Enable squeeze and excitation
            se_reduction_ratio: The reduction ratio to apply in the excitation
                stage. Only used if `use_se` is `True`.
        """
        super().__init__()

        # assertions on inputs:
        assert isinstance(num_blocks, Sequence)
        assert all(is_pos_int(n) for n in num_blocks)
        assert is_pos_int(init_planes) and is_pos_int(reduction)
        assert type(small_input) == bool
        assert (
            type(zero_init_bn_residuals) == bool
        ), "zero_init_bn_residuals must be a boolean, set to true if gamma of last\
             BN of residual block should be initialized to 0.0, false for 1.0"
        assert base_width_and_cardinality is None or (
            isinstance(base_width_and_cardinality, Sequence)
            and len(base_width_and_cardinality) == 2
            and is_pos_int(base_width_and_cardinality[0])
            and is_pos_int(base_width_and_cardinality[1])
        )
        assert isinstance(use_se, bool), "use_se has to be a boolean"

        # initial convolutional block:
        self.num_blocks = num_blocks
        self.small_input = small_input
        self._make_initial_block(small_input, init_planes, basic_layer)

        # compute number of planes at each spatial resolution:
        out_planes = [init_planes * 2**i * reduction for i in range(len(num_blocks))]
        in_planes = [init_planes] + out_planes[:-1]

        # create subnetworks for each spatial resolution:
        blocks = []
        for idx in range(len(out_planes)):
            mid_planes_and_cardinality = None
            if base_width_and_cardinality is not None:
                w, c = base_width_and_cardinality
                mid_planes_and_cardinality = (w * 2**idx, c)
            new_block = self._make_resolution_block(
                in_planes[idx],
                out_planes[idx],
                idx,
                num_blocks[idx],  # num layers
                stride=1 if idx == 0 else 2,
                mid_planes_and_cardinality=mid_planes_and_cardinality,
                reduction=reduction,
                final_bn_relu=final_bn_relu or (idx != (len(out_planes) - 1)),
                use_se=use_se,
                se_reduction_ratio=se_reduction_ratio,
            )
            blocks.append(new_block)
        self.blocks = nn.Sequential(*blocks)

        self.out_planes = out_planes[-1]

        self.asp = AttentiveStatisticsPooling(channels=out_planes[-1])
        self.asp_bn = BatchNorm1d(input_size=out_planes[-1] * 2)
        self.fc = Conv1d(
            in_channels=out_planes[-1] * 2,
            out_channels=embedding_size,
            kernel_size=1,
        )

        # initialize weights:
        self._initialize_weights(zero_init_bn_residuals)

    def _initialize_weights(self, zero_init_bn_residuals):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Init BatchNorm gamma to 0.0 for last BN layer, it gets 0.2-0.3% higher
        # final val top1 for larger batch sizes.
        if zero_init_bn_residuals:
            for m in self.modules():
                if isinstance(m, GenericLayer):
                    if hasattr(m, "bn"):
                        nn.init.constant_(m.bn.weight, 0)

    def _make_initial_block(self, small_input, init_planes, basic_layer):
        if small_input:
            self.initial_block = SmallInputInitialBlock(init_planes)
            self.layer_type = BasicLayer
        else:
            self.initial_block = InitialBlock(init_planes)
            self.layer_type = BasicLayer if basic_layer else BottleneckLayer

    # helper function that creates ResNet blocks at single spatial resolution:
    def _make_resolution_block(
        self,
        in_planes,
        out_planes,
        resolution_idx,
        num_blocks,
        stride=1,
        mid_planes_and_cardinality=None,
        reduction=4,
        final_bn_relu=True,
        use_se=False,
        se_reduction_ratio=16,
    ):
        # add the desired number of residual blocks:
        blocks = OrderedDict()
        for idx in range(num_blocks):
            block_name = "block{}-{}".format(resolution_idx, idx)
            blocks[block_name] = self.layer_type(
                in_planes if idx == 0 else out_planes,
                out_planes,
                stride=stride if idx == 0 else 1,  # only first block has stride
                mid_planes_and_cardinality=mid_planes_and_cardinality,
                reduction=reduction,
                final_bn_relu=final_bn_relu or (idx != (num_blocks - 1)),
                use_se=use_se,
                se_reduction_ratio=se_reduction_ratio,
            )
        return nn.Sequential(blocks)

    def pooler(self, out):
        out = out.contiguous().view(out.shape[0], self.out_planes, -1)
        out = self.asp(out)
        out = self.asp_bn(out)
        out = self.fc(out)
        out = out.transpose(1, 2).squeeze(1)
        return out

    def forward(self, x):
        out = self.initial_block(x)
        out = self.blocks(out)
        out = self.pooler(out)
        return out


class InitialBlockForAudio(nn.Module):
    """
    A modified InitialBlock for audio data. Accepts a configurable
    number of input channels.
    """
    def __init__(self, in_channels: int, init_planes: int):
        super().__init__()
        self._module = nn.Sequential(
            # The key change is here: `in_channels` is now a parameter
            nn.Conv2d(in_channels, init_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self._module(x)


class ResNet18(ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[2, 2, 2, 2],
            basic_layer=True,
            zero_init_bn_residuals=True,
            reduction=1,
            **kwargs,
        )


class ResNet34(ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 6, 3],
            basic_layer=True,
            zero_init_bn_residuals=True,
            reduction=1,
            **kwargs,
        )


class ResNet50(ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 6, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            **kwargs,
        )


class ResNet101(ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 23, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            **kwargs,
        )


class ResNet152(ResNeXt):

    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 8, 36, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            **kwargs,
        )


class ResNeXt50(ResNeXt):
    """
    resnext50_32x4d
    """
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 6, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            base_width_and_cardinality=(4, 32),
            **kwargs,
        )


class ResNeXt101(ResNeXt):
    """
    resnext101_32x4d
    """
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 4, 23, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            base_width_and_cardinality=(4, 32),
            **kwargs,
        )


class ResNeXt152(ResNeXt):
    """
    resnext152_32x4d
    """
    def __init__(self, **kwargs):
        super().__init__(
            num_blocks=[3, 8, 36, 3],
            basic_layer=False,
            zero_init_bn_residuals=True,
            base_width_and_cardinality=(4, 32),
            **kwargs,
        )


class AudioResNet18(ResNet18):
    """
    A ResNet18 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNet18
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))


class AudioResNet34(ResNet34):
    """
    A ResNet34 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNet34
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))


class AudioResNet50(ResNet50):
    """
    A ResNet50 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNet50
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))


class AudioResNet101(ResNet101):
    """
    A ResNet101 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNet101
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))


class AudioResNet152(ResNet152):
    """
    A ResNet152 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNet152
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))
    

class AudioResNeXt50(ResNeXt50):
    """
    A ResNeXt50 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNeXt50
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))


class AudioResNeXt101(ResNeXt101):
    """
    A ResNeXt101 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNeXt101
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))


class AudioResNeXt152(ResNeXt152):
    """
    A ResNeXt152 model adapted for audio spectrogram inputs.

    This model modifies the initial block to accept a single-channel input
    (representing the spectrogram) instead of the standard 3-channel RGB image.
    """
    def __init__(self, **kwargs: Any):
        # First, initialize the standard ResNeXt152
        super().__init__(**kwargs)

        # Now, override the initial_block to handle single-channel input
        # We get the original number of output planes from the existing block
        init_planes = self.initial_block._module[0].out_channels

        # Create a new initial block that takes 1 channel as input
        self.initial_block = InitialBlockForAudio(in_channels=1, init_planes=init_planes)

    def forward(self, x):
        return super().forward(x.unsqueeze(dim=1))


if __name__ == '__main__':
    # 1. Create a dummy input tensor with the correct 4D shape
    batch_size = 16
    num_mels = 80
    num_frames = 3000
    dummy_audio_input = torch.randn(batch_size, num_mels, num_frames).to('cuda')

    # 2. Instantiate your new audio-ready model
    # You can still pass other ResNet50 arguments if needed
    audio_model = AudioResNeXt152().to('cuda')

    # 3. Perform a forward pass
    print(f"Input shape: {dummy_audio_input.shape}")
    output = audio_model(dummy_audio_input)

    # The output will be the feature map from the ResNet body
    print(f"Output shape: {output.shape}")

    # Example Output:
    # Input shape: torch.Size([16, 1, 80, 300])
    # Output shape: torch.Size([16, 2048, 3, 10])