import math
from typing import Optional, Tuple, Dict, Union

from rich.console import Console

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from torchaudio.transforms import SpecAugment
from huggingface_hub import PyTorchModelHubMixin
from transformers.modeling_outputs import ModelOutput

from ...logging import get_logger
from .configuration_titanet import TitaNetConfig
    
logger = get_logger()


class InputNormalization(torch.nn.Module):
    """Performs sentence-level mean and variance normalization of the input tensor."""

    def __init__(self, mean_norm=True, std_norm=True, eps=1e-10):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.eps = eps

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """Apply sentence-level normalization.

        Arguments
        ---------
        x : torch.Tensor
            A batch of input tensors with shape [batch, time, feat_dim].
        lengths : torch.Tensor
            A tensor containing the relative length of each sentence.

        Returns
        -------
        x : torch.Tensor
            The normalized tensor.
        """
        N_batches = x.shape[0]
        out = torch.empty_like(x)

        for snt_id in range(N_batches):
            actual_size = torch.round(lengths[snt_id] * x.shape[1]).int()
            x_snt = x[snt_id, 0:actual_size]

            mean, std = self._compute_current_stats(x_snt)
            out[snt_id] = (x[snt_id] - mean) / std

        return out

    def _compute_current_stats(self, x):
        """Compute mean and std for a single sentence."""
        mean = torch.mean(x, dim=0) if self.mean_norm else torch.tensor([0.0], device=x.device)
        std = torch.std(x, dim=0) if self.std_norm else torch.tensor([1.0], device=x.device)
        std = torch.max(std, self.eps * torch.ones_like(std))  # numerical stability
        return mean.detach(), std.detach()


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


class Conv1dSamePadding(nn.Conv1d):
    """
    1D convolutional layer with "same" padding (no downsampling),
    that is also compatible with strides > 1
    """

    def __init__(self, *args, **kwargs):
        super(Conv1dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where WO = [CI + 2P - K - (K - 1) * (D - 1)] / S + 1,
        by computing P on-the-fly ay forward time

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        P: padding
        K: kernel size
        D: dilation
        S: stride
        """
        padding = (
            self.stride[0] * (inputs.shape[-1] - 1)
            - inputs.shape[-1]
            + self.kernel_size[0]
            + (self.dilation[0] - 1) * (self.kernel_size[0] - 1)
        ) // 2
        return self._conv_forward(
            F.pad(inputs, (padding, padding)),
            self.weight,
            self.bias,
        )


class DepthwiseConv1d(nn.Module):
    """
    Compute a depth-wise separable convolution, by performing
    a depth-wise convolution followed by a point-wise convolution

    "Xception: Deep Learning with Depthwise Separable Convolutions",
    Chollet, https://arxiv.org/abs/1610.02357
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        device=None,
        dtype=None,
    ):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Sequential(
            Conv1dSamePadding(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            Conv1dSamePadding(
                in_channels, out_channels, kernel_size=1, device=device, dtype=dtype
            ),
        )

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where CO is given as a parameter and WO
        depends on the convolution operation attributes

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        """
        return self.conv(inputs)


class ConvBlock1d(nn.Module):
    """
    Standard convolution, normalization, activation block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        activation="relu",
        dropout=0,
        depthwise=False,
    ):
        super(ConvBlock1d, self).__init__()
        assert activation is None or activation in (
            "relu",
            "tanh",
        ), "Incompatible activation function"

        # Define architecture
        conv_module = DepthwiseConv1d if depthwise else Conv1dSamePadding
        modules = [
            conv_module(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activation is not None:
            modules += [nn.ReLU() if activation == "relu" else nn.Tanh()]
        if dropout > 0:
            modules += [nn.Dropout(p=dropout)]
        self.conv_block = nn.Sequential(*modules)

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where CO is given as a parameter and WO
        depends on the convolution operation attributes

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        """
        return self.conv_block(inputs)


class SqueezeExcitation(nn.Module):
    """
    The SE layer squeezes a sequence of local feature vectors into
    a single global context vector, broadcasts this context back to
    each local feature vector, and merges the two via multiplications

    "Squeeze-and-Excitation Networks", Hu et al.,
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()

        # Define architecture
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        """
        Given an input of shape [B, C, W], returns an
        output of the same shape

        B: batch size
        C: number of channels
        W: input width
        """
        # [B, C, W] -> [B, C]
        squeezed = self.squeeze(inputs).squeeze(-1)

        # [B, C] -> [B, C]
        excited = self.excitation(squeezed).unsqueeze(-1)

        # [B, C] -> [B, C, W]
        return inputs * excited.expand_as(inputs)


class Squeeze(nn.Module):
    """
    Remove dimensions of size 1 from the input tensor
    """

    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return inputs.squeeze(self.dim)


class MegaBlock(nn.Module):
    """
    The TitaNet mega block, part of its encoder, comprises a sequence
    of sub-blocks, where each one contains a time-channel separable
    convolution followed by batch normalization, activation and dropout;
    the output of the sequence of sub-blocks is then processed by a SE
    module and merged with the initial input through a skip connection

    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        n_sub_blocks,
        se_reduction=16,
        dropout=0.5,
    ):
        super(MegaBlock, self).__init__()

        # Store attributes
        self.dropout = dropout

        # Define sub-blocks composed of depthwise convolutions
        channels = [input_size] + [output_size] * n_sub_blocks
        self.sub_blocks = nn.Sequential(
            *[
                ConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    activation="relu",
                    dropout=dropout,
                    depthwise=True,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ],
            SqueezeExcitation(output_size, reduction=se_reduction)
        )

        # Define the final skip connection
        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size),
        )

    def forward(self, prolog_outputs):
        """
        Given prolog outputs of shape [B, H, T], return
        a feature tensor of shape [B, H, T]

        B: batch size
        H: hidden size
        T: maximum number of time steps (frames)
        """
        # [B, H, T] -> [B, H, T]
        mega_block_outputs = self.skip_connection(prolog_outputs) + self.sub_blocks(
            prolog_outputs
        )
        return F.dropout(
            F.relu(mega_block_outputs), p=self.dropout, training=self.training
        )


class TitaNetEncoder(nn.Module):
    """
    The TitaNet encoder starts with a prologue block, followed by a number
    of mega blocks and ends with an epilogue block; all blocks comprise
    convolutions, batch normalization, activation and dropout, while mega
    blocks are also equipped with residual connections and SE modules

    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
        self,
        num_mel_bins,
        num_mega_blocks,
        num_sub_blocks,
        encoder_hidden_size,
        encoder_output_size,
        mega_block_kernel_size,
        prolog_kernel_size=3,
        epilog_kernel_size=1,
        se_reduction=16,
        dropout=0.5,
    ):
        super().__init__()

        # Define encoder as sequence of prolog, mega blocks and epilog
        self.prolog = ConvBlock1d(num_mel_bins, encoder_hidden_size, prolog_kernel_size)
        self.mega_blocks = nn.Sequential(
            *[
                MegaBlock(
                    encoder_hidden_size,
                    encoder_hidden_size,
                    mega_block_kernel_size,
                    num_sub_blocks,
                    se_reduction=se_reduction,
                    dropout=dropout,
                )
                for _ in range(num_mega_blocks)
            ]
        )
        self.epilog = ConvBlock1d(encoder_hidden_size, encoder_output_size, epilog_kernel_size)

    def forward(self, spectrograms):
        """
        Given input spectrograms of shape [B, M, T], return encodings
        of shape [B, DE, T]

        B: batch size
        M: number of mel frequency bands
        T: maximum number of time steps (frames)
        DE: encoding output size
        H: hidden size
        """
        # [B, M, T] -> [B, H, T]
        prolog_outputs = self.prolog(spectrograms)

        # [B, H, T] -> [B, H, T]
        mega_blocks_outputs = self.mega_blocks(prolog_outputs)

        # [B, H, T] -> [B, DE, T]
        return self.epilog(mega_blocks_outputs)


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


class TitaNetDecoder(nn.Module):
    """
    The TitaNet decoder computes intermediate time-independent features
    using an attentive statistics pooling layer and downsamples such
    representation using two linear layers, to obtain a fixed-size
    embedding vector first and class logits afterwards

    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
        self,
        encoder_output_size,
        attention_hidden_size: int = 128,
        emb_sizes: int = 192,
        global_context: bool = True
    ):
        super().__init__()
        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            encoder_output_size,
            attention_channels=attention_hidden_size,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=encoder_output_size * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=encoder_output_size * 2,
            out_channels=emb_sizes,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x


class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape : tuple
        It is the shape of the input tensor.
    input_size : int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    max_norm : float
        weight max-norm.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        max_norm=None,
        combine_dims=False,
    ):
        super().__init__()
        self.max_norm = max_norm
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.

        Returns
        -------
        wx : torch.Tensor
            The linearly transformed outputs.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        if self.max_norm is not None:
            self.w.weight.data = torch.renorm(
                self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        wx = self.w(x)

        return wx


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    input_size : int
        Expected size of input dimension.
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.

        Returns
        -------
        out : torch.Tensor
            Output probabilities over speakers.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)


class AngularMargin(nn.Module):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity
    scale : float
        The scale for cosine similarity

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity.
    scale : float
        The scale for cosine similarity.
    easy_margin : bool

    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super().__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class LogSoftmaxWrapper(nn.Module):
    """
    Arguments
    ---------
    loss_fn : Callable
        The LogSoftmax function to wrap.

    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([ [0], [1], [0], [1] ])
    >>> log_prob = LogSoftmaxWrapper(nn.Identity())
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> log_prob = LogSoftmaxWrapper(AngularMargin(margin=0.2, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.3, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    """

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].
        length : torch.Tensor
            The lengths of the corresponding inputs.

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


class TitaNetForSpeakerClassification(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config: TitaNetConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_labels
        self.mean_var_norm = InputNormalization(std_norm=False)
        self.encoder = TitaNetEncoder(
            num_mel_bins=config.num_mel_bins,
            num_mega_blocks=config.num_mega_blocks,
            num_sub_blocks=config.num_sub_blocks,
            encoder_hidden_size=config.encoder_hidden_size,
            encoder_output_size=config.encoder_output_size,
            mega_block_kernel_size=config.mega_block_kernel_size,
            prolog_kernel_size=config.prolog_kernel_size,
            epilog_kernel_size=config.epilog_kernel_size,
            se_reduction=config.se_reduction,
            dropout=config.dropout,
        )
        self.decoder = TitaNetDecoder(
            encoder_output_size=config.encoder_output_size,
            attention_hidden_size=config.attention_hidden_size,
            emb_sizes=config.emb_sizes,
        )
        self.classifier = Classifier(
            input_size=config.emb_sizes, 
            lin_blocks=0, 
            lin_neurons=config.emb_sizes, 
            out_neurons=self.num_classes
        )
        self.loss_fn = LogSoftmaxWrapper(
            AdditiveAngularMargin(config.margin, config.scale)
        )

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        B, C, T = input_features.shape
        # Input tensor should be (B, T, C) for InputNormalization
        input_features = self.mean_var_norm(
            input_features.transpose(1, 2), torch.ones(B).to(input_features.device)
        )
        if self.training:
            spec_augment = SpecAugment(
                n_time_masks=self.config.num_time_masks, 
                time_mask_param=self.config.time_mask_width, 
                n_freq_masks=self.config.num_freq_masks, 
                freq_mask_param=self.config.freq_mask_width, 
                zero_masking=True
            )
            # Input tensor should be (B, C, T) for SpecAugment
            input_features = spec_augment(input_features.transpose(1, 2))
            input_features = input_features.transpose(1, 2)
        # Input tensor should be (B, C, T) for TitaNetEncoder
        encodings = self.encoder(input_features.transpose(1, 2))
        embeddings = self.decoder(encodings)
        logits = self.classifier(embeddings)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(B, 1, self.num_classes), labels.view(-1, 1))

        return ModelOutput(
            loss=loss,
            logits=logits.view(B, self.num_classes),
            embeddings=embeddings.view(B, self.config.emb_sizes)
        )