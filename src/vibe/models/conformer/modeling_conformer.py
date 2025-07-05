import math
from typing import List, Optional, Tuple, Union

import numpy as np
from rich.console import Console

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torchaudio.transforms import SpecAugment
from huggingface_hub import PyTorchModelHubMixin
from transformers.modeling_outputs import ModelOutput

from ...logging import get_logger
from .configuration_conformer import ConformerConfig
from .feature_extractor_conformer import ConformerFeatureExtractor

logger = get_logger()

WENET_NORM_CLASSES = {
    'layer_norm': nn.LayerNorm,
    'batch_norm': nn.BatchNorm1d,
    'rms_norm': nn.RMSNorm
}


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


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True,
                            max_chunk_size: int = 25):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, max_chunk_size] or full context(max_len)
            False: chunk size ~ U[1, max_chunk_size]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, max_chunk_size] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % max_chunk_size + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


# copy from:https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L84
def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# modified from:
#     https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L95
def google_apply_rotary_emb(x: torch.Tensor,
                            freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    return x_out


def llama_apply_rotary_emb(x: torch.Tensor,
                           freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + \
                torch.arange(0, size).to(offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):
    """ Sinusoids position encoding used in openai-whisper.encoder
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment *
                                   torch.arange(d_model // 2))
        scaled_time = torch.arange(max_len)[:, np.newaxis] * \
            inv_timescales[np.newaxis, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        delattr(self, "pe")
        self.register_buffer("pe", pe.unsqueeze(0))


class LearnablePositionalEncoding(PositionalEncoding):
    """ Learnable position encoding used in openai-whisper.decoder
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        super().__init__(d_model, dropout_rate, max_len)
        # NOTE(xcsong): overwrite self.pe & self.xscale
        self.pe = torch.nn.Parameter(torch.empty(1, max_len, d_model))
        self.xscale = 1.0


class NoPositionalEncoding(torch.nn.Module):
    """ No position encoding
    """

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return torch.zeros(1, size, self.d_model)


class RopePositionalEncoding(PositionalEncoding):

    def __init__(self,
                 d_model: int,
                 head_dim: int,
                 dropout_rate: float,
                 max_len: int = 3000,
                 rope_theta=10000.0,
                 scale: bool = True):
        super().__init__(d_model, dropout_rate=dropout_rate, max_len=max_len)
        delattr(self, 'pe')
        self.max_len = max_len * 2
        pe = precompute_freqs_cis(head_dim, self.max_len, rope_theta)
        self.register_buffer("pe", torch.view_as_real(pe.unsqueeze(0)))
        self.dropout_rate = dropout_rate
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int,
                      torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:

        pos_emb = self.position_encoding(offset, x.size(1), True)
        pos_emb = pos_emb.unsqueeze(2)  # [1,seq, 1, head_dim//2]
        # NOTE(Mddct): some model don't scale
        if self.scale:
            x = x * self.xscale
        return self.dropout(x), pos_emb

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:

        pe = torch.view_as_complex(self.pe)
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = pe[:, offset:offset + size]
        else:
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size).to(
                offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, pe[0])  # B X T X head_dim//2
        if apply_dropout:
            # NOTE(Mddct) dropout don't suuport complex float for pos_emb
            pos_emb = self.dropout_complex(pos_emb)
        return pos_emb

    def dropout_complex(self, x):
        mask = torch.nn.functional.dropout(
            torch.ones_like(x.real),
            training=self.training,
            p=self.dropout_rate,
        )
        return x * mask


WENET_APPLY_ROTARY_EMB = {
    'google': google_apply_rotary_emb,
    'llama': llama_apply_rotary_emb,
}
T_CACHE = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    if n_kv_head != None and n_kv_head != n_head
    see: https://arxiv.org/pdf/1911.02150.pdf
         https://arxiv.org/pdf/2305.13245.pdf

    Example:
        case 1: n_kv_head == None, head_dim == None, MultiHead attention (MHSA)
        case 2: n_kv_head=1, n_head = 16, MultiQuery attention (MQA)
        case 3: nv_kv_head=2, n_head = 16, GroupedQuery attention (GQA)

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.inner_dim = n_feat if head_dim is None else head_dim * n_head
        if n_kv_head is not None:
            assert head_dim is not None
            self.inner_kv_dim = head_dim * n_kv_head
            n_kv_head = n_kv_head
        else:
            self.inner_kv_dim = self.inner_dim
            n_kv_head = n_head
        # We assume d_v always equals d_k
        self.d_k = self.inner_dim // n_head
        assert self.d_k == self.inner_kv_dim // n_kv_head
        self.h = n_head
        self.h_kv = n_kv_head

        self.linear_q = nn.Linear(n_feat, self.inner_dim, bias=query_bias)
        self.linear_k = nn.Linear(n_feat, self.inner_kv_dim, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, self.inner_kv_dim, bias=value_bias)
        self.linear_out = nn.Linear(self.inner_dim, n_feat, bias=query_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.use_sdpa = use_sdpa
        self.dropout_rate = dropout_rate

    def _forward_linearx(self,
                         name: str,
                         x: torch.Tensor,
                         head_first: bool = True) -> torch.Tensor:
        assert x.ndim >= 3
        if name == 'query':
            x = self.linear_q(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h, self.d_k])
        elif name == 'key':
            x = self.linear_k(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])
        else:
            assert name == 'value'
            x = self.linear_v(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])

        # split last dim
        x = x.view(x_shape)
        if head_first:
            x = x.transpose(-3,
                            -2)  # (batch, ...,  head or head_kv, time, d_k)
        return x

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, ..., time1, size).
            key (torch.Tensor): Key tensor (#batch, ..., time2, size).
            value (torch.Tensor): Value tensor (#batch, ..., time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, ..., n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, ..., n_head_kv, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, ..., n_head_kv, time2, d_k).

        """
        q = self._forward_linearx('query', query)
        k = self._forward_linearx('key', key)
        v = self._forward_linearx('value', value)
        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, ..., n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, ..., n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, ..., time1, time2), (0, ..., 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(-1) > 0:  # time2 > 0
            mask = mask.unsqueeze(-3).eq(0)  # (batch, .., 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[..., :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores.float(),
                                 dim=-1).type_as(value).masked_fill(
                                     mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores.float(), dim=-1).type_as(
                value)  # (batch, ..., head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, ...,  head, time1, d_k)
        x = x.transpose(-3, -2).contiguous()  # [batch, ..., time1, head, d_k]
        x_shape = x.size()[:-2] + torch.Size([self.h * self.d_k])
        x = x.view(x_shape)  # (batch, ..., time1, d_model)
        return self.linear_out(x)  # (batch, ...,  time1, d_model)

    def _update_kv_and_cache(
            self,
            k: torch.Tensor,
            v: torch.Tensor,
            cache: T_CACHE,
            head_first: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE]:
        new_cache = cache
        seq_axis = -2 if head_first else -3
        head_axis = -3 if head_first else -2
        if not self.training:
            # NOTE(xcsong):
            #   when export onnx model, for 1st chunk, we feed
            #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
            #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
            #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
            #       and we will always do splitting and
            #       concatnation(this will simplify onnx export). Note that
            #       it's OK to concat & split zero-shaped tensors(see code below).
            #   when export jit  model, for 1st chunk, we always feed
            #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
            # >>> a = torch.ones((1, 2, 0, 4))
            # >>> b = torch.ones((1, 2, 3, 4))
            # >>> c = torch.cat((a, b), dim=2)
            # >>> torch.equal(b, c)        # True
            # >>> d = torch.split(a, 2, dim=-1)
            # >>> torch.equal(d[0], d[1])  # True
            key_cache, value_cache = cache
            if key_cache.size(0) > 0:
                k = torch.cat([key_cache, k], dim=seq_axis)
            if value_cache.size(0) > 0:
                v = torch.cat([value_cache, v], dim=seq_axis)
            # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
            #   non-trivial to calculate `next_cache_start` here.
            # new_cache = torch.cat((k, v), dim=-1) if not self.training else cache
            new_cache = (k, v)
        # for multi query or multi group attention
        if self.h_kv != self.h and self.h_kv != 1:
            # NOTE: onnxruntime issues:
            #     https://github.com/wenet-e2e/wenet/issues/2517
            # k = torch.repeat_interleave(
            #     k,
            #     self.h // self.h_kv,
            #     dim=-3,
            # )
            # v = torch.repeat_interleave(
            #     v,
            #     self.h // self.h_kv,
            #     dim=-3,
            # )
            n_repeat = self.h // self.h_kv
            k_shape = k.size()
            repeat_axis = head_axis + 1
            k = k.unsqueeze(head_axis).expand(
                k_shape[:repeat_axis] + torch.Size([n_repeat]) +
                k_shape[repeat_axis:]).reshape(
                    k_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) +
                    k_shape[repeat_axis:])
            v_shape = v.size()
            v = v.unsqueeze(head_axis).expand(
                v_shape[:repeat_axis] + torch.Size([n_repeat]) +
                v_shape[(repeat_axis):]).reshape(
                    v_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) +
                    v_shape[repeat_axis:])

        return k, v, new_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.unsqueeze(1),
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            output = (output.transpose(1, 2).contiguous().view(
                query.size(0), -1,
                self.h * self.d_k))  # (batch, time1, d_model)
            return self.linear_out(output), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa, n_kv_head, head_dim)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)
        if not self.use_sdpa:
            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

            scores = (matrix_ac + matrix_bd) / math.sqrt(
                self.d_k)  # (batch, head, time1, time2)

            return self.forward_attention(v, scores, mask), new_cache
        else:
            # NOTE(Mddct): we need mask bias, not boolean mask
            assert mask.dtype != torch.bool
            mask = mask.unsqueeze(1)
            # matrix_bd as a mask bias
            mask = (matrix_bd + mask) / math.sqrt(self.d_k)
            output = torch.nn.functional.scaled_dot_product_attention(
                q_with_bias_u,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            output = (output.transpose(1, 2).contiguous().view(
                query.size(0), -1,
                self.h * self.d_k))  # (batch, time1, d_model)
            return self.linear_out(output), new_cache


class MultiHeadedCrossAttention(MultiHeadedAttention):

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa, n_kv_head, head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))
    ) -> Tuple[torch.Tensor, T_CACHE]:
        del pos_emb
        key_cache, value_cache = cache
        assert key_cache.size(0) == value_cache.size(0)
        if key_cache.size(0) > 0:
            assert not self.training
            q = self._forward_linearx('query', query)
            k, v = key_cache, value_cache

        else:
            q, k, v = self.forward_qkv(query, key, value)
        new_cache = (k, v) if not self.training else cache
        # for multi query or multi groups attention
        if self.h_kv != self.h and self.h_kv != 1:
            k = torch.repeat_interleave(
                k,
                self.h // self.h_kv,
                dim=-3,
            )
            v = torch.repeat_interleave(
                v,
                self.h // self.h_kv,
                dim=-3,
            )
        B = query.size(0)
        Beams = 1
        if B != k.size(0):
            assert not self.training
            Beams = B // k.size(0)
            B = k.size(0)
            q = q.view(B, Beams, q.size(-3), q.size(-2), q.size(-1))
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            mask = mask.unsqueeze(1)

        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            output = self.forward_attention(v, scores, mask)
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.unsqueeze(1),
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            output = output.transpose(-2, -3).contiguous()
            output_shape = output.size()[:-2] + torch.Size([self.h * self.d_k])
            output = output.view(output_shape)  # (batch, ...,  time1, d_model)
            output = self.linear_out(output)

        if query.size(0) != B:
            assert not self.training
            output_shape = torch.Size([B * Beams]) + output.size()[2:]
            output = output.view(output_shape)
        return output, new_cache


class ShawRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """ https://arxiv.org/pdf/1803.02155.pdf
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        del n_kv_head, head_dim
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa, None, None)
        # TODO(Mddct): 64 8 1 as args
        self.max_right_rel_pos = 8
        self.max_left_rel_pos = 64
        self.rel_k_embed = torch.nn.Embedding(
            self.max_left_rel_pos + self.max_right_rel_pos + 1, self.d_k)

    def _relative_indices(self, keys: torch.Tensor) -> torch.Tensor:
        # (S, 1)
        indices = torch.arange(keys.size(2), device=keys.device).unsqueeze(0)

        # (S, S)
        rel_indices = indices - indices.transpose(0, 1)

        rel_indices = torch.clamp(rel_indices, -self.max_left_rel_pos,
                                  self.max_right_rel_pos)

        return rel_indices + self.max_left_rel_pos

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, T_CACHE]:
        del pos_emb
        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        rel_k = self.rel_k_embed(self._relative_indices(k))  # (t2, t2, d_k)
        rel_k = rel_k[-q.size(2):]
        rel_att_weights = torch.einsum("bhld,lrd->bhlr", q, rel_k)

        if not self.use_sdpa:
            scores = (torch.matmul(q, k.transpose(-2, -1)) +
                      rel_att_weights) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            # NOTE(Mddct): we need mask bias, not boolean mask
            assert mask.dtype != torch.bool
            mask = mask.unsqueeze(1)
            # matrix_bd as a mask bias
            mask = (rel_att_weights + mask) / math.sqrt(self.d_k)
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            output = (output.transpose(1, 2).contiguous().view(
                query.size(0), -1,
                self.h * self.d_k))  # (batch, time1, d_model)
            return self.linear_out(output), new_cache


class RopeMultiHeadedAttention(MultiHeadedAttention):

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None,
                 style='google'):
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa, n_kv_head, head_dim)
        self.style = style

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute rope scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q = self._forward_linearx('query', query, head_first=False)
        k = self._forward_linearx('key', key, head_first=False)
        v = self._forward_linearx('value', value, head_first=False)
        # NOTE(Mddct): In order to make the code easier to read,
        #    these two lines are not placed in MultiHeadedAttention.
        q = WENET_APPLY_ROTARY_EMB[self.style](q, pos_emb)
        k = WENET_APPLY_ROTARY_EMB[self.style](k, pos_emb)

        k, v, new_cache = self._update_kv_and_cache(k,
                                                    v,
                                                    cache,
                                                    head_first=False)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.unsqueeze(1),
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            output = (output.transpose(1, 2).contiguous().view(
                query.size(0), -1,
                self.h * self.d_k))  # (batch, time1, d_model)
            return self.linear_out(output), new_cache


class GroupedRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper:
        https://arxiv.org/abs/1901.02860
        https://arxiv.org/abs/2109.01163
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, group_size=3):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.group_size = group_size
        self.d_k = n_feat // n_head  # for GroupedAttention
        self.n_feat = n_feat
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(
            torch.Tensor(self.h, self.d_k * self.group_size))
        self.pos_bias_v = nn.Parameter(
            torch.Tensor(self.h, self.d_k * self.group_size))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def pad4group(self, Q, K, V, P, mask, group_size: int = 3):
        """
        q: (#batch, time1, size) -> (#batch, head, time1, size/head)
        k,v: (#batch, time2, size) -> (#batch, head, time2, size/head)
        p: (#batch, time2, size)
        """
        # Compute Overflows
        overflow_Q = Q.size(2) % group_size
        overflow_KV = K.size(2) % group_size

        # if-else for ONNX export
        #   0 // 0.00000000000000001 = 0
        #   1 // 1.00000000000000001 = 1
        padding_Q = (group_size - overflow_Q) * int(
            overflow_Q // (overflow_Q + 0.00000000000000001))
        padding_KV = (group_size - overflow_KV) * int(
            overflow_KV // (overflow_KV + 0.00000000000000001))

        batch_size, _, seq_len_KV, _ = K.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0.0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0.0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0.0)

        if mask is not None and mask.size(2) > 0:  # time2 > 0:
            mask = mask[:, ::group_size, ::group_size]

        Q = Q.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        K = K.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        V = V.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)

        # process pos_emb
        P_batch_size = P.size(0)
        overflow_P = P.size(1) % group_size
        padding_P = group_size - overflow_P if overflow_P else 0
        P = F.pad(P, (0, 0, 0, padding_P), value=0.0)
        P = P.view(P_batch_size, -1, self.h,
                   self.d_k * group_size).transpose(1, 2)

        return Q, K, V, P, mask, padding_Q

    def forward_attention(self,
                          value: torch.Tensor,
                          scores: torch.Tensor,
                          mask: torch.Tensor = torch.ones((0, 0, 0),
                                                          dtype=torch.bool),
                          padding_q: Optional[int] = None) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            padding_q : for GroupedAttention in efficent conformer

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)

        # n_feat!=h*d_k may be happened in GroupAttention
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_feat)
             )  # (batch, time1, d_model)
        if padding_q is not None:
            # for GroupedAttention in efficent conformer
            x = x[:, :x.size(1) - padding_q]

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q = self.linear_q(query)
        k = self.linear_k(key)  # (#batch, time2, size)
        v = self.linear_v(value)
        p = self.linear_pos(pos_emb)  # (#batch, time2, size)

        batch_size, seq_len_KV, _ = k.size()  # seq_len_KV = time2

        # (#batch, time2, size) -> (#batch, head, time2, size/head)
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        if cache.size(0) > 0:
            # use attention cache
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        # May be k and p does not match.  eg. time2=18+18/2=27 > mask=36/2=18
        if mask is not None and mask.size(2) > 0:
            time2 = mask.size(2)
            k = k[:, :, -time2:, :]
            v = v[:, :, -time2:, :]

        # q k v p: (batch, head, time1, d_k)
        q, k, v, p, mask, padding_q = self.pad4group(q, k, v, p, mask,
                                                     self.group_size)

        # q_with_bias_u & q_with_bias_v = (batch, head, time1, d_k)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k * self.group_size)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask, padding_q), new_cache


class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class EmbedinigNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        super().__init__()
        self.embed = torch.nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 4 = (3 - 1) * 1 + (3 - 1) * 1
        self.right_context = 4

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        time = x.size(1)
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (b, t, f)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2::2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3),
                                      odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


class StackNFramesSubsampling(BaseSubsampling):

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: torch.nn.Module,
                 stride: int = 2):

        super().__init__()
        del dropout_rate
        self.pos_enc_class = pos_enc_class
        self.stride = stride
        self.idim = idim

        self.norm = torch.nn.LayerNorm(idim * stride, eps=1e-5)
        self.out = torch.nn.Linear(idim * stride, odim)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // stride.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // stride.
            torch.Tensor: positional encoding
        """
        with torch.no_grad():
            b, s, _ = x.size()

            seq_len = x_mask.sum(-1).view(b)
            r = s % self.stride
            s -= r
            x = x[:, :s, :]
            seq_len = torch.where(seq_len > s, s, seq_len)
            seq_len = seq_len // self.stride
            new_mask = ~make_pad_mask(seq_len, max_len=s // self.stride)
            x = x.view(b, s // self.stride, self.idim * self.stride)
            _, pos_emb = self.pos_enc_class(x, offset)
        x = self.norm(x)
        x = self.out(x)
        return x, pos_emb, new_mask.unsqueeze(1)

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc_class.position_encoding(offset, size)


class Conv2dSubsampling2(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2),
                                        torch.nn.ReLU())
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 2 = (3 - 1) * 1
        self.right_context = 2

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2]


class DepthwiseConv2dSubsampling4(BaseSubsampling):
    """Depthwise Convolutional 2D subsampling (to 1/4 length).

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            pos_enc_class (nn.Module): position encoding class.
            dw_stride (int): Whether do depthwise convolution.
            input_size (int): filter bank dimension.

        """

    def __init__(self,
                 idim: int,
                 odim: int,
                 pos_enc_class: torch.nn.Module,
                 dw_stride: bool = False,
                 input_size: int = 80,
                 input_dropout_rate: float = 0.1,
                 init_weights: bool = True):
        super(DepthwiseConv2dSubsampling4, self).__init__()
        self.idim = idim
        self.odim = odim
        self.pw_conv = nn.Conv2d(in_channels=idim,
                                 out_channels=odim,
                                 kernel_size=3,
                                 stride=2)
        self.act1 = nn.ReLU()
        self.dw_conv = nn.Conv2d(in_channels=odim,
                                 out_channels=odim,
                                 kernel_size=3,
                                 stride=2,
                                 groups=odim if dw_stride else 1)
        self.act2 = nn.ReLU()
        self.pos_enc = pos_enc_class
        self.input_proj = nn.Sequential(
            nn.Linear(odim * (((input_size - 1) // 2 - 1) // 2), odim),
            nn.Dropout(p=input_dropout_rate),
        )
        if init_weights:
            linear_max = (odim * input_size / 4)**-0.5
            torch.nn.init.uniform_(self.input_proj.state_dict()['0.weight'],
                                   -linear_max, linear_max)
            torch.nn.init.uniform_(self.input_proj.state_dict()['0.bias'],
                                   -linear_max, linear_max)
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.pw_conv(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(b, t, c * f)
        x, pos_emb = self.pos_enc(x, offset)
        x = self.input_proj(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths)


class FireRedConv2dSubsampling4(Conv2dSubsampling4):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 idim: int,
                 d_model: int,
                 dropout_rate: float,
                 pos_enc_class: torch.nn.Module,
                 odim: int = 32):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__(idim, d_model, dropout_rate, pos_enc_class)
        del self.conv, self.out
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), d_model))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_lens = torch.sum(x_mask.squeeze(1), dim=1)
        x_lens = x_lens + self.right_context
        x_mask = make_non_pad_mask(x_lens).unsqueeze(1)
        x = torch.nn.functional.pad(x, (0, 0, 0, self.right_context),
                                    'constant', 0.0)
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        return x, pos_emb, mask


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 adaptive_scale: bool = False,
                 init_weights: bool = False):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.idim = idim
        self.hidden_units = hidden_units
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.ada_scale = None
        self.ada_bias = None
        self.adaptive_scale = adaptive_scale
        self.ada_scale = torch.nn.Parameter(torch.ones([1, 1, idim]),
                                            requires_grad=adaptive_scale)
        self.ada_bias = torch.nn.Parameter(torch.zeros([1, 1, idim]),
                                           requires_grad=adaptive_scale)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        ffn1_max = self.idim**-0.5
        ffn2_max = self.hidden_units**-0.5
        torch.nn.init.uniform_(self.w_1.weight.data, -ffn1_max, ffn1_max)
        torch.nn.init.uniform_(self.w_1.bias.data, -ffn1_max, ffn1_max)
        torch.nn.init.uniform_(self.w_2.weight.data, -ffn2_max, ffn2_max)
        torch.nn.init.uniform_(self.w_2.bias.data, -ffn2_max, ffn2_max)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        if self.adaptive_scale:
            xs = self.ada_scale * xs + self.ada_bias
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class MoEFFNLayer(torch.nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_activated: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = False,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        self.experts = torch.nn.ModuleList(
            PositionwiseFeedForward(
                idim, hidden_units, dropout_rate, activation, bias=bias)
            for _ in range(n_expert))
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        B, L, D = xs.size(
        )  # batch size, sequence length, embedding dimension (idim)
        xs = xs.view(-1, D)  # (B*L, D)
        router = self.gate(xs)  # (B*L, n_expert)
        logits, selected_experts = torch.topk(
            router, self.n_expert_activated
        )  # probs:(B*L, n_expert_activated), selected_exp: (B*L, n_expert_activated)
        weights = torch.nn.functional.softmax(
            logits, dim=1,
            dtype=torch.float).to(dtype=xs.dtype)  # (B*L, n_expert_activated)
        output = torch.zeros_like(xs)  # (B*L, D)
        for i, expert in enumerate(self.experts):
            mask = selected_experts == i
            token_ids, ith_expert = torch.where(mask)
            output[token_ids] += weights[token_ids, ith_expert, None] * expert(
                xs[token_ids])
        return output.view(B, L, D)


class GatedVariantsMLP(torch.nn.Module):
    """ https://arxiv.org/pdf/2002.05202.pdf
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.GELU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(GatedVariantsMLP, self).__init__()
        self.gate = torch.nn.Linear(idim, hidden_units, bias=False)
        self.activation = activation
        # w_1 as up proj
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.dropout = torch.nn.Dropout(dropout_rate)
        # w_2 as down proj
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, x) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        gate = self.activation(self.gate(x))
        up = self.w_1(x)
        fuse = gate * up
        return self.w_2(self.dropout(fuse))


WENET_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    # "abs_pos_paraformer": ParaformerPositinoalEncoding,
    'rope_pos': RopePositionalEncoding,
    # 'rel_pos_firered': FireRedRelPositionalEncoding
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "grouped_rel_selfattn": GroupedRelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
    # 'shaw_rel_selfattn': ShawRelPositionMultiHeadedAttention,
    'rope_abs_selfattn': RopeMultiHeadedAttention,
    # 'firered_rel_selfattn': FiredRelPositionMultiHeadedAttention
}

WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d2": Conv2dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "dwconv2d4": DepthwiseConv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity,
    'stack_n_frames': StackNFramesSubsampling,
    'firered_conv2d4': FireRedConv2dSubsampling4
}

WENET_MLP_CLASSES = {
    'position_wise_feed_forward': PositionwiseFeedForward,
    'moe': MoEFFNLayer,
    'gated': GatedVariantsMLP
}

WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": nn.SiLU,
    "gelu": torch.nn.GELU,
}


class BaseEncoder(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        final_norm: bool = True,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            query_bias: whether use bias in attention.linear_q
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            value_bias: whether use bias in attention.linear_v
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
            use_sdpa: whether to use SDPA, currently only support transformer for now
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        pos_emb_class = WENET_EMB_CLASSES[pos_enc_layer_type]
        # NOTE(Mddct): head_dim == output_size // attention_heads for most of
        #    speech tasks,  but for other task (LLM),
        #    head_dim == hidden_size * attention_heads. refactor later
        self.embed = WENET_SUBSAMPLE_CLASSES[input_layer](
            input_size, output_size, dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate)
            if pos_enc_layer_type != 'rope_pos' else pos_emb_class(
                output_size, output_size //
                attention_heads, positional_dropout_rate))

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.final_norm = final_norm
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size,
                                                              eps=norm_eps)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
            # Since we allow up to 1s(100 frames) delay, the maximum
            # chunk_size is 100 / 4 = 25.
            max_chunk_size=int(100.0 / self.embed.subsampling_rate))
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks, pos_emb,
                                                  mask_pad)
        else:
            xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    @torch.jit.unused
    def forward_layers_checkpointed(self, xs: torch.Tensor,
                                    chunk_masks: torch.Tensor,
                                    pos_emb: torch.Tensor,
                                    mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = ckpt.checkpoint(layer.__call__,
                                                    xs,
                                                    chunk_masks,
                                                    pos_emb,
                                                    mask_pad,
                                                    use_reentrant=False)
        return xs

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1,
                                               size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            if elayers == 0:
                kv_cache = (att_cache, att_cache)
            else:
                i_kv_cache = att_cache[i:i + 1]
                size = att_cache.size(-1) // 2
                kv_cache = (i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :,
                                                                   size:])
            xs, _, new_kv_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=kv_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache,
             cnn_cache) = self.forward_chunk(chunk_xs, offset,
                                             required_cache_size, att_cache,
                                             cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)),
                           device=ys.device,
                           dtype=torch.bool)
        return ys, masks


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = WENET_NORM_CLASSES[layer_norm_type](
            size, eps=norm_eps)  # for the FNN module
        self.norm_mha = WENET_NORM_CLASSES[layer_norm_type](
            size, eps=norm_eps)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)  # for the CNN module
            self.norm_final = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: T_CACHE = (torch.zeros(
            (0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
        norm_eps: float = 1e-5,
        conv_inner_factor: int = 2,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            conv_inner_factor * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            conv_inner_factor * channels // 2,
            conv_inner_factor * channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            groups=conv_inner_factor * channels // 2,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm', 'rms_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = WENET_NORM_CLASSES['batch_norm'](conv_inner_factor *
                                                         channels // 2,
                                                         eps=norm_eps)
        else:
            self.use_layer_norm = True
            self.norm = WENET_NORM_CLASSES[norm](conv_inner_factor *
                                                 channels // 2,
                                                 eps=norm_eps)

        self.pointwise_conv2 = nn.Conv1d(
            conv_inner_factor * channels // 2,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module with Multi-Layer Feature Aggregation."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
        conv_norm_eps: float = 1e-5,
        conv_inner_factor: int = 2,
        final_norm: bool = True,
        use_mfa: bool = True,  # Add MFA option
    ):
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, gradient_checkpointing,
                         use_sdpa, layer_norm_type, norm_eps, final_norm)
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, conv_bias,
                                  conv_norm_eps, conv_inner_factor)

        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                WENET_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])
        
        # Add Multi-Layer Feature Aggregation
        self.use_mfa = use_mfa
        if use_mfa:
            # MFA TDNN layer - combine features from all layers
            self.mfa = TDNNBlock(
                output_size * (num_blocks),  # Concatenate all layer outputs
                output_size,                 # Output same dimension as each layer
                kernel_size=1,
                dilation=1,
                activation=nn.ReLU,
                dropout=dropout_rate,
            )

    def forward(
        self,
        xs: torch.Tensor,
        lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xs: padded input tensor (B, T, D)
            lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use full chunk.
                > 0: for decoding, use fixed chunk size as set.
                < 0: for decoding, use full chunk.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks

        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        masks = ~make_pad_mask(lens, xs.size(1)).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        
        # Collect layer outputs for MFA if enabled
        layer_outputs = []
        
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            layer_outputs.append(xs)
            
        # Apply Multi-Layer Feature Aggregation if enabled
        if self.use_mfa:
            # Convert to the format expected by TDNNBlock (B, C, T)
            layer_outputs_t = [output.transpose(1, 2) for output in layer_outputs]
            
            # Concatenate along channel dimension
            concat_features = torch.cat(layer_outputs_t, dim=1)
            
            # Apply MFA block
            mfa_output = self.mfa(concat_features)
            
            # Convert back to (B, T, C) format
            xs = mfa_output.transpose(1, 2)

        if self.final_norm:
            xs = self.after_norm(xs)
            
        return xs, masks


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


# from wenet.transformer.encoder import ConformerEncoder


class ConformerForSpeakerClassification(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.mean_var_norm = InputNormalization(std_norm=False)
        self.conformer = ConformerEncoder(
            input_size=config.num_mel_bins, 
            output_size=config.output_hidden_size,
            attention_heads=config.attention_heads,
            linear_units=config.linear_units,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout_rate,
            positional_dropout_rate=config.positional_dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            input_layer=config.input_layer,
            pos_enc_layer_type=config.pos_enc_layer_type,
            normalize_before=config.normalize_before,
            static_chunk_size=config.static_chunk_size,
            use_dynamic_chunk=config.use_dynamic_chunk,
            use_dynamic_left_chunk=config.use_dynamic_left_chunk,
            positionwise_conv_kernel_size=config.positionwise_conv_kernel_size,
            macaron_style=config.macaron_style,
            selfattention_layer_type=config.selfattention_layer_type,
            activation_type=config.activation_type,
            use_cnn_module=config.use_cnn_module,
            cnn_module_kernel=config.cnn_module_kernel,
            causal=config.causal,
            cnn_module_norm=config.cnn_module_norm,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            conv_bias=config.conv_bias,
            gradient_checkpointing=config.gradient_checkpointing,
            use_sdpa=config.use_sdpa,
            layer_norm_type=config.layer_norm_type,
            norm_eps=config.norm_eps,
            n_kv_head=config.n_kv_head,
            head_dim=config.head_dim,
            mlp_type=config.mlp_type,
            mlp_bias=config.mlp_bias,
            n_expert=config.n_expert,
            n_expert_activated=config.n_expert_activated,
            conv_norm_eps=config.conv_norm_eps,
            conv_inner_factor=config.conv_inner_factor,
            final_norm=config.final_norm
        )
        self.asp = AttentiveStatisticsPooling(config.output_hidden_size)
        self.asp_bn = BatchNorm1d(input_size=config.output_hidden_size * 2)
        # Final linear transformation
        self.fc = Conv1d(
            in_channels=config.output_hidden_size * 2,
            out_channels=config.emb_sizes,
            kernel_size=1,
        )
        self.classifier = Classifier(
            input_size=config.emb_sizes, 
            lin_blocks=0, 
            lin_neurons=config.emb_sizes, 
            out_neurons=self.config.num_labels
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
        input_features = input_features.transpose(1, 2)
        if self.training:
            spec_augment = SpecAugment(
                n_time_masks=self.config.num_time_masks, 
                time_mask_param=self.config.time_mask_width, 
                n_freq_masks=self.config.num_freq_masks, 
                freq_mask_param=self.config.freq_mask_width, 
                zero_masking=True
            )
            # Input tensor should be (B, C, T) for SpecAugment
            input_features = spec_augment(input_features)
        lens = torch.ones(B).to(input_features.device)
        lens = torch.round(lens * T).int()
        # Input tensor should be (B, T, C) for Conformer
        features, masks = self.conformer(input_features.transpose(1, 2), lens)
        features = features.transpose(1, 2)
        features = self.asp(features)
        features = self.asp_bn(features)
        embeddings = self.fc(features)
        embeddings = embeddings.transpose(1, 2)
        logits = self.classifier(embeddings)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(B, 1, self.config.num_labels), labels.view(-1, 1))

        return ModelOutput(
            loss=loss,
            logits=logits.view(B, self.config.num_labels),
            embeddings=embeddings.view(B, self.config.emb_sizes)
        )
