import math
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def length_to_mask(
    length: torch.Tensor, 
    max_len: Optional[int] = None, 
    dtype: Optional[torch.dtype] = None, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert lengths to a binary mask representation.
    
    Args:
        length: Tensor containing sequence lengths
        max_len: Maximum length for the mask, defaults to max length in input
        dtype: Data type for the output mask
        device: Device for the output mask
        
    Returns:
        Binary mask tensor of shape (batch_size, max_len)
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()
    
    # Create mask where each sequence only has 1s up to its length
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def get_padding_elem(
    L_in: int, 
    stride: int, 
    kernel_size: int, 
    dilation: int
) -> List[int]:
    """Calculate required padding elements for 1D convolution.
    
    Args:
        L_in: Input length
        stride: Stride size
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        
    Returns:
        List containing padding values for left and right sides
    """
    if stride > 1:
        n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
        L_out = stride * (n_steps - 1) + kernel_size * dilation
        padding = [kernel_size // 2, kernel_size // 2]
    else:
        L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1
        padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
        
    return padding


class Conv1d(nn.Module):
    """Custom 1D convolution with flexible padding options.
    
    Supports 'same', 'causal', and 'valid' padding modes.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1,
        padding: str = 'same',
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'reflect',
    ):
        """Initialize Conv1d with custom padding.
        
        Args:
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            in_channels: Number of input channels
            stride: Stride of the convolution
            dilation: Dilation factor
            padding: Padding mode ('same', 'causal', or 'valid')
            groups: Number of groups for grouped convolution
            bias: Whether to include bias parameters
            padding_mode: Mode for padding ('reflect', 'replicate', etc.)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,  # We handle padding separately
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with custom padding handling.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Convolved output tensor
        """
        if self.padding == 'same':
            x = self._manage_padding(x, self.kernel_size, self.dilation, self.stride)
        elif self.padding == 'causal':
            # Causal padding only pads the left side
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))
        elif self.padding == 'valid':
            # No padding needed for 'valid'
            pass
        else:
            raise ValueError(
                f"Padding must be 'same', 'valid' or 'causal'. Got {self.padding}"
            )

        wx = self.conv(x)
        return wx

    def _manage_padding(
        self,
        x: torch.Tensor,
        kernel_size: int,
        dilation: int,
        stride: int,
    ) -> torch.Tensor:
        """Apply appropriate padding to ensure output size matches input size.
        
        Args:
            x: Input tensor
            kernel_size: Kernel size for the convolution
            dilation: Dilation factor
            stride: Stride size
            
        Returns:
            Padded tensor
        """
        L_in = x.shape[-1]
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)
        x = F.pad(x, padding, mode=self.padding_mode)
        return x


class BatchNorm1d(nn.Module):
    """Wrapper for PyTorch's BatchNorm1d with simplified interface."""
    
    def __init__(
        self,
        input_size: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
    ):
        """Initialize BatchNorm1d.
        
        Args:
            input_size: Size of the input features
            eps: Value added to denominator for numerical stability
            momentum: Value for running stats calculation
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        return self.norm(x)


class Linear(nn.Module):
    """Computes a linear transformation y = wx + b.

    Args:
        n_neurons: Number of output neurons (dimensionality of output)
        input_shape: Shape of the input tensor
        input_size: Size of the input tensor
        bias: If True, the additive bias b is adopted
        max_norm: Weight max-norm
        combine_dims: If True and the input is 4D, combine 3rd and 4th dimensions of input

    Example:
        >>> inputs = torch.rand(10, 50, 40)
        >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
        >>> output = lin_t(inputs)
        >>> output.shape
        torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons: int,
        input_shape: Optional[tuple] = None,
        input_size: Optional[int] = None,
        bias: bool = True,
        max_norm: Optional[float] = None,
        combine_dims: bool = False,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the linear transformation of input tensor.

        Args:
            x: Input tensor to transform linearly

        Returns:
            wx: The linearly transformed outputs
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        if self.max_norm is not None:
            self.w.weight.data = torch.renorm(
                self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        wx = self.w(x)
        return wx


class StatisticsPooling(nn.Module):
    """Statistical pooling layer that returns the mean and/or std of input tensor.

    Args:
        return_mean: If True, the average pooling will be returned
        return_std: If True, the standard deviation will be returned

    Example:
        >>> inp_tensor = torch.rand([5, 100, 50])
        >>> sp_layer = StatisticsPooling()
        >>> out_tensor = sp_layer(inp_tensor)
        >>> out_tensor.shape
        torch.Size([5, 1, 100])
    """

    def __init__(self, return_mean: bool = True, return_std: bool = True):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False\n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculates mean and std for a batch (input tensor).

        Args:
            x: Tensor for a mini-batch
            lengths: The lengths of the samples in the input

        Returns:
            pooled_stats: The mean and/or std for the input
        """
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                if self.return_mean:
                    mean.append(
                        torch.mean(x[snt_id, 0:actual_size, ...], dim=0)
                    )
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats

    def _get_gauss_noise(
        self, shape_of_tensor: torch.Size, device: str = "cpu"
    ) -> torch.Tensor:
        """Returns a tensor of epsilon Gaussian noise.

        Args:
            shape_of_tensor: Size of tensor for generating Gaussian noise
            device: Device on which to perform computations

        Returns:
            gnoise: The Gaussian noise tensor
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise


class Xvector(nn.Module):
    """X-vector extractor for speaker recognition and diarization.

    Args:
        in_channels: Expected size of input features (default=40)
        tdnn_blocks: Number of time-delay neural (TDNN) layers (default=5)
        tdnn_channels: Output channels for each TDNN layer
        tdnn_kernel_sizes: List of kernel sizes for each TDNN layer
        tdnn_dilations: List of dilations for kernels in each TDNN layer
        lin_neurons: Number of neurons in the final linear layer
        activation: Activation function class to use
        return_mean: Whether to return mean in statistical pooling
        return_std: Whether to return std in statistical pooling

    Example:
        >>> compute_xvect = Xvector(in_channels=40)
        >>> input_feats = torch.rand([5, 10, 40])
        >>> outputs = compute_xvect(input_feats)
        >>> outputs.shape
        torch.Size([5, 512])
    """

    def __init__(
        self,
        in_channels: int = 40,
        tdnn_blocks: int = 5,
        tdnn_channels: List[int] = [512, 512, 512, 512, 1500],
        tdnn_kernel_sizes: List[int] = [5, 3, 3, 1, 1],
        tdnn_dilations: List[int] = [1, 2, 3, 1, 1],
        lin_neurons: int = 512,
        activation: Type[nn.Module] = nn.LeakyReLU,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN layers
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend([
                Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=tdnn_kernel_sizes[block_index],
                    dilation=tdnn_dilations[block_index],
                ),
                activation(),
                BatchNorm1d(input_size=out_channels),
            ])
            in_channels = tdnn_channels[block_index]

        # Statistical pooling
        self.blocks.append(
            StatisticsPooling(return_mean=True, return_std=True)
        )

        # Final linear transformation
        self.fc = Linear(
            input_size=out_channels * 2,
            n_neurons=lin_neurons,
            bias=True,
            combine_dims=False,
        )

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extracts x-vectors from input features.

        Args:
            x: Input features for extracting x-vectors
            lengths: The corresponding relative lengths of the inputs

        Returns:
            x: Output x-vectors
        """
        for layer in self.blocks:
            x = layer(x, lengths) if isinstance(layer, StatisticsPooling) else layer(x)
        
        x = self.fc(x.squeeze(dim=1))
        return x