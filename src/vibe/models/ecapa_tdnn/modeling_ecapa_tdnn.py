import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def length_to_mask(length, max_len=None, dtype=None, device=None):
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
        max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
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
        out_channels,
        kernel_size,
        in_channels,
        stride=1,
        dilation=1,
        padding='same',
        groups=1,
        bias=True,
        padding_mode='reflect',
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

    def forward(self, x):
        """Forward pass with custom padding handling.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Convolved output tensor
        """
        if self.padding == 'same':
            x = self._manage_padding(x, self.kernel_size, self.dilation,
                                     self.stride)
        elif self.padding == 'causal':
            # Causal padding only pads the left side
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))
        elif self.padding == 'valid':
            # No padding needed for 'valid'
            pass
        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding)

        wx = self.conv(x)
        return wx

    def _manage_padding(
        self,
        x,
        kernel_size: int,
        dilation: int,
        stride: int,
    ):
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
        input_size,
        eps=1e-05,
        momentum=0.1,
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

    def forward(self, x):
        """Apply batch normalization to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        return self.norm(x)


class TDNNBlock(nn.Module):
    """Time Delay Neural Network block.
    
    Basic building block for TDNN (Time Delay Neural Network) architecture.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
    ):
        """Initialize TDNN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            dilation: Dilation factor for the convolution
            activation: Activation function class
            groups: Number of groups for grouped convolution
        """
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        """Forward pass through TDNN block.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(torch.nn.Module):
    """Res2Net block with dilation support.
    
    Implements multi-scale feature extraction as described in the Res2Net paper,
    with added support for dilated convolutions.
    """
    
    def __init__(
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1
    ):
        """Initialize Res2Net block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            scale: Number of scales for feature extraction
            kernel_size: Size of the convolution kernel
            dilation: Dilation factor for the convolution
        """
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0, "in_channels must be divisible by scale"
        assert out_channels % scale == 0, "out_channels must be divisible by scale"

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        # Create a TDNN block for each scale (except the first)
        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        """Forward pass with multi-scale hierarchical feature extraction.
        
        Args:
            x: Input tensor
            
        Returns:
            Multi-scale processed tensor
        """
        y = []
        # Split input into scale groups along channel dimension
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                # First chunk passes through unchanged
                y_i = x_i
            elif i == 1:
                # Second chunk processed by first TDNN block
                y_i = self.blocks[i - 1](x_i)
            else:
                # Subsequent chunks are processed by their TDNN block
                # with residual connection from previous output
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        
        # Concatenate all scale outputs
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.
    
    Implements channel recalibration mechanism that adaptively adjusts
    channel-wise feature responses by explicitly modeling interdependencies
    between channels.
    """
    
    def __init__(self, in_channels, se_channels, out_channels):
        """Initialize SE block.
        
        Args:
            in_channels: Number of input channels
            se_channels: Number of intermediate channels for dimensionality reduction
            out_channels: Number of output channels
        """
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        """Forward pass for squeeze and excitation.
        
        Args:
            x: Input tensor
            lengths: Optional tensor of sequence lengths for masked processing
            
        Returns:
            Channel recalibrated tensor
        """
        L = x.shape[-1]
        
        # Global average pooling with optional masking
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        # Two-layer bottleneck with ReLU and sigmoid activations
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        # Scale the input features
        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling layer.
    
    Uses a self-attention mechanism to compute weighted statistics (mean and standard deviation)
    of the input tensor, optionally with global context information.
    """
    
    def __init__(self, channels, attention_channels=128, global_context=True):
        """Initialize attentive statistics pooling.
        
        Args:
            channels: Number of input channels
            attention_channels: Number of channels in attention mechanism
            global_context: Whether to use global context information
        """
        super().__init__()

        self.eps = 1e-12  # Small value for numerical stability
        self.global_context = global_context
        
        # Input size to TDNN depends on whether we use global context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
            
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculate attentive statistics pooling.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            lengths: Optional tensor of sequence lengths for masked processing
            
        Returns:
            Tensor containing concatenated mean and standard deviation
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            """Helper function to compute mean and standard deviation with mask.
            
            Args:
                x: Input tensor
                m: Weight/mask tensor
                dim: Dimension along which to compute statistics
                eps: Small value for numerical stability
                
            Returns:
                Tuple of (mean, standard deviation)
            """
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        # Create mask from lengths if provided
        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)  # [N, 1, L]

        # Add global context if enabled
        if self.global_context:
            # Calculate mean and std for global context
            # Note: Using custom implementation instead of torch.std for
            # stable backward computation
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            
            # Expand mean and std to same time dimension as input
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            
            # Concatenate input with global statistics
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply attention layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Mask out padded regions with negative infinity
        attn = attn.masked_fill(mask == 0, float("-inf"))

        # Compute softmax attention weights
        attn = F.softmax(attn, dim=2)
        
        # Calculate weighted statistics
        mean, std = _compute_statistics(x, attn)
        
        # Concatenate mean and std for the final output
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """Building block for ECAPA-TDNN.
    
    Combines TDNN, Res2Net, and Squeeze-Excitation in a residual structure:
    TDNN -> Res2Net -> TDNN -> SE -> (+ residual)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
        groups=1,
    ):
        """Initialize SERes2Net block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            res2net_scale: Scale parameter for Res2Net block
            se_channels: Number of channels in SE block
            kernel_size: Kernel size for convolutions
            dilation: Dilation factor for convolutions
            activation: Activation function class
            groups: Number of groups for grouped convolution
        """
        super().__init__()
        self.out_channels = out_channels
        
        # First TDNN layer for dimensionality adjustment
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        
        # Multi-scale feature extraction
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        
        # Second TDNN layer
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        
        # Channel recalibration
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        # Optional shortcut for residual if dimensions don't match
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        """Forward pass through the entire block.
        
        Args:
            x: Input tensor
            lengths: Optional sequence lengths for masked processing
            
        Returns:
            Processed tensor with residual connection
        """
        # Preserve input for residual connection
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        # Process through main branch
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        # Add residual connection
        return x + residual


class ECAPA_TDNN(torch.nn.Module):
    """ECAPA-TDNN speaker embedding model.
    
    Implementation of the paper:
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    
    This architecture combines several techniques:
    - Multi-layer feature aggregation
    - Res2Net with dilated convolutions
    - Squeeze-and-Excitation blocks
    - Attentive Statistics Pooling
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
    ):
        """Initialize ECAPA-TDNN model.
        
        Args:
            input_size: Size of the input features
            device: Device to place the model on
            lin_neurons: Number of neurons in the final embedding layer
            activation: Activation function class
            channels: List of channel sizes for each block
            kernel_sizes: List of kernel sizes for each block
            dilations: List of dilation factors for each block
            attention_channels: Number of channels in attention mechanism
            res2net_scale: Scale parameter for Res2Net blocks
            se_channels: Number of channels in SE blocks
            global_context: Whether to use global context in pooling
            groups: List of group counts for each block
        """
        super().__init__()
        assert len(channels) == len(kernel_sizes), "channels and kernel_sizes must have same length"
        assert len(channels) == len(dilations), "channels and dilations must have same length"
        
        self.channels = channels
        self.blocks = nn.ModuleList()

        # Initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

        # Add SE-Res2Net blocks
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        
        # Batch normalization for pooled features
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation to embedding dimension
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """Extract speaker embeddings from input features.
        
        Args:
            x: Input tensor of shape (batch, time, channel)
            lengths: Optional tensor of sequence lengths for masked processing
            
        Returns:
            Speaker embedding vectors
        """
        # Transpose from (batch, time, channel) to (batch, channel, time)
        x = x.transpose(1, 2)

        # Store outputs from each block for multi-layer aggregation
        xl = []
        for layer in self.blocks:
            try:
                # Some layers support lengths parameter
                x = layer(x, lengths=lengths)
            except TypeError:
                # Others don't
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation (concatenate all block outputs except first)
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Apply attentive statistical pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final embedding transformation
        x = self.fc(x)

        # Convert back to expected output format
        x = x.transpose(1, 2)
        x = x.squeeze(1)

        return x