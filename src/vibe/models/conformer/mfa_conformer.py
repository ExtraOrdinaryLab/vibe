#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MFA Conformer implementation for speaker embedding extraction.

This module implements a Conformer-based architecture with Multi-head 
self-attention (MFA) mechanism and Attentive Statistics Pooling for 
speaker embedding extraction tasks.
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConformerEncoder


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int) -> List[int]:
    """Calculate the padding elements needed for convolution operations.
    
    Args:
        L_in: Input length
        stride: Stride value for convolution
        kernel_size: Size of the convolutional kernel
        dilation: Dilation factor for convolution
    
    Returns:
        List containing left and right padding values
    """
    if stride > 1:
        # For stride > 1, use symmetric padding based on kernel size
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]
    else:
        # For stride = 1, calculate padding to maintain input length
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


def length_to_mask(
    length: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a binary mask for variable-length sequences in a batch.
    
    Args:
        length: Tensor containing the length of each sequence (1D)
        max_len: Maximum length for the mask (default: max length in batch)
        dtype: Data type for the generated mask
        device: Device to place the mask on
    
    Returns:
        Binary mask tensor where 1 indicates valid positions
    
    Example:
        >>> length = torch.Tensor([1, 2, 3])
        >>> mask = length_to_mask(length)
        >>> mask
        tensor([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]])
    """
    assert len(length.shape) == 1, "Length tensor must be 1D"

    # Determine maximum sequence length if not provided
    if max_len is None:
        max_len = length.max().long().item()
    
    # Create mask using broadcasting
    # Each row will have ones up to the specified length, zeros afterward
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    # Set data type and device if specified
    if dtype is None:
        dtype = length.dtype
    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


class _Conv1d(nn.Module):
    """Base 1D convolution implementation with flexible padding options.
    
    This implementation supports various padding modes and flexible
    tensor dimension handling for different input formats.
    
    Args:
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        input_shape: Shape of the input tensor (optional)
        in_channels: Number of input channels (optional)
        stride: Stride factor for convolution
        dilation: Dilation factor for convolution
        padding: Padding mode ('same', 'valid', 'causal')
        groups: Grouped convolution factor
        bias: Whether to include bias parameters
        padding_mode: Mode for padding (e.g., 'reflect', 'zeros')
        skip_transpose: Whether to skip transposing input/output tensors
        weight_norm: Whether to apply weight normalization
        conv_init: Initialization method for convolution weights
        default_padding: Default padding value for nn.Conv1d
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        input_shape: Optional[Tuple[int, ...]] = None,
        in_channels: Optional[int] = None,
        stride: int = 1,
        dilation: int = 1,
        padding: str = "same",
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        skip_transpose: bool = False,
        weight_norm: bool = False,
        conv_init: Optional[str] = None,
        default_padding: Union[str, int] = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        # Validate input parameters
        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        # Determine number of input channels
        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        # Create PyTorch Conv1d layer
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

        # Initialize convolution weights if specified
        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal":
            nn.init.normal_(self.conv.weight, std=1e-6)

        # Apply weight normalization if requested
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the 1D convolution.
        
        Args:
            x: Input tensor of shape (batch, time, channel) or (batch, channel, time)
               depending on skip_transpose setting
        
        Returns:
            Convolved output tensor with same dimension arrangement as input
        """
        # Handle tensor dimension arrangement
        if not self.skip_transpose:
            x = x.transpose(1, -1)  # (B, T, C) -> (B, C, T)

        # Add channel dimension for 2D inputs if needed
        if self.unsqueeze:
            x = x.unsqueeze(1)  # Add channel dimension

        # Apply appropriate padding based on mode
        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )
        elif self.padding == "causal":
            # For causal convolution, pad only on the left
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))
        elif self.padding == "valid":
            # No padding for 'valid' mode
            pass
        else:
            raise ValueError(
                f"Padding must be 'same', 'valid' or 'causal'. Got {self.padding}"
            )

        # Apply convolution
        wx = self.conv(x)

        # Remove extra dimension if added earlier
        if self.unsqueeze:
            wx = wx.squeeze(1)

        # Restore original dimension arrangement if needed
        if not self.skip_transpose:
            wx = wx.transpose(1, -1)  # (B, C, T) -> (B, T, C)

        return wx

    def _manage_padding(
        self, x: torch.Tensor, kernel_size: int, dilation: int, stride: int
    ) -> torch.Tensor:
        """Apply padding to maintain input length after convolution.
        
        Args:
            x: Input tensor
            kernel_size: Size of the kernel
            dilation: Dilation factor
            stride: Stride value
        
        Returns:
            Padded tensor
        """
        # Calculate padding elements
        padding = get_padding_elem(
            self.in_channels, stride, kernel_size, dilation
        )
        
        # Apply padding with specified mode
        x = F.pad(x, padding, mode=self.padding_mode)
        
        return x

    def _check_input_shape(self, shape: Tuple[int, ...]) -> int:
        """Validate input shape and determine number of input channels.
        
        Args:
            shape: Tuple containing input tensor dimensions
        
        Returns:
            Number of input channels
        
        Raises:
            ValueError: If input shape is invalid or kernel size is even with non-valid padding
        """
        # Determine input channels based on input shape
        if len(shape) == 2:
            # For 2D input (batch, features), treat as single channel
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            # For (batch, channel, time) format
            in_channels = shape[1]
        elif len(shape) == 3:
            # For (batch, time, channel) format
            in_channels = shape[2]
        else:
            raise ValueError(
                f"conv1d expects 2d or 3d inputs. Got {len(shape)}D input"
            )

        # Validate kernel size (must be odd for 'same' and 'causal' padding)
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                f"The kernel size must be an odd number for padding='{self.padding}'. "
                f"Got {self.kernel_size}."
            )

        return in_channels

    def remove_weight_norm(self):
        """Remove weight normalization at inference time if applied during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


class _BatchNorm1d(nn.Module):
    """1D batch normalization with flexible dimension handling.
    
    Args:
        input_shape: Expected shape of input tensor (optional)
        input_size: Expected feature dimension size (optional)
        eps: Small constant for numerical stability
        momentum: Momentum factor for running statistics
        affine: Whether to use learnable affine parameters
        track_running_stats: Whether to track running statistics
        combine_batch_time: Whether to combine batch and time dimensions
        skip_transpose: Whether to skip transposing input/output
    """

    def __init__(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        input_size: Optional[int] = None,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        combine_batch_time: bool = False,
        skip_transpose: bool = False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        # Determine input size from shape if not provided directly
        if input_size is None and skip_transpose:
            input_size = input_shape[1]  # (B, C, T) format
        elif input_size is None:
            input_size = input_shape[-1]  # (B, T, C) format

        # Create the batch normalization layer
        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization to input tensor.
        
        Args:
            x: Input tensor of shape (batch, time, channels) or (batch, channels, time)
               depending on skip_transpose setting
        
        Returns:
            Normalized tensor with same dimension arrangement as input
        """
        # Store original shape for restoration later
        shape_or = x.shape
        
        # Handle different input formats
        if self.combine_batch_time:
            # Combine batch and time dimensions for certain input types
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )
        elif not self.skip_transpose:
            # Switch to (B, C, T) format if needed
            x = x.transpose(-1, 1)

        # Apply batch normalization
        x_n = self.norm(x)

        # Restore original format
        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


# Efficient implementations that skip unnecessary transpositions
class Conv1d(_Conv1d):
    """1D convolution optimized for (batch, channel, time) input format.
    
    This is a convenience wrapper around _Conv1d that defaults to
    skip_transpose=True for improved efficiency with (B, C, T) format tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    """1D batch normalization optimized for (batch, channel, time) input format.
    
    This is a convenience wrapper around _BatchNorm1d that defaults to
    skip_transpose=True for improved efficiency with (B, C, T) format tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class TDNNBlock(nn.Module):
    """Time Delay Neural Network block.
    
    TDNN applies 1D convolution over time dimension with specified dilation
    to effectively capture temporal context.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        dilation: Dilation factor for convolution
        activation: Activation function class
        groups: Grouped convolution factor
        dropout: Dropout probability for regularization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation: nn.Module = nn.ReLU,
        groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Create sequential processing pipeline:
        # 1. Dilated convolution
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        # 2. Activation function
        self.activation = activation()
        # 3. Batch normalization
        self.norm = BatchNorm1d(input_size=out_channels)
        # 4. Dropout for regularization
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through TDNN block.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Processed tensor after convolution, activation, normalization and dropout
        """
        # Apply operations in sequence
        return self.dropout(self.norm(self.activation(self.conv(x))))


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling for speaker embedding extraction.
    
    This pooling layer computes weighted means and standard deviations
    of frame-level features, with weights determined by a learned attention
    mechanism. It can use global utterance context to improve attention.
    
    Args:
        channels: Number of input channels
        attention_channels: Number of attention network channels
        global_context: Whether to use global context for attention
    """

    def __init__(
        self, 
        channels: int, 
        attention_channels: int = 128, 
        global_context: bool = True
    ):
        super().__init__()
        # Small constant for numerical stability
        self.eps = 1e-12
        self.global_context = global_context
        
        # Attention network components
        if global_context:
            # With global context, input has 3x channels (original + global mean + global std)
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            # Without global context, use only local features
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
            
        self.tanh = nn.Tanh()
        # Final 1x1 convolution to produce attention weights
        self.conv = Conv1d(
            in_channels=attention_channels, 
            out_channels=channels, 
            kernel_size=1
        )

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted statistics pooling.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            lengths: Optional tensor of relative lengths for variable-length inputs
        
        Returns:
            Pooled statistics tensor of shape (batch, channels*2, 1) containing
            concatenated weighted means and standard deviations
        """
        # Extract sequence length
        L = x.shape[-1]

        # Helper function to compute weighted statistics
        def _compute_statistics(x, m, dim=2, eps=self.eps):
            # Weighted mean
            mean = (m * x).sum(dim)
            # Weighted standard deviation with numerical stability
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        # Default to equal length if not provided
        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Create binary mask for valid positions: shape [batch, 1, time]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Process with or without global context
        if self.global_context:
            # Compute global statistics for context
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            
            # Expand global statistics to all time steps
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            
            # Concatenate local features with global context
            attn = torch.cat([x, mean, std], dim=1)
        else:
            # Use only local features
            attn = x

        # Apply attention network
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Mask out padding positions
        attn = attn.masked_fill(mask == 0, float("-inf"))
        
        # Normalize attention weights with softmax
        attn = F.softmax(attn, dim=2)
        
        # Compute weighted statistics using attention weights
        mean, std = _compute_statistics(x, attn)
        
        # Concatenate mean and std to form output embedding
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class Conformer(nn.Module):
    """Conformer-based speaker embedding model.
    
    This model uses a Conformer encoder followed by attentive statistics pooling
    to extract fixed-dimensional speaker embeddings from variable-length inputs.
    
    Args:
        num_mel_bins: Number of input mel-spectrogram features
        num_blocks: Number of Conformer encoder blocks
        output_size: Dimension of Conformer encoder output
        embedding_size: Dimension of final speaker embedding
        input_layer: Type of input layer for the Conformer
        attention_channels: Number of channels in attention mechanism
        pos_enc_layer_type: Type of positional encoding to use
    """

    def __init__(
        self, 
        num_mel_bins: int = 80, 
        num_blocks: int = 6, 
        output_size: int = 256, 
        embedding_size: int = 192, 
        input_layer: str = "conv2d2", 
        pos_enc_layer_type: str = "rel_pos"
    ):
        super(Conformer, self).__init__()        
        # Conformer encoder processes frame-level features
        self.conformer = ConformerEncoder(
            input_size=num_mel_bins, 
            num_blocks=num_blocks, 
            output_size=output_size, 
            input_layer=input_layer, 
            pos_enc_layer_type=pos_enc_layer_type
        )
        
        # Attentive statistics pooling aggregates across time
        self.asp = AttentiveStatisticsPooling(
            output_size * num_blocks, 
            attention_channels=128, 
            global_context=True
        )
        
        # Batch normalization on pooled statistics
        self.asp_bn = BatchNorm1d(input_size=output_size * num_blocks * 2)
        
        # Final embedding projection
        self.fc = Conv1d(
            in_channels=output_size * num_blocks * 2,
            out_channels=embedding_size,
            kernel_size=1,
        )

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract speaker embeddings from input features.
        
        Args:
            x: Input tensor of shape (batch, time, channels)
            lengths: Optional tensor of sequence lengths for masked processing
            
        Returns:
            Speaker embedding vectors of shape (batch, embedding_size)
        """
        # Get batch and sequence dimensions
        B, T, C = x.shape
        
        # Create length tensor if not provided (assume full length)
        lens = torch.ones(B).to(x.device)
        lens = torch.round(lens * T).int()
        
        # Process through Conformer encoder
        features, masks = self.conformer(x, lens)
        
        # Transpose to (B, C, T) for pooling
        features = features.transpose(1, 2)
        
        # Apply attentive statistics pooling
        features = self.asp(features)
        
        # Apply batch normalization
        features = self.asp_bn(features)
        
        # Project to final embedding dimension
        embeddings = self.fc(features)
        
        # Reshape to (B, embedding_size)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings.squeeze(1)
        
        return embeddings