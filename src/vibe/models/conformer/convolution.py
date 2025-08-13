#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convolution module for Conformer architecture."""

from typing import Optional, Tuple

import torch
from torch import nn


class ConvolutionModule(nn.Module):
    """Convolution Module for Conformer model.
    
    This module implements the convolution block used in the Conformer architecture,
    which includes a pointwise convolution, a depthwise convolution, layer normalization,
    and another pointwise convolution with a Gated Linear Unit (GLU) activation.
    
    Args:
        channels (int): The number of input and output channels.
        kernel_size (int, optional): Kernel size of depthwise convolution layers. Default: 15.
        activation (nn.Module, optional): Activation function. Default: nn.ReLU().
        norm (str, optional): Type of normalization, either 'batch_norm' or 'layer_norm'. Default: "batch_norm".
        causal (bool, optional): Whether to use causal convolution. Default: False.
        bias (bool, optional): Whether to include bias in convolution layers. Default: True.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True
    ):
        """Initialize the ConvolutionModule."""
        super().__init__()

        # First pointwise convolution (expansion)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # Determine padding and left context size based on causality
        # self.lorder indicates the amount of left padding needed:
        # - If self.lorder > 0: This is a causal convolution, requiring left padding
        # - If self.lorder = 0: This is a symmetrical (non-causal) convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # For non-causal convolution, kernel size should be odd to maintain
            # the same input and output time dimension
            assert (kernel_size - 1) % 2 == 0, "Kernel size should be odd for non-causal convolution"
            padding = (kernel_size - 1) // 2
            self.lorder = 0
            
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,  # Depthwise convolution: one filter per channel
            bias=bias,
        )

        # Normalization layer
        assert norm in ['batch_norm', 'layer_norm'], "Normalization must be either 'batch_norm' or 'layer_norm'"
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        # Second pointwise convolution (projection)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        
        # Activation function
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input through the convolution module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, channels).
            mask_pad (torch.Tensor, optional): Padding mask of shape (batch, 1, time).
                1 indicates valid element, 0 indicates padding.
            cache (torch.Tensor, optional): Left context cache for causal convolution.
                Shape should be (batch, channels, left_context_length).
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch, time, channels)
                - New cache tensor for subsequent calls
        """
        # Transpose from (batch, time, channels) to (batch, channels, time)
        # This is required for PyTorch's Conv1d which expects channels first
        x = x.transpose(1, 2)  # (batch, channels, time)

        # Apply padding mask if provided
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)

        # Handle causal convolution and caching mechanism
        if self.lorder > 0:  # Causal convolution case
            if cache is None:
                # If no cache is provided, pad the left side with zeros
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                # Verify cache dimensions match the input
                assert cache.size(0) == x.size(0), "Cache batch size must match input batch size"
                assert cache.size(1) == x.size(1), "Cache channel dimension must match input channel dimension"
                
                # Concatenate cache with current input along time dimension
                x = torch.cat((cache, x), dim=2)
                
            # Ensure we have enough context after concatenation
            assert x.size(2) > self.lorder, "Input is too short for the specified kernel size"
            
            # Update cache for next forward pass
            new_cache = x[:, :, -self.lorder:]
        else:
            # For non-causal convolution, we don't need a cache
            # But return a dummy tensor for consistency (needed for JIT export)
            new_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)

        # Apply the first pointwise convolution (expansion)
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        
        # Apply Gated Linear Unit (GLU)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # Apply depthwise convolution
        x = self.depthwise_conv(x)
        
        # Apply normalization - layer norm requires (batch, time, channels)
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # (batch, time, channels)
            
        x = self.activation(self.norm(x))
        
        # Transpose back if needed
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # (batch, channels, time)
            
        # Apply the second pointwise convolution (projection)
        x = self.pointwise_conv2(x)
        
        # Apply padding mask again if provided
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)

        # Transpose back to (batch, time, channels) format
        x = x.transpose(1, 2)
        
        return x, new_cache