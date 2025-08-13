#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Transformer and Conformer encoder layer implementations."""

from typing import Optional, Tuple

import torch
from torch import nn


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder layer module.
    
    This layer implements the encoder described in "Attention Is All You Need"
    paper, consisting of self-attention and feed-forward networks with residual
    connections and layer normalization.
    
    Args:
        size (int): Input dimension (d_model).
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate applied to residual connections.
        normalize_before (bool): Order of layer normalization.
            If True, use layer_norm before each sub-block (Pre-LN).
            If False, use layer_norm after each sub-block (Post-LN).
        concat_after (bool): Whether to concat attention layer's input and output.
            If True: x -> x + linear(concat(x, att(x)))
            If False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Initialize the TransformerEncoderLayer."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        
        # Layer normalization with small epsilon for numerical stability
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features through the transformer encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the self-attention (#batch, time, time).
            pos_emb (torch.Tensor): Positional embedding tensor. Included for 
                interface compatibility with ConformerEncoderLayer but not used.
            mask_pad (torch.Tensor): Batch padding mask. Not used in transformer layer,
                included for API compatibility with conformer.
            output_cache (torch.Tensor, optional): Cache tensor of the output
                (#batch, time2, size), where time2 < time in x. Used for streaming.
            cnn_cache (torch.Tensor, optional): Not used here, included for interface
                compatibility with ConformerEncoderLayer.
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Output tensor (#batch, time, size).
                - Mask tensor (#batch, time, time).
                - Dummy CNN cache tensor.
        """
        # Self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        # Handle incremental decoding with caching
        if output_cache is None:
            x_q = x
        else:
            # Process only the newly added part
            assert output_cache.size(0) == x.size(0), "Batch size mismatch in cache"
            assert output_cache.size(2) == self.size, "Feature dimension mismatch in cache"
            assert output_cache.size(1) < x.size(1), "Cache size should be smaller than input length"
            
            # Calculate the chunk of new frames to process
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        # Apply self-attention with residual connection
        x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # Feed-forward module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
            
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        # Combine with cache for incremental processing
        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        # Return dummy CNN cache for API compatibility
        fake_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        return x, mask, fake_cnn_cache


class ConformerEncoderLayer(nn.Module):
    """Conformer Encoder layer module.
    
    This implements the Conformer architecture as described in
    "Conformer: Convolution-augmented Transformer for Speech Recognition"
    
    The layer consists of:
    1. Macaron-style feed-forward module (optional)
    2. Multi-headed self-attention module
    3. Convolution module (optional)
    4. Feed-forward module
    
    Args:
        size (int): Input dimension (d_model).
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module, optional): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module, optional): Additional feed-forward module
            instance for Macaron-style architecture.
        conv_module (torch.nn.Module, optional): Convolution module instance.
            `ConvolutionModule` instance can be used as the argument.
        dropout_rate (float, optional): Dropout rate. Default: 0.1.
        normalize_before (bool, optional): Order of layer normalization.
            If True, use layer_norm before each sub-block (Pre-LN).
            If False, use layer_norm after each sub-block (Post-LN).
            Default: True.
        concat_after (bool, optional): Whether to concat attention layer's input and output.
            If True: x -> x + linear(concat(x, att(x)))
            If False: x -> x + att(x)
            Default: False.
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
        concat_after: bool = False,
    ):
        """Initialize the ConformerEncoderLayer."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        
        # Layer normalization modules for different components
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        
        # Macaron-style FNN uses a 0.5 scale factor for both feed-forward modules
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5  # Scale factor for FF in Macaron-net
        else:
            self.ff_scale = 1.0  # Regular scale for single FF
            
        # Convolution-specific layers
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(size, eps=1e-12)  # for the final output
            
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features through the conformer encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for self-attention (#batch, time, time).
            pos_emb (torch.Tensor): Positional embedding tensor, required for
                relative positional encoding in self-attention.
            mask_pad (torch.Tensor, optional): Batch padding mask used for convolution module
                (#batch, 1, time).
            output_cache (torch.Tensor, optional): Cache tensor of the output
                (#batch, time2, size), where time2 < time in x. Used for streaming.
            cnn_cache (torch.Tensor, optional): Convolution cache for streaming inference.
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Output tensor (#batch, time, size).
                - Mask tensor (#batch, time, time).
                - New CNN cache tensor.
        """
        # 1. Macaron-style feed-forward module (if enabled)
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
                
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
                
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # 2. Multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        # Handle incremental decoding with caching
        if output_cache is None:
            x_q = x
        else:
            # Process only the newly added part
            assert output_cache.size(0) == x.size(0), "Batch size mismatch in cache"
            assert output_cache.size(2) == self.size, "Feature dimension mismatch in cache"
            assert output_cache.size(1) < x.size(1), "Cache size should be smaller than input length"
            
            # Calculate the chunk of new frames to process
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        # Apply self-attention with residual connection
        x_att, _ = self.self_attn(x_q, x, x, mask, pos_emb)
        x = residual + self.dropout(x_att)
        
        if not self.normalize_before:
            x = self.norm_mha(x)

        # 3. Convolution module (if enabled)
        # Initialize dummy CNN cache
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
                
            # Apply convolution module with caching for streaming
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # 4. Feed-forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        
        if not self.normalize_before:
            x = self.norm_ff(x)

        # Final layer normalization (only for Conformer)
        if self.conv_module is not None:
            x = self.norm_final(x)

        # Combine with cache for incremental processing
        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        return x, mask, new_cnn_cache