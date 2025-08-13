#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Positional encoding and embedding modules for Conformer architecture."""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope_utils import precompute_freqs_cis


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models.
    
    This class implements the sinusoidal positional encoding described in
    "Attention Is All You Need" paper. The encoding adds position information
    to the input embeddings.
    
    Args:
        d_model (int): Dimensionality of the embedding.
        dropout_rate (float): Dropout probability applied to the encoding.
        max_len (int, optional): Maximum sequence length. Default: 50000.
        reverse (bool, optional): Whether to reverse position indexing. Default: False.
    
    Formula:
        PE(pos, 2i)   = sin(pos/(10000^(2i/d_model)))
        PE(pos, 2i+1) = cos(pos/(10000^(2i/d_model)))
    """
    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 50000,
        reverse: bool = False
    ):
        """Initialize the PositionalEncoding module."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        # Create position encoding buffer
        self.pe = torch.zeros(self.max_len, self.d_model)
        
        # Generate position indices
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        
        # Calculate scaling factors for sine and cosine functions
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model)
        )
        
        # Apply sine to even indices
        self.pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension for broadcasting
        self.pe = self.pe.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features)
            offset (int, optional): Position offset for encoding. Default: 0.
                Useful for streaming inference.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Encoded tensor with positional information added.
                - Positional embedding tensor (returned for compatibility with RelPositionalEncoding).
        """
        # Ensure sequence length with offset doesn't exceed maximum length
        assert offset + x.size(1) < self.max_len, f"Sequence length {offset + x.size(1)} exceeds maximum length {self.max_len}"
        
        # Move positional encoding to the same device as input
        self.pe = self.pe.to(x.device)
        
        # Extract relevant positional encodings
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        
        # Scale input and add positional encoding
        x = x * self.xscale + pos_emb
        
        # Apply dropout to both outputs
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """Get positional encoding for a specific segment.
        
        This method is used for getting encoding in a streaming fashion.
        
        Important note:
            In non-streaming inference, dropout is applied once to the entire sequence.
            In streaming scenarios, this function may be called multiple times with
            increasing input sizes, causing dropout to be applied multiple times,
            which may affect the consistency of the output.
        
        Args:
            offset (int): Starting position offset.
            size (int): Required size of position encoding.
        
        Returns:
            torch.Tensor: Positional encoding for the specified segment.
        """
        assert offset + size < self.max_len, f"Requested encoding (offset={offset}, size={size}) exceeds maximum length {self.max_len}"
        return self.dropout(self.pe[:, offset:offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    
    This implements the relative positional encoding described in
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context".
    
    See: Appendix B in https://arxiv.org/abs/1901.02860
    
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int, optional): Maximum input length. Default: 100000.
    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 100000):
        """Initialize the RelPositionalEncoding module."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute relative positional encoding.
        
        Unlike standard positional encoding, this returns the positional embeddings
        separately without adding them to the input, allowing the attention mechanism
        to use relative position information.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).
            offset (int, optional): Position offset. Default: 0.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Scaled input tensor (without positional encoding added).
                - Positional embedding tensor that will be used by the attention mechanism.
        """
        assert offset + x.size(1) < self.max_len, f"Sequence length {offset + x.size(1)} exceeds maximum length {self.max_len}"
        
        # Move positional encoding to the same device as input
        self.pe = self.pe.to(x.device)
        
        # Scale input (without adding positional encoding)
        x = x * self.xscale
        
        # Extract relevant positional encodings
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(nn.Module):
    """Module that provides a dummy positional encoding implementation.
    
    This module can be used as a drop-in replacement when positional encoding
    is not desired, while maintaining API compatibility with other positional 
    encoding classes.
    
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate to apply to the input.
    """
    def __init__(self, d_model: int, dropout_rate: float):
        """Initialize the NoPositionalEncoding module."""
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input without adding positional information.
        
        This method maintains interface compatibility with other positional encoding classes
        but returns a zero tensor for positional embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).
            offset (int, optional): Ignored, present for API compatibility. Default: 0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Input tensor with dropout applied.
                - Zero tensor for positional embedding.
        """
        # Create a zero tensor with the right shape for positional embeddings
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        
        # Apply dropout to input without adding positional information
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """Return a zero tensor in place of positional encoding.
        
        Args:
            offset (int): Ignored, present for API compatibility.
            size (int): Size of the requested positional encoding.
            
        Returns:
            torch.Tensor: Zero tensor of shape (1, size, d_model).
        """
        return torch.zeros(1, size, self.d_model)


class RopePositionalEncoding(PositionalEncoding):
    """Rotary Position Embedding (RoPE) implementation.
    
    This class implements Rotary Position Embeddings as described in the paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    
    RoPE encodes absolute positional information with a rotation matrix
    that naturally incorporates explicit relative position dependency.
    
    Args:
        d_model (int): Model dimension.
        head_dim (int): Dimension of each attention head.
        dropout_rate (float): Dropout rate.
        max_len (int, optional): Maximum sequence length. Default: 3000.
        rope_theta (float, optional): Base value for frequency calculation. Default: 10000.0.
        scale (bool, optional): Whether to apply scaling to input. Default: True.
    
    References:
        - Paper: https://arxiv.org/abs/2104.09864
    """
    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 100000,
        head_dim: int = 64,
        rope_theta: float = 10000.0,
        scale: bool = True
    ):
        """Initialize the RopePositionalEncoding module."""
        super().__init__(d_model, dropout_rate=dropout_rate, max_len=max_len)
        # Remove the positional encoding from parent class as we'll use a different implementation
        delattr(self, 'pe')
        
        # Double max_len to accommodate bidirectional attention if needed
        self.max_len = max_len * 2
        
        # Precompute the frequency complex exponentials for RoPE
        pe = precompute_freqs_cis(head_dim, self.max_len, rope_theta)
        
        # Register the computed frequencies as a buffer (not a parameter)
        # Convert complex values to real representation for easier handling
        self.register_buffer("pe", torch.view_as_real(pe.unsqueeze(0)))
        
        self.dropout_rate = dropout_rate
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).
            offset (Union[int, torch.Tensor]): Position offset for encoding.
                Can be integer or tensor for batch-specific offsets.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Input tensor with optional scaling.
                - Positional embedding tensor in complex representation.
        """
        # Get position encoding with appropriate offset
        pos_emb = self.position_encoding(offset, x.size(1), True)
        
        # Add an extra dimension for compatibility with attention mechanism
        pos_emb = pos_emb.unsqueeze(2)  # [1, seq, 1, head_dim//2]
        
        # Apply scaling if configured
        if self.scale:
            x = x * self.xscale
            
        return self.dropout(x), pos_emb

    def position_encoding(
        self,
        offset: Union[int, torch.Tensor],
        size: int,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Get positional encoding for RoPE.
        
        This method supports both integer offsets for the entire batch
        and tensor offsets for batch-specific positions.
        
        Args:
            offset (Union[int, torch.Tensor]): Position offset(s).
            size (int): Length of sequence to encode.
            apply_dropout (bool, optional): Whether to apply dropout to encoding. Default: True.
        
        Returns:
            torch.Tensor: Complex tensor containing position encodings.
        """
        # Convert from real representation back to complex
        pe = torch.view_as_complex(self.pe)
        
        if isinstance(offset, int):
            # Single offset for entire batch
            assert offset + size <= self.max_len, f"Offset {offset} + size {size} exceeds max_len {self.max_len}"
            pos_emb = pe[:, offset:offset + size]
        else:
            # Batch-specific offsets
            assert torch.max(offset) + size <= self.max_len, f"Max offset {torch.max(offset).item()} + size {size} exceeds max_len {self.max_len}"
            
            # Create indices for each position in the sequence for each batch item
            index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)  # B X T
            
            # Create mask to handle negative offsets (set them to 0)
            flag = index > 0
            index = index * flag
            
            # Use embedding lookup to get position-specific encodings for each batch item
            pos_emb = F.embedding(index, pe[0])  # B X T X head_dim//2
            
        # Apply dropout if requested
        if apply_dropout:
            pos_emb = self.dropout_complex(pos_emb)
            
        return pos_emb

    def dropout_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to complex tensor.
        
        Since PyTorch dropout doesn't directly support complex numbers,
        we create a mask using the real part and apply it to the original tensor.
        
        Args:
            x (torch.Tensor): Complex tensor to apply dropout to.
        
        Returns:
            torch.Tensor: Complex tensor with dropout applied.
        """
        # Create dropout mask using real part of complex tensor
        mask = torch.nn.functional.dropout(
            torch.ones_like(x.real),
            training=self.training,
            p=self.dropout_rate,
        )
        # Apply the same mask to both real and imaginary parts
        return x * mask