#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rotary Position Embedding (RoPE) utilities.

This module implements Rotary Position Embedding techniques for transformer models,
providing functions to compute and apply rotational position-dependent transformations
to attention module inputs. These embeddings help models understand token positions
through a rotational transformation in the complex plane.

References:
    - RoFormer: Enhanced Transformer with Rotary Position Embedding
      https://arxiv.org/abs/2104.09864
    - Implementation approaches derived from Google's Gemma and Meta's LLaMA models
"""

import torch
from typing import Tuple


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute frequency-based complex rotation factors for rotary embeddings.
    
    This function generates complex values for rotary position embeddings (RoPE)
    as proposed in the RoFormer paper. These values are used to encode positional
    information in transformer models through rotations in the complex plane.
    
    Args:
        dim: Feature dimension of the input, must be divisible by 2
        end: Maximum sequence length to precompute
        theta: Base value for frequency calculation, controls rotation rate
    
    Returns:
        Complex tensor of shape (end, dim//2) containing rotation factors
        in the complex plane for each position and feature pair
    
    Source:
        Adapted from Google's Gemma implementation:
        https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L84
    """
    # Calculate frequencies using logarithmically spaced values
    # Each pair of dimensions shares the same frequency
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    
    # Create position indices
    t = torch.arange(end, device=freqs.device)
    
    # Compute outer product to get frequencies for each position and feature pair
    freqs = torch.outer(t, freqs).float()
    
    # Convert to complex numbers with unit magnitude and phase determined by freqs
    # Using polar representation: e^(i*θ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    
    return freqs_cis


def google_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor using Google's Gemma approach.
    
    This implementation follows Google's method for applying rotary position
    embeddings, which uses a specific reshaping pattern to transform the input
    into complex values, apply rotations, and then convert back to real values.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
        freqs_cis: Complex tensor of rotation factors from precompute_freqs_cis
    
    Returns:
        Tensor with same shape as input but with rotary position embedding applied
    
    Source:
        Modified from Google's Gemma implementation:
        https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L95
    """
    # Convert input to complex numbers by splitting last dimension into real/imaginary parts
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1)
    )
    
    # Apply complex multiplication with rotation factors
    # This rotates each vector in the complex plane based on its position
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    
    # Reshape back to the original format
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    
    return x_out


def llama_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor using Meta's LLaMA approach.
    
    This implementation follows Meta's method for applying rotary position
    embeddings, which uses a different reshaping pattern compared to Google's
    approach but achieves the same mathematical rotation effect.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
        freqs_cis: Complex tensor of rotation factors from precompute_freqs_cis
    
    Returns:
        Tensor with same shape as input but with rotary position embedding applied
    
    Source:
        Based on Meta's LLaMA implementation approach
    """
    # Reshape to view last dimension as complex numbers
    # This creates a complex tensor by treating pairs of values as real/imaginary parts
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Apply complex multiplication for rotation
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    
    # Restore original data type
    return x_out.type_as(x)