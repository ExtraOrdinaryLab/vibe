#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mask utility functions for transformer and conformer models.

This module provides various mask generation functions used in both 
encoder and decoder for handling padding, causal relationships, and 
streaming processing.
"""

from typing import Tuple, Optional

import torch


def subsequent_mask(
    size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a causal (triangular) mask for autoregressive decoding.
    
    This mask is used in decoder which works in an auto-regressive mode,
    where the current step can only attend to previous (left) positions.
    
    In encoder, full attention is used when streaming is not necessary and
    the sequence is not long. In this case, no attention mask is needed.
    
    When streaming is needed, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.
    
    Args:
        size (int): Size of the square mask matrix
        device (torch.device, optional): Device for the mask tensor.
            Defaults to CPU.
    
    Returns:
        torch.Tensor: Boolean mask tensor of shape (size, size), where
            True values indicate allowed attention positions.
    
    Examples:
        >>> subsequent_mask(3)
        tensor([[True, False, False],
                [True, True, False],
                [True, True, True]])
    """
    # Create a square matrix filled with ones
    ret = torch.ones(size, size, device=device, dtype=torch.bool)
    # Convert to lower triangular matrix (including diagonal)
    return torch.tril(ret, out=ret)


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for chunk-based attention used in streaming encoder.
    
    This function generates a mask that restricts attention to a local chunk
    and a specified number of preceding chunks, enabling efficient streaming
    processing.
    
    Args:
        size (int): Size of the square mask matrix
        chunk_size (int): Size of each attention chunk
        num_left_chunks (int, optional): Number of left chunks to attend to
            - If < 0: use full context (attend to all left chunks)
            - If >= 0: attend to exactly this many left chunks
            Defaults to -1 (full context).
        device (torch.device, optional): Device for the mask tensor.
            Defaults to CPU.
    
    Returns:
        torch.Tensor: Boolean mask tensor of shape (size, size), where
            True values indicate allowed attention positions.
    
    Examples:
        >>> subsequent_chunk_mask(4, 2)
        tensor([[True, True, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True]])
    """
    # Initialize with zeros (all positions masked)
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    
    # For each position, determine which other positions it can attend to
    for i in range(size):
        # Determine current chunk and valid attention range
        current_chunk = i // chunk_size
        
        # Calculate start position based on num_left_chunks
        if num_left_chunks < 0:
            # Attend to all previous chunks
            start = 0
        else:
            # Attend to limited number of previous chunks
            first_accessible_chunk = max(0, current_chunk - num_left_chunks)
            start = first_accessible_chunk * chunk_size
        
        # End position is the end of the current chunk
        ending = min((current_chunk + 1) * chunk_size, size)
        
        # Set attention mask for current position
        ret[i, start:ending] = True
    
    return ret


def add_optional_chunk_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int
) -> torch.Tensor:
    """Apply optional chunk-based mask for encoder.
    
    This function selects and applies the appropriate mask based on configuration:
    - Dynamic chunk mask with random or fixed chunk size
    - Static chunk mask with fixed chunk size
    - No chunking (full context mask)
    
    Args:
        xs (torch.Tensor): Padded input sequence, shape (B, L, D)
        masks (torch.Tensor): Base attention mask, shape (B, L, L)
        use_dynamic_chunk (bool): Whether to use dynamic chunk sizes
        use_dynamic_left_chunk (bool): Whether to use dynamic left chunks for training
        decoding_chunk_size (int): Chunk size for dynamic chunking:
            - 0: Training mode, use random dynamic chunk size
            - < 0: Decoding mode, use full context
            - > 0: Decoding mode, use fixed chunk size
        static_chunk_size (int): Chunk size for static chunking
            (ignored if use_dynamic_chunk is True)
        num_decoding_left_chunks (int): Number of left chunks to attend to:
            - >= 0: Use specific number of left chunks
            - < 0: Use all left chunks
    
    Returns:
        torch.Tensor: Chunked attention mask, shape (B, L, L)
    """
    # Case 1: Dynamic chunking
    if use_dynamic_chunk:
        max_len = xs.size(1)
        
        # Determine chunk size and number of left chunks based on mode
        if decoding_chunk_size < 0:
            # Full context mode
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            # Fixed chunk size mode (for decoding)
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # Random chunk size mode (for training)
            # Select either a random chunk size in [1, 25] or full context
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_left_chunks = -1
            
            if chunk_size > max_len // 2:
                # Use full context if random size is large
                chunk_size = max_len
            else:
                # Limit chunk size to range [1, 25]
                # Since we use 4x subsampling and allow up to 1s (100 frames)
                # delay, the maximum frame is 100 / 4 = 25
                chunk_size = chunk_size % 25 + 1
                
                # Optionally use random number of left chunks
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1,)).item()
        
        # Create and apply chunk mask
        chunk_masks = subsequent_chunk_mask(
            xs.size(1), chunk_size, num_left_chunks, xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    
    # Case 2: Static chunking
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(
            xs.size(1), static_chunk_size, num_left_chunks, xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    
    # Case 3: No chunking (use original mask)
    else:
        chunk_masks = masks
    
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create a boolean mask for padding regions in variable-length sequences.
    
    This function returns a mask where padded positions are marked as True.
    
    Args:
        lengths (torch.Tensor): Batch of sequence lengths, shape (batch_size,)
    
    Returns:
        torch.Tensor: Boolean mask where True indicates padded positions,
            shape (batch_size, max_length)
    
    Examples:
        >>> lengths = torch.tensor([5, 3, 2])
        >>> make_pad_mask(lengths)
        tensor([[False, False, False, False, False],
                [False, False, False, True, True],
                [False, False, True, True, True]])
    """
    batch_size = int(lengths.size(0))
    max_len = int(lengths.max().item())
    
    # Create a range tensor [0, 1, 2, ..., max_len-1]
    seq_range = torch.arange(
        0, max_len, dtype=torch.int64, device=lengths.device)
    
    # Expand to shape (batch_size, max_len)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    
    # Expand lengths to same shape, but with length value repeated for each position
    seq_length_expand = lengths.unsqueeze(-1)
    
    # Create mask: True for padded positions (where position index >= sequence length)
    mask = seq_range_expand >= seq_length_expand
    
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create a boolean mask for non-padded regions in variable-length sequences.
    
    The sequences in a batch may have different lengths. To enable batch
    processing, padding is needed to make all sequences the same size.
    To avoid the padding affecting operations like attention or convolution,
    these padded positions are masked.
    
    This function returns a mask where non-padded positions are marked as True.
    
    Args:
        lengths (torch.Tensor): Batch of sequence lengths, shape (batch_size,)
    
    Returns:
        torch.Tensor: Boolean mask where True indicates valid (non-padded) positions,
            shape (batch_size, max_length)
    
    Examples:
        >>> lengths = torch.tensor([5, 3, 2])
        >>> make_non_pad_mask(lengths)
        tensor([[True, True, True, True, True],
                [True, True, True, False, False],
                [True, True, False, False, False]])
    """
    # Simply invert the pad mask
    return ~make_pad_mask(lengths)


def mask_finished_scores(
    score: torch.Tensor,
    flag: torch.Tensor
) -> torch.Tensor:
    """Mask scores for finished sequences in beam search.
    
    When a sequence is finished in beam search, we only allow one alive branch.
    This function gives one branch a zero score and the rest -inf score.
    
    Args:
        score (torch.Tensor): Score tensor with shape (batch_size * beam_size, beam_size)
        flag (torch.Tensor): Boolean tensor indicating finished sequences,
            shape (batch_size * beam_size, 1)
    
    Returns:
        torch.Tensor: Masked score tensor with same shape as input
    """
    beam_size = score.size(-1)
    
    # Create a mask of zeros with same shape as flag
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    
    if beam_size > 1:
        # For beam size > 1, create masks for finished and unfinished branches
        # unfinished: First position is 0, rest are from flag
        unfinished = torch.cat(
            (zero_mask, flag.repeat([1, beam_size - 1])), dim=1)
        
        # finished: First position is from flag, rest are 0
        finished = torch.cat(
            (flag, zero_mask.repeat([1, beam_size - 1])), dim=1)
    else:
        # For beam size = 1, masks are simpler
        unfinished = zero_mask
        finished = flag
    
    # Apply masks: -inf for unfinished paths, 0 for the single finished path
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    
    return score


def mask_finished_preds(
    pred: torch.Tensor,
    flag: torch.Tensor,
    eos: int
) -> torch.Tensor:
    """Replace predictions for finished sequences with EOS token.
    
    When a sequence is finished in beam search, all of its branches
    should predict the EOS token.
    
    Args:
        pred (torch.Tensor): Prediction tensor with shape (batch_size * beam_size, beam_size)
        flag (torch.Tensor): Boolean tensor indicating finished sequences,
            shape (batch_size * beam_size, 1)
        eos (int): End-of-sequence token ID
    
    Returns:
        torch.Tensor: Modified prediction tensor where finished sequences
            have been replaced with EOS token
    """
    beam_size = pred.size(-1)
    
    # Expand finished flags to cover all beam positions
    finished = flag.repeat([1, beam_size])
    
    # Replace predictions at finished positions with EOS token
    return pred.masked_fill_(finished, eos)