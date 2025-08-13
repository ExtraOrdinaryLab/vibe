#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility functions for Transformer and Conformer models."""

import math
from typing import Tuple, List, Dict, Any, Optional, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# Special token ID for padding positions
IGNORE_ID = -1


def pad_list(xs: List[torch.Tensor], pad_value: Union[int, float]) -> torch.Tensor:
    """Perform padding for a list of tensors with variable lengths.
    
    This function takes tensors with different first dimensions (sequence lengths)
    and pads them to have the same first dimension, creating a batch.
    
    Args:
        xs (List[torch.Tensor]): List of tensors [(T_1, *), (T_2, *), ..., (T_B, *)],
            where T_i is the sequence length of the i-th sample.
        pad_value (Union[int, float]): Value used for padding.
    
    Returns:
        torch.Tensor: Padded tensor (B, Tmax, *), where B is batch size and
            Tmax is the maximum sequence length in the batch.
    
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    
    # Create padded tensor with the same dtype and device as input
    pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    
    # Copy each tensor into padded tensor
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    
    return pad


def add_sos_eos(
    ys_pad: torch.Tensor, 
    sos: int, 
    eos: int,
    ignore_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add start-of-sequence and end-of-sequence tokens to target sequences.
    
    Args:
        ys_pad (torch.Tensor): Batch of padded target sequences (B, Lmax)
        sos (int): Index of start-of-sequence token <sos>
        eos (int): Index of end-of-sequence token <eos>
        ignore_id (int): Index used for padding
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ys_in (torch.Tensor): Input sequences with <sos> token (B, Lmax + 1)
            - ys_out (torch.Tensor): Output sequences with <eos> token (B, Lmax + 1)
    
    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in, ys_out = add_sos_eos(ys_pad, sos_id, eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    # Create singleton tensors for sos and eos tokens
    _sos = torch.tensor(
        [sos],
        dtype=torch.long,
        requires_grad=False,
        device=ys_pad.device
    )
    _eos = torch.tensor(
        [eos],
        dtype=torch.long,
        requires_grad=False,
        device=ys_pad.device
    )
    
    # Remove padding from each sequence
    ys = [y[y != ignore_id] for y in ys_pad]
    
    # Add <sos> to the beginning of each sequence for decoder input
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    
    # Add <eos> to the end of each sequence for decoder output
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    
    # Pad sequences to the same length
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def reverse_pad_list(
    ys_pad: torch.Tensor,
    ys_lens: torch.Tensor,
    pad_value: float = -1.0
) -> torch.Tensor:
    """Reverse sequences and re-pad them.
    
    This function flips each sequence along the time dimension and pads them
    to the same length.
    
    Args:
        ys_pad (torch.Tensor): Padded tensor of shape (B, Tokenmax)
        ys_lens (torch.Tensor): Lengths of token sequences of shape (B)
        pad_value (float, optional): Value used for padding. Default: -1.0
    
    Returns:
        torch.Tensor: Reversed and padded tensor of shape (B, Tokenmax)
    
    Examples:
        >>> x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> lens = torch.tensor([4, 3, 2])
        >>> reverse_pad_list(x, lens, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])
    """
    # For each sequence: convert to int, slice to actual length, and reverse
    r_ys_pad = pad_sequence(
        [torch.flip(y.int()[:i], [0]) for y, i in zip(ys_pad, ys_lens)],
        batch_first=True,
        padding_value=pad_value
    )
    
    return r_ys_pad


def th_accuracy(
    pad_outputs: torch.Tensor, 
    pad_targets: torch.Tensor,
    ignore_label: int
) -> float:
    """Calculate classification accuracy ignoring padding positions.
    
    Args:
        pad_outputs (torch.Tensor): Prediction logits tensor (B * Lmax, D)
        pad_targets (torch.Tensor): Target label tensor (B, Lmax, D)
        ignore_label (int): Label index to ignore in accuracy calculation
    
    Returns:
        float: Accuracy value between 0.0 and 1.0
    """
    # Reshape predictions to match target shape and get the predicted class
    pad_pred = pad_outputs.view(
        pad_targets.size(0), 
        pad_targets.size(1),
        pad_outputs.size(1)
    ).argmax(2)
    
    # Create mask to ignore padding positions
    mask = pad_targets != ignore_label
    
    # Count correct predictions only at valid positions
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    
    return float(numerator) / float(denominator)


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x).
    
    This implementation follows the original paper:
    "Searching for Activation Functions" (Ramachandran et al., 2017)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation function.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after applying Swish activation
        """
        return x * torch.sigmoid(x)


def get_activation(act: str) -> nn.Module:
    """Get activation function module by name.
    
    Args:
        act (str): Name of the activation function
        
    Returns:
        nn.Module: Instantiated activation function module
        
    Raises:
        KeyError: If the specified activation name is not supported
    """
    # Dictionary mapping activation names to their PyTorch implementations
    activation_funcs = {
        "hardtanh": nn.Hardtanh,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "swish": Swish,
        "gelu": nn.GELU
    }
    
    if act not in activation_funcs:
        raise KeyError(f"Activation function '{act}' is not supported. "
                      f"Available options: {list(activation_funcs.keys())}")
        
    return activation_funcs[act]()


def get_subsample(config: Dict[str, Any]) -> int:
    """Get subsampling factor from model configuration.
    
    Args:
        config (Dict[str, Any]): Model configuration dictionary
        
    Returns:
        int: Subsampling factor (4, 6, or 8)
        
    Raises:
        AssertionError: If input_layer is not one of the supported types
    """
    input_layer = config["encoder_conf"]["input_layer"]
    assert input_layer in ["conv2d", "conv2d6", "conv2d8"], \
        f"input_layer should be conv2d, conv2d6 or conv2d8, got {input_layer}"
    
    if input_layer == "conv2d":
        return 4
    elif input_layer == "conv2d6":
        return 6
    elif input_layer == "conv2d8":
        return 8


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    """Remove blank tokens and consecutive duplicates from hypothesis.
    
    This function is used in CTC decoding to collapse repeated characters
    and remove blank tokens (assumed to be 0).
    
    Args:
        hyp (List[int]): Hypothesis sequence with possible duplicates and blanks
        
    Returns:
        List[int]: Cleaned hypothesis sequence
    """
    new_hyp: List[int] = []
    cur = 0
    
    while cur < len(hyp):
        # Skip blank tokens (token id 0)
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
            
        # Skip consecutive duplicates
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
            
    return new_hyp


def log_add(args: List[float]) -> float:
    """Stable log addition in log-space to prevent numerical overflow.
    
    Computes log(sum(exp(args))) in a numerically stable way.
    
    Args:
        args (List[float]): List of log-domain values
        
    Returns:
        float: Result of log(sum(exp(args)))
    """
    # Handle the case where all values are -infinity
    if all(a == -float('inf') for a in args):
        return -float('inf')
    
    # Use the log-sum-exp trick for numerical stability
    a_max = max(args)
    # log(sum(exp(args))) = log(sum(exp(args - a_max) * exp(a_max)))
    # = log(exp(a_max) * sum(exp(args - a_max)))
    # = a_max + log(sum(exp(args - a_max)))
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    
    return a_max + lsp