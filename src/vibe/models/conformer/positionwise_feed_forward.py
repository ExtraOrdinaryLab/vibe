#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positionwise feed forward layer definition for transformer-based models."""

from typing import Optional

import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.
    
    This implements the position-wise feed-forward network described in
    "Attention Is All You Need" paper. The feed-forward network is applied 
    independently to each position in the input sequence, consisting of two 
    linear transformations with a non-linearity in between.
    
    Args:
        idim (int): Input dimension.
        hidden_units (int): The number of hidden units (dimensionality of inner layer).
        dropout_rate (float): Dropout rate applied between the two linear layers.
        activation (nn.Module, optional): Activation function used between the two linear 
            transformations. Default: nn.ReLU().
    """
    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: Optional[nn.Module] = None,
    ):
        """Initialize the PositionwiseFeedForward module."""
        super().__init__()
        
        # Default activation is ReLU if none provided
        if activation is None:
            activation = nn.ReLU()
            
        # First linear transformation (idim -> hidden_units)
        self.w_1 = nn.Linear(idim, hidden_units)
        
        # Activation function
        self.activation = activation
        
        # Dropout applied after the first transformation and activation
        self.dropout = nn.Dropout(dropout_rate)
        
        # Second linear transformation (hidden_units -> idim)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Process input through the position-wise feed-forward network.
        
        The computation follows: FFN(x) = W₂(Dropout(Activation(W₁(x))))
        
        Args:
            xs (torch.Tensor): Input tensor of shape (batch_size, sequence_length, idim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, idim)
        """
        # Apply the first linear transformation
        hidden = self.w_1(xs)
        
        # Apply activation function
        hidden = self.activation(hidden)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Apply the second linear transformation
        output = self.w_2(hidden)
        
        return output