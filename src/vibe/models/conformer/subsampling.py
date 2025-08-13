#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subsampling layer implementations for transformer-based models."""

from typing import Tuple

import torch
from torch import nn


class BaseSubsampling(nn.Module):
    """Base class for subsampling layers.
    
    This serves as the foundation for all subsampling implementations,
    defining common attributes and methods.
    """
    def __init__(self):
        """Initialize the base subsampling module."""
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """Get position encoding.
        
        Args:
            offset (int): Position offset.
            size (int): Size of position encoding.
            
        Returns:
            torch.Tensor: Position encoding.
        """
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling.
    
    This module applies a linear transformation to the input without
    reducing the sequence length, followed by layer normalization and dropout.
    
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (nn.Module): Positional encoding module.
    """
    def __init__(
        self, 
        idim: int, 
        odim: int, 
        dropout_rate: float,
        pos_enc_class: nn.Module
    ):
        """Initialize the LinearNoSubsampling module."""
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-12),
            nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input through linear transformation.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int, optional): Position offset. Default: 0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Transformed tensor (#batch, time, odim).
                - Positional embedding tensor.
                - Unchanged mask tensor (#batch, 1, time).
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling2(BaseSubsampling):
    """Convolutional 2D subsampling that reduces sequence length to half.
    
    This module uses a single 2D convolution with stride 2 to reduce
    the sequence length by a factor of 2.
    
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate (not used in this module).
        pos_enc_class (nn.Module): Positional encoding module.
    """
    def __init__(
        self, 
        idim: int, 
        odim: int, 
        dropout_rate: float,
        pos_enc_class: nn.Module
    ):
        """Initialize the Conv2dSubsampling2 module."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (idim // 2 - 1), odim)
        )
        self.pos_enc = pos_enc_class
        
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 2 = (3 - 1) * 1
        self.right_context = 2

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample input tensor by a factor of 2.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int, optional): Position offset. Default: 0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Subsampled tensor (#batch, time//2, odim).
                - Positional embedding tensor.
                - Subsampled mask tensor (#batch, 1, time//2).
        """
        x = x.unsqueeze(1)  # (batch, channel=1, time, freq)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        
        # Subsample mask: remove last 2 frames and take every other frame
        return x, pos_emb, x_mask[:, :, :-2:2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling that reduces sequence length to 1/4.
    
    This module uses two 2D convolutions with stride 2 to reduce
    the sequence length by a factor of 4.
    
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate (not used in this module).
        pos_enc_class (nn.Module): Positional encoding module.
    """
    def __init__(
        self, 
        idim: int, 
        odim: int, 
        dropout_rate: float,
        pos_enc_class: nn.Module
    ):
        """Initialize the Conv2dSubsampling4 module."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        )
        self.pos_enc = pos_enc_class
        
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample input tensor by a factor of 4.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int, optional): Position offset. Default: 0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Subsampled tensor (#batch, time//4, odim).
                - Positional embedding tensor.
                - Subsampled mask tensor (#batch, 1, time//4).
        """
        x = x.unsqueeze(1)  # (batch, channel=1, time, freq)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        
        # Subsample mask: apply the same subsampling operation twice
        # Each operation removes last 2 frames and takes every other frame
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling that reduces sequence length to 1/6.
    
    This module uses a 2D convolution with stride 2 followed by
    another with stride 3 to reduce the sequence length by a factor of 6.
    
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate (not used in this module).
        pos_enc_class (nn.Module): Positional encoding module.
    """
    def __init__(
        self, 
        idim: int, 
        odim: int, 
        dropout_rate: float,
        pos_enc_class: nn.Module
    ):
        """Initialize the Conv2dSubsampling6 module."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 5, 3),
            nn.ReLU(),
        )
        self.linear = nn.Linear(
            odim * (((idim - 1) // 2 - 2) // 3),
            odim
        )
        self.pos_enc = pos_enc_class
        
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample input tensor by a factor of 6.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int, optional): Position offset. Default: 0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Subsampled tensor (#batch, time//6, odim).
                - Positional embedding tensor.
                - Subsampled mask tensor (#batch, 1, time//6).
        """
        x = x.unsqueeze(1)  # (batch, channel, time, freq)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        
        # Subsample mask: first by factor of 2, then by factor of 3
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling that reduces sequence length to 1/8.
    
    This module uses three 2D convolutions with stride 2 to reduce
    the sequence length by a factor of 8.
    
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate (not used in this module).
        pos_enc_class (nn.Module): Positional encoding module.
    """
    def __init__(
        self, 
        idim: int, 
        odim: int, 
        dropout_rate: float,
        pos_enc_class: nn.Module
    ):
        """Initialize the Conv2dSubsampling8 module."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), 
            odim
        )
        self.pos_enc = pos_enc_class
        
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample input tensor by a factor of 8.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int, optional): Position offset. Default: 0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Subsampled tensor (#batch, time//8, odim).
                - Positional embedding tensor.
                - Subsampled mask tensor (#batch, 1, time//8).
        """
        x = x.unsqueeze(1)  # (batch, channel, time, freq)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        
        # Subsample mask: apply the same subsampling operation three times
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]