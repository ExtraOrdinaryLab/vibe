import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginLoss(nn.Module):
    """
    Implementation of Additive Angular Margin Loss (ArcFace).
    
    This loss function enhances the discriminative power of the feature embeddings by adding 
    an angular margin penalty to the target logit.
    
    Args:
        scale (float): The scaling factor for cosine values.
        margin (float): The angular margin penalty in radians.
        easy_margin (bool): If True, use a simplified version for better convergence.
    
    References:
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        https://arxiv.org/abs/1801.07698
    """
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False
    ) -> None:
        super(ArcMarginLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ArcMarginLoss.
        
        Args:
            cosine: Cosine similarity matrix of shape [batch_size, num_classes]
            label: Ground truth labels of shape [batch_size]
            
        Returns:
            The computed loss value
        """
        # Calculate sine values using the Pythagorean identity
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Apply angular margin projection
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Conditional selection based on margin strategy
        if self.easy_margin:
            # Use simplified version for better convergence
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Use original ArcFace formulation
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        # Create one-hot encoded labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        
        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale the outputs
        output *= self.scale

        # Calculate cross entropy loss
        loss = self.criterion(output, label)
        return loss

    def update(self, margin: float = 0.2) -> None:
        """
        Update the margin-related parameters.
        
        Args:
            margin: Angular margin penalty in radians
        """
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)


class AddMarginLoss(nn.Module):
    """
    Implementation of Additive Margin Loss (CosFace).
    
    This loss function enhances feature discrimination by adding a cosine margin
    penalty to the target logit.
    
    Args:
        scale (float): The scaling factor for cosine values.
        margin (float): The cosine margin penalty.
        easy_margin (bool): Not used in this implementation, kept for API consistency.
        
    References:
        CosFace: Large Margin Cosine Loss for Deep Face Recognition
        https://arxiv.org/abs/1801.09414
    """
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False
    ) -> None:
        super(AddMarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of AddMarginLoss.
        
        Args:
            cosine: Cosine similarity matrix of shape [batch_size, num_classes]
            label: Ground truth labels of shape [batch_size]
            
        Returns:
            The computed loss value
        """
        # Apply additive margin to cosine similarity
        phi = cosine - self.margin
        
        # Create one-hot encoded labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        
        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale the outputs
        output *= self.scale

        # Calculate cross entropy loss
        loss = self.criterion(output, label)
        return loss

    def update(self, margin: float = 0.2) -> None:
        """
        Update the margin parameter.
        
        Args:
            margin: Cosine margin penalty
        """
        self.margin = margin