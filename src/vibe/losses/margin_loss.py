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
        pythagorean_identity (bool): If True, use Pythagorean identity to calculate phi.
        pos_squash_k (int): Insert a monotone squashing transformation only on the positive branch.
    
    References:
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        https://arxiv.org/abs/1801.07698
    """
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False, 
        pythagorean_identity: bool = False, 
        pos_squash_k: float = 1
    ) -> None:
        super(ArcMarginLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.pythagorean_identity = pythagorean_identity
        self.pos_squash_k = pos_squash_k
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
        # NEW: squash the positive cosine before margin
        if self.pos_squash_k > 1:
            cosine_y = cosine[torch.arange(cosine.size(0)), label]
            cosine_y = 1.0 - (1.0 - cosine_y).pow(self.pos_squash_k)
            # replace the original positives with this sharper version
            cosine = cosine.clone()
            cosine[torch.arange(cosine.size(0)), label] = cosine_y

        if self.pythagorean_identity:
            # Calculate sine values using the Pythagorean identity
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            # Apply angular margin projection
            phi = cosine * self.cos_m - sine * self.sin_m
        else:
            phi = torch.cos(torch.arccos(cosine) + self.margin)
        
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


class ArcFaceLoss(ArcMarginLoss):

    def __init__(self, scale = 30, margin = 0.2, easy_margin = False, pythagorean_identity = False, pos_squash_k = 1):
        super().__init__(scale, margin, easy_margin, pythagorean_identity, pos_squash_k)


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


class CosFaceLoss(AddMarginLoss):

    def __init__(self, scale = 30, margin = 0.2):
        super().__init__(scale, margin)


class SphereFaceLoss(nn.Module):
    """
    Implementation of Angular Softmax Loss (SphereFace).

    This loss function introduces a multiplicative angular margin `m` to the target logit,
    which makes the decision boundary on a hypersphere more discriminative.

    Args:
        scale (float): The scaling factor for cosine values.
        margin (float): The multiplicative angular margin `m`. Must be >= 1.

    References:
        SphereFace: Deep Hypersphere Embedding for Face Recognition
        https://arxiv.org/abs/1704.08063
    """
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 4.0
    ) -> None:
        super(SphereFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, cosine: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SphereFaceLoss.

        Args:
            cosine: Cosine similarity matrix of shape [batch_size, num_classes]
            label: Ground truth labels of shape [batch_size]

        Returns:
            The computed loss value
        """
        with torch.no_grad():
            m_theta = torch.acos(cosine)
            m_theta.scatter_(
                1, label.view(-1, 1), self.margin, reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - cosine

        logits = self.scale * (cosine + d_theta)
        loss = F.cross_entropy(logits, label)
        return loss

    def update(self, margin: float = 4.0) -> None:
        """
        Update the margin parameter.

        Args:
            margin: Multiplicative angular margin `m`
        """
        self.margin = margin