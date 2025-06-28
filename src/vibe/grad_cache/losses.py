"""
Loss functions for contrastive learning.

This module provides implementations of supervised and self-supervised
contrastive learning losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as torch_dist


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, reduction: str = 'mean'):
        """
        Calculate NTXentLoss loss
        
        Args:
            z_i: embeddings from view 1, shape [batch_size, embed_dim]
            z_j: embeddings from view 2, shape [batch_size, embed_dim]
            
        Returns:
            loss: NTXentLoss loss value
        """
        # Get batch size
        batch_size = z_i.shape[0]
        
        # Concatenate embeddings from both views
        features = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, embed_dim]
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels, labels], dim=0)  # [2*batch_size]
        
        # Create positive pair mask (where labels are equal)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [2*batch_size, 2*batch_size]
        
        # Normalize feature vectors
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # [2*batch_size, 2*batch_size]
        
        # Discard the main diagonal from both labels and similarity matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=z_i.device)
        labels = labels[~mask].view(labels.shape[0], -1)  # [2*batch_size, 2*batch_size-1]
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # [2*batch_size, 2*batch_size-1]
        
        # Select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [2*batch_size, num_positives]
        
        # Select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [2*batch_size, num_negatives]
        
        # Combine positives and negatives
        logits = torch.cat([positives, negatives], dim=1)  # [2*batch_size, num_positives+num_negatives]
        
        # Create labels for the contrastive prediction task (positives are at the beginning)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z_i.device)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        
        return loss


class DistributedNTXentLoss(NTXentLoss):

    def __init__(self, temperature=0.07):
        super().__init__(temperature=temperature)
        assert torch_dist.is_initialized(), "Distributed training has not been properly initialized."
        self.word_size = torch_dist.get_world_size()
        self.rank = torch_dist.get_rank()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, reduction: str = 'mean'):
        dist_z_i = self.gather_tensor(z_i)
        dist_z_j = self.gather_tensor(z_j)
        return super().forward(dist_z_i, dist_z_j, reduction=reduction)

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        torch_dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
    
    Modified to accept variable number of view embeddings (*z) instead of stacked features.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        Initialize the SupConLoss module.
        
        Args:
            temperature: Temperature scaling parameter for the contrastive loss
            contrast_mode: 'all' or 'one', determines which views to use as anchors
            base_temperature: Base temperature for loss scaling
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, *z_views, labels: torch.Tensor = None, mask: torch.Tensor = None):
        """
        Compute supervised contrastive loss with variable number of views.
        If labels is None, it becomes SimCLR unsupervised loss.
        
        Args:
            *z_views: Variable number of view embeddings, each of shape [batch_size, embed_dim]
            labels: Class labels of shape [batch_size]
            mask: Optional contrastive mask of shape [batch_size, batch_size]
                  mask_{i,j}=1 if sample j has the same class as sample i
        
        Returns:
            A loss scalar
        """
        # Check that we have at least one view
        if len(z_views) == 0:
            raise ValueError("At least one view must be provided")
            
        # Get device and batch size from the first view
        device = z_views[0].device
        batch_size = z_views[0].shape[0]
        
        # Check that all views have the same batch size and are on the same device
        for i, view in enumerate(z_views[1:], 1):
            if view.shape[0] != batch_size:
                raise ValueError(f"View {i} has batch size {view.shape[0]}, "
                               f"but first view has batch size {batch_size}")
            if view.device != device:
                raise ValueError(f"View {i} is on device {view.device}, "
                               f"but first view is on device {device}")
        
        # Stack all views to create features of shape [batch_size, n_views, embed_dim]
        features = torch.stack(z_views, dim=1)
        
        # Create mask based on labels or default to identity matrix (only self is positive)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Number of views
        contrast_count = features.shape[1]
        
        # Flatten the views dimension to get all embeddings in a single dimension
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size*n_views, embed_dim]
        
        # Select anchor features based on contrast mode
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # Use only the first view as anchors
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # Use all views as anchors
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown contrast mode: {self.contrast_mode}')

        # Compute similarity matrix (dot product) and apply temperature scaling
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # Numerical stability: subtract max for each row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Expand mask to account for multiple views
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Create mask to exclude self-contrastive cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Handle edge cases where there are no positive pairs for an anchor
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Compute final loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
