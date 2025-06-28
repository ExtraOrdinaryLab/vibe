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