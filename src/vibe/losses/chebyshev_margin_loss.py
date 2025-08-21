import math

import numpy as np
import torch
import torch.nn as nn


def clenshaw_curtis_chebyshev_coefficients(
    func, degree: int = 30, num_samples: int = 1000, margin: float = 0.2
):
    """
    Computes Chebyshev coefficients using Clenshaw-Curtis nodes.
    
    Args:
        func: The target function to approximate
        degree: Degree of the Chebyshev polynomial
        num_samples: Number of sample points to use
        margin: Margin parameter for the target function
        
    Returns:
        Coefficients for Chebyshev approximation
    """
    j = np.arange(num_samples)
    x = np.cos(np.pi * j / (num_samples - 1))
    y = func(x, margin)
    coeffs = np.polynomial.chebyshev.chebfit(x, y, degree)
    return coeffs


class ChebyshevClenshawFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, coeffs):
        """
        Evaluates the Chebyshev series using Clenshaw's recurrence:
        b_k = coeffs[k] + 2*x*b_{k+1} - b_{k+2}, and
        f(x) = b_0 - x * b_2,
        iterating from k = n down to 0.
        
        Args:
            x: Input tensor
            coeffs: Chebyshev coefficients
            
        Returns:
            The evaluated polynomial at x
        """
        n = coeffs.shape[0] - 1
        b_kplus1 = torch.zeros_like(x)
        b_kplus2 = torch.zeros_like(x)
        x2 = 2 * x
        for k in range(n, -1, -1):
            b_k = coeffs[k] + x2 * b_kplus1 - b_kplus2
            b_kplus2 = b_kplus1
            b_kplus1 = b_k
        result = b_k - b_kplus2 * x
        ctx.save_for_backward(x, coeffs)
        ctx.n = n
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Chebyshev evaluation.
        
        Uses the derivative expression:
        f'(x) = sum_{k=1}^{n} coeffs[k] * k * U_{k-1}(x)
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradients with respect to inputs
        """
        x, coeffs = ctx.saved_tensors
        n = ctx.n
        
        # Use the derivative expression:
        # f'(x) = sum_{k=1}^{n} coeffs[k] * k * U_{k-1}(x)
        U = []
        U0 = torch.ones_like(x)
        U.append(U0)
        if n >= 1:
            U1 = 2 * x
            U.append(U1)
            for k in range(2, n):
                U_k = 2 * x * U[k - 1] - U[k - 2]
                U.append(U_k)
        
        derivative = torch.zeros_like(x)
        for k in range(1, n + 1):
            derivative = derivative + coeffs[k] * k * U[k - 1]
        
        grad_input = grad_output * derivative
        return grad_input, None


class ChebyshevArcMarginLoss(nn.Module):
    """
    Implementation of Additive Angular Margin Loss (ArcFace) using Chebyshev approximation.
    
    This implementation uses Chebyshev polynomials to approximate cos(arccos(x) + margin),
    which provides more stable gradients compared to the original trigonometric implementation.
    
    Args:
        scale (float): The scaling factor for cosine values.
        margin (float): The angular margin penalty in radians.
        easy_margin (bool): If True, use a simplified version for better convergence.
        chebyshev_degree (int): Degree of Chebyshev polynomial. Higher degrees provide better approximation.
        num_samples (int): Number of samples to compute Chebyshev coefficients.
    
    References:
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        https://arxiv.org/abs/1801.07698
    """
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False,
        chebyshev_degree: int = 30,
        num_samples: int = 1000, 
        pos_squash_k: int = 1
    ) -> None:
        super(ChebyshevArcMarginLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()
        self.eps = 1e-7
        self.chebyshev_degree = chebyshev_degree
        self.num_samples = num_samples
        self.pos_squash_k = pos_squash_k
        
        self.update(margin)

    def forward(self, cosine: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ChebyshevArcMarginLoss.
        
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
        
        # Compute cos(arccos(cosine) + margin) using Chebyshev approximation
        phi = self.chebyshev_eval(cosine)
        
        # Handle easy_margin case
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
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
    
    def chebyshev_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Chebyshev approximation using Clenshaw recurrence.
        
        Args:
            x: Input tensor
            
        Returns:
            The evaluated polynomial at x
        """
        return ChebyshevClenshawFunction.apply(x, self.coefficients)

    def update(self, margin: float = 0.2) -> None:
        """
        Update the margin-related parameters and recompute Chebyshev coefficients.
        
        Args:
            margin: Angular margin penalty in radians
        """
        self.margin = margin
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        
        # Define target function for Chebyshev approximation
        def target_func(u, margin):
            """Compute f(u) = cos(arccos(u) + margin)"""
            u_clipped = np.clip(u, -1, 1)
            return np.cos(np.arccos(u_clipped) + margin)
        
        # Precompute Chebyshev coefficients
        cheb_coeffs_np = clenshaw_curtis_chebyshev_coefficients(
            target_func, 
            degree=self.chebyshev_degree, 
            num_samples=self.num_samples, 
            margin=margin
        )
        
        # Register coefficients as a buffer (persistent state)
        if not hasattr(self, 'coefficients'):
            self.register_buffer('coefficients', torch.from_numpy(cheb_coeffs_np).float())
        else:
            self.coefficients = torch.from_numpy(cheb_coeffs_np).float()