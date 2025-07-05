import math
import logging

import numpy as np
from rich.console import Console

import torch
import torch.nn as nn

from ..logging import get_logger

logger = get_logger()


def clenshaw_curtis_chebyshev_coefficients(func, degree=30, num_samples=1000, margin=0.2):
    """
    Computes Chebyshev coefficients using Clenshaw-Curtis nodes.
    The nodes are given by: x_j = cos(pi * j / (num_samples-1)) for j = 0,...,num_samples-1.
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


class ChebyshevAdditiveAngularMargin(nn.Module):
    """
    An implementation of Additive Angular Margin using Chebyshev approximation.
    
    This version uses Chebyshev polynomials to approximate cos(arccos(x) + margin),
    which provides more stable gradients compared to the original implementation.
    
    Arguments
    ---------
    margin : float
        The margin for cosine similarity.
    scale : float
        The scale for cosine similarity.
    easy_margin : bool
        Whether to use an easier version of the margin.
    chebyshev_degree : int
        Degree of Chebyshev polynomial. Higher degrees provide better approximation.
    num_samples : int
        Number of samples to compute Chebyshev coefficients.
    """

    def __init__(self, margin=0.2, scale=30.0, easy_margin=False, 
                 chebyshev_degree=30, num_samples=1000):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.eps = 1e-7
        
        # For easy_margin thresholding
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        
        # Target function for Chebyshev approximation
        def target_func(u, margin):
            """Compute f(u) = cos(arccos(u) + margin)"""
            u_clipped = np.clip(u, -1, 1)
            return np.cos(np.arccos(u_clipped) + margin)
        
        # Precompute Chebyshev coefficients
        cheb_coeffs_np = clenshaw_curtis_chebyshev_coefficients(
            target_func, degree=chebyshev_degree, num_samples=num_samples, margin=margin
        )
        self.register_buffer('coefficients', torch.from_numpy(cheb_coeffs_np).float())
        
        logger.log(f"Initialized ChebyshevAdditiveAngularMargin with margin={margin}, scale={scale}")

    def chebyshev_eval(self, x):
        """Evaluate the Chebyshev approximation using Clenshaw recurrence"""
        return ChebyshevClenshawFunction.apply(x, self.coefficients)

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors using Chebyshev approximation

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        # Ensure numerical stability
        cosine = outputs.float()
        cosine = torch.clamp(cosine, -1 + self.eps, 1 - self.eps)
        
        # Compute cos(arccos(cosine) + margin) using Chebyshev approximation
        phi = self.chebyshev_eval(cosine)
        
        # Handle easy_margin case
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # Apply margin to target logits
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        
        # Apply scaling
        return self.scale * outputs