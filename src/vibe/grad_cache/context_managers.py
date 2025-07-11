"""
Context managers for gradient caching.

This module provides context managers for maintaining random states
during forward and backward passes.
"""
from typing import Any

import torch
from torch import Tensor
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """
    Context manager for maintaining random states.
    
    Saves and restores random states to ensure reproducibility
    between forward and backward passes.
    """
    
    def __init__(self, *tensors: Tensor):
        """
        Initialize random context with tensors.
        
        Args:
            *tensors: Tensors used to capture device states
        """
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)
        self._fork = None

    def __enter__(self) -> None:
        """
        Enter the context, saving current random states.
        """
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the context, restoring previous random states.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None