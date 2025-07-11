import random
from typing import Dict, List, Optional, Union

import torch
import numpy as np


def set_seed(seed: int = 1016) -> None:
    """
    Set random seeds for reproducibility across numpy, random, and PyTorch.
    
    Args:
        seed: Integer seed value (default: 1016)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    
    Useful for tracking metrics during training and evaluation.
    """
    def __init__(self, name: str, fmt: str = ':f'):
        """
        Initialize the average meter.
        
        Args:
            name: Name of the metric being tracked
            fmt: Format string for display (default: ':f' for float)
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.
        
        Args:
            val: New value to include in average
            n: Number of instances this value represents (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self) -> str:
        """Return formatted string with current value and average."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeters:
    """
    Container for multiple AverageMeter objects.
    
    Allows tracking and managing multiple metrics simultaneously.
    """
    def __init__(self, names: Optional[List[str]] = None, fmts: Optional[List[str]] = None):
        """
        Initialize with optional list of meters.
        
        Args:
            names: List of metric names
            fmts: List of format strings corresponding to each name
        """
        self.cont: Dict[str, AverageMeter] = {}
        
        if names is None or fmts is None:
            return
            
        for name, fmt in zip(names, fmts):
            self.cont[name] = AverageMeter(name, fmt)

    def add(self, name: str, fmt: str = ':f') -> None:
        """
        Add a new metric to track.
        
        Args:
            name: Name of the new metric
            fmt: Format string for display (default: ':f')
        """
        self.cont[name] = AverageMeter(name, fmt)

    def update(self, name: str, val: float, n: int = 1) -> None:
        """
        Update a specific metric.
        
        Args:
            name: Name of the metric to update
            val: New value to include in average
            n: Number of instances this value represents (default: 1)
        """
        self.cont[name].update(val, n)

    def avg(self, name: str) -> float:
        """
        Get the average value of a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Average value of the specified metric
        """
        return self.cont[name].avg

    def val(self, name: str) -> float:
        """
        Get the current value of a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Current value of the specified metric
        """
        return self.cont[name].val

    def __str__(self) -> str:
        """Return formatted string with all metrics."""
        return '\t'.join(str(meter) for meter in self.cont.values())


class ProgressMeter:
    """
    Display training progress with batch information and metrics.
    
    Useful for displaying training progress in a consistent format.
    """
    def __init__(self, num_batches: int, meters: AverageMeters, prefix: str = ""):
        """
        Initialize the progress meter.
        
        Args:
            num_batches: Total number of batches
            meters: AverageMeters instance containing metrics to display
            prefix: String to prepend to the output (default: "")
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> str:
        """
        Generate a string displaying the current progress.
        
        Args:
            batch: Current batch number
            
        Returns:
            Formatted string showing progress and metrics
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries.append(str(self.meters))
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """
        Create format string for batch display.
        
        Args:
            num_batches: Total number of batches
            
        Returns:
            Format string for batch display (e.g., "[10/100]")
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'