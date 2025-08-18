"""
Model weight averaging utilities.

This module provides functionality for averaging model weights across multiple checkpoints,
which can help improve model robustness and generalization.
"""

import os
import logging
import pathlib
from typing import List, Dict, Optional, Union, Tuple

import torch

logger = logging.getLogger(__name__)


def average_checkpoints(
    checkpoint_paths: List[Union[str, pathlib.Path]],
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Average model weights from multiple checkpoint files.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        device: Device to load the checkpoints to
        
    Returns:
        Dict containing averaged model state dict
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for averaging")
    
    # Load first checkpoint to initialize the average
    logger.info(f"Loading checkpoint 1/{len(checkpoint_paths)}: {checkpoint_paths[0]}")
    avg_state_dict = torch.load(checkpoint_paths[0], map_location=device)
    
    # Average with subsequent checkpoints
    for i, path in enumerate(checkpoint_paths[1:], 2):
        logger.info(f"Loading checkpoint {i}/{len(checkpoint_paths)}: {path}")
        state_dict = torch.load(path, map_location=device)
        
        # Accumulate each parameter
        for key in avg_state_dict:
            if key in state_dict:
                avg_state_dict[key] += state_dict[key]
    
    # Divide by number of checkpoints to get average
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / len(checkpoint_paths)
    
    logger.info(f"Successfully averaged {len(checkpoint_paths)} checkpoints")
    return avg_state_dict


def find_checkpoints_in_range(
    checkpoints_dir: Union[str, pathlib.Path],
    start_epoch: int,
    end_epoch: int,
    model_name: str
) -> List[pathlib.Path]:
    """
    Find checkpoint files in the specified epoch range.
    
    Args:
        checkpoints_dir: Directory containing checkpoints
        start_epoch: Starting epoch (inclusive)
        end_epoch: Ending epoch (inclusive)
        model_name: Name of the model component in the checkpoint
        
    Returns:
        List of paths to checkpoint files
    """
    checkpoints_dir = pathlib.Path(checkpoints_dir)
    checkpoint_paths = []
    
    # Look through all subdirectories that might contain epoch information
    for path in checkpoints_dir.glob("**/CKPT-EPOCH-*"):
        if not path.is_dir():
            continue
            
        # Extract epoch number from directory name
        try:
            dir_name = path.name
            parts = dir_name.split("-")
            if len(parts) >= 3:
                epoch = int(parts[2])
                if start_epoch <= epoch <= end_epoch:
                    # Check if model file exists
                    model_path = path / f"{model_name}.ckpt"
                    if model_path.exists():
                        checkpoint_paths.append(model_path)
        except (ValueError, IndexError):
            # Skip directories that don't match expected format
            continue
    
    # Sort by epoch number
    checkpoint_paths.sort(key=lambda p: int(p.parent.name.split("-")[2]))
    
    return checkpoint_paths


def perform_weight_averaging(
    model: torch.nn.Module,
    checkpoints_dir: Union[str, pathlib.Path],
    start_epoch: int,
    end_epoch: int,
    model_name: str,
    device: str = "cuda",
    save_path: Optional[Union[str, pathlib.Path]] = None,
    delete_checkpoints: bool = False
) -> None:
    """
    Perform weight averaging on model checkpoints.
    
    Args:
        model: Model to load the averaged weights into
        checkpoints_dir: Directory containing checkpoints
        start_epoch: Starting epoch (inclusive)
        end_epoch: Ending epoch (inclusive)
        model_name: Name of the model component in the checkpoint
        device: Device to load the checkpoints to
        save_path: Path to save the averaged model weights
        delete_checkpoints: Whether to delete the original checkpoints after averaging
    """
    # Find checkpoint files in the specified range
    checkpoint_paths = find_checkpoints_in_range(
        checkpoints_dir, start_epoch, end_epoch, model_name
    )
    
    if not checkpoint_paths:
        logger.warning(f"No checkpoints found in range {start_epoch}-{end_epoch}")
        return
    
    logger.info(f"Found {len(checkpoint_paths)} checkpoints for averaging")
    
    # Average checkpoints
    averaged_state_dict = average_checkpoints(checkpoint_paths, device)
    
    # Load averaged weights into model
    model.load_state_dict(averaged_state_dict)
    logger.info(f"Loaded weight-averaged model from epochs {start_epoch}-{end_epoch}")
    
    # Save averaged model if requested
    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(averaged_state_dict, save_path)
        logger.info(f"Saved weight-averaged model to {save_path}")
    
    # Delete original checkpoints if requested
    if delete_checkpoints and save_path:
        for path in checkpoint_paths:
            if path != save_path:  # Don't delete the averaged model
                os.remove(path)
                logger.info(f"Deleted original checkpoint: {path}")