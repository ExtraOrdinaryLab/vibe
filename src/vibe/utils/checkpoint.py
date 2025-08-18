"""
Checkpoint management module for saving and loading model states.
This is a simplified version of the checkpointer used in SpeechBrain.
"""

import os
import time
import logging
import pathlib
import warnings
import collections
from typing import Dict, Any, Optional, Union, List

import yaml
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "CKPT"
META_FILENAME = f"{CHECKPOINT_PREFIX}.yaml"
CHECKPOINT_EXT = ".ckpt"


def ckpt_recency(ckpt) -> float:
    """
    Helper function to get checkpoint recency based on timestamp.
    
    Args:
        ckpt: A Checkpoint namedtuple containing metadata
    
    Returns:
        float: Unix timestamp of checkpoint creation
    """
    return ckpt.meta["unixtime"]


def ckpt_epoch(ckpt) -> int:
    """
    Helper function to get checkpoint epoch.
    
    Args:
        ckpt: A Checkpoint namedtuple containing metadata
    
    Returns:
        int: Epoch number of the checkpoint, or -1 if not found
    """
    return ckpt.meta.get("epoch", -1)


# Define Checkpoint structure as a namedtuple for efficient storage
Checkpoint = collections.namedtuple(
    "Checkpoint", ["path", "meta", "paramfiles"]
)
# Creating a hash allows making checkpoint sets
Checkpoint.__hash__ = lambda self: hash(self.path)


class Checkpointer:
    """
    Handles saving and loading of model checkpoints.
    
    This implementation provides functionality to:
    - Save model states to disk
    - Load model states from disk
    - Automatically manage checkpoint versioning
    - Support custom checkpoint naming
    - Support weight averaging of model checkpoints
    """
    
    def __init__(
        self,
        checkpoints_dir: Union[str, pathlib.Path],
        recoverables: Dict[str, Any],
        allow_partial_load: bool = False,
    ):
        """
        Initialize the Checkpointer.
        
        Args:
            checkpoints_dir: Directory where checkpoints will be saved
            recoverables: Dictionary of objects to save/restore (name: object)
            allow_partial_load: If True, allows loading checkpoints that don't 
                               contain all recoverables
        """
        self.checkpoints_dir = pathlib.Path(checkpoints_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.recoverables = recoverables
        self.allow_partial_load = allow_partial_load

    def recover_if_possible(
        self,
        epoch: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Try to find and load a checkpoint if available.
        
        Args:
            epoch: If specified, load checkpoint from this specific epoch
            device: Device to load the checkpoint to (e.g., 'cpu', 'cuda')
        """
        # Collect all available checkpoints
        checkpoints = []
        for ckpt_dir in self._list_checkpoint_dirs():
            with open(ckpt_dir / META_FILENAME) as fi:
                meta = yaml.load(fi, Loader=yaml.Loader)
            paramfiles = {}
            for ckptfile in ckpt_dir.iterdir():
                if ckptfile.suffix == CHECKPOINT_EXT:
                    paramfiles[ckptfile.stem] = ckptfile
            checkpoints.append(Checkpoint(ckpt_dir, meta, paramfiles))

        if len(checkpoints) > 0:
            if epoch is None:
                # Load the most recent checkpoint
                logger.info("Loading the most recent checkpoint")
                checkpoints = sorted(checkpoints, key=ckpt_recency, reverse=True)
                chosen_ckpt = checkpoints[0]
                self.load_checkpoint(chosen_ckpt, device)
            else:
                # Load checkpoint from specific epoch
                is_found = False
                for chosen_ckpt in checkpoints:
                    if 'epoch' in chosen_ckpt.meta and chosen_ckpt.meta['epoch'] == epoch:
                        is_found = True
                        self.load_checkpoint(chosen_ckpt, device)
                        break
                if not is_found:
                    raise Exception(f"Checkpoint for epoch {epoch} not found, please check the {META_FILENAME} files.")
        else:
            logger.info("No checkpoints found to load.")

    def _list_checkpoint_dirs(self):
        """
        List all valid checkpoint directories.
        
        Returns:
            list: Paths to valid checkpoint directories
        """
        return [
            x
            for x in self.checkpoints_dir.iterdir()
            if Checkpointer._is_checkpoint_dir(x)
        ]

    def load_checkpoint(self, checkpoint: Checkpoint, device: Optional[str] = None):
        """
        Load a specific checkpoint into the recoverable objects.
        
        Args:
            checkpoint: Checkpoint namedtuple containing path and metadata
            device: Device to load the parameters to
        """
        logger.info(f"Loading checkpoint from {checkpoint.path}")
        for name, obj in self.recoverables.items():
            try:
                loadpath = checkpoint.paramfiles[name]
            except KeyError:
                if self.allow_partial_load:
                    MSG = f"Loading checkpoint from {checkpoint.path}, but missing a load path for {name}"
                    warnings.warn(MSG, UserWarning)
                    continue
                else:
                    MSG = f"Loading checkpoint from {checkpoint.path}, but missing a load path for {name}"
                    raise RuntimeError(MSG)

            # Handle different types of recoverable objects
            if hasattr(obj, 'load'):
                # Object has custom load method
                obj.load(loadpath, device)
            elif isinstance(obj, torch.nn.Module):
                # PyTorch module
                state = torch.load(loadpath, map_location=device)
                obj.load_state_dict(state)
            elif hasattr(obj, 'load_state_dict'):
                # Object with state dict
                state = torch.load(loadpath)
                obj.load_state_dict(state)
            else:
                MSG = f"Don't know how to load object of type {type(obj)}."
                raise RuntimeError(MSG)

    def save_checkpoint(
        self, 
        meta: Dict[str, Any] = {},
        name: Optional[str] = None,
        epoch: Optional[int] = None
    ):
        """
        Save the current state of all recoverables.
        
        Args:
            meta: Additional metadata to save with the checkpoint
            name: If provided, use this name for the checkpoint
            epoch: If provided, include epoch number in checkpoint path and metadata
        """
        # Determine checkpoint directory path
        if name is None:
            ckpt_dir = self._new_checkpoint_dirpath(epoch)
        else:
            ckpt_dir = self._custom_checkpoint_dirpath(name)
        
        # Create checkpoint directory
        os.makedirs(ckpt_dir)
        
        # Add epoch to metadata if provided
        if epoch is not None:
            meta.update({'epoch': int(epoch)})
        
        # Save metadata file
        self._save_checkpoint_metafile(
            ckpt_dir / META_FILENAME, meta,
        )
        
        # Save each recoverable object
        saved_paramfiles = {}
        for name, obj in self.recoverables.items():
            objfname = f"{name}{CHECKPOINT_EXT}"
            savepath = ckpt_dir / objfname
            saved_paramfiles[name] = savepath

            # Handle different types of saveable objects
            if hasattr(obj, 'save'):
                # Object has custom save method
                obj.save(savepath)
            elif isinstance(obj, torch.nn.Module):
                # PyTorch module
                state_dict = obj.state_dict()
                torch.save(state_dict, savepath)
            elif hasattr(obj, 'state_dict'):
                # Object with state dict
                state_dict = obj.state_dict()
                torch.save(state_dict, savepath)
            else:
                MSG = f"Don't know how to save object of type {type(obj)}."
                raise RuntimeError(MSG)
                
        logger.info(f"Saved checkpoint in {ckpt_dir}")

    def get_checkpoints_in_range(
        self, 
        start_epoch: int, 
        end_epoch: int
    ) -> List[Checkpoint]:
        """
        Get all checkpoints within the specified epoch range.
        
        Args:
            start_epoch: Starting epoch (inclusive)
            end_epoch: Ending epoch (inclusive)
            
        Returns:
            List of Checkpoint objects within the specified range
        """
        checkpoints = []
        for ckpt_dir in self._list_checkpoint_dirs():
            with open(ckpt_dir / META_FILENAME) as fi:
                meta = yaml.load(fi, Loader=yaml.Loader)
                
            # Check if checkpoint is within epoch range
            if 'epoch' in meta and start_epoch <= meta['epoch'] <= end_epoch:
                paramfiles = {}
                for ckptfile in ckpt_dir.iterdir():
                    if ckptfile.suffix == CHECKPOINT_EXT:
                        paramfiles[ckptfile.stem] = ckptfile
                checkpoints.append(Checkpoint(ckpt_dir, meta, paramfiles))
                
        # Sort by epoch
        checkpoints.sort(key=ckpt_epoch)
        return checkpoints

    def average_checkpoints(
        self, 
        start_epoch: int, 
        end_epoch: int, 
        device: str = "cuda",
        save_name: str = "weight_averaged"
    ) -> bool:
        """
        Average model weights from checkpoints in the specified epoch range.
        
        Args:
            start_epoch: Starting epoch (inclusive)
            end_epoch: Ending epoch (inclusive)
            device: Device to load checkpoints to
            save_name: Name for the saved averaged checkpoint
            
        Returns:
            bool: True if averaging was successful, False otherwise
        """
        checkpoints = self.get_checkpoints_in_range(start_epoch, end_epoch)
        
        if not checkpoints:
            logger.warning(f"No checkpoints found in range {start_epoch}-{end_epoch}")
            return False
            
        logger.info(f"Averaging {len(checkpoints)} checkpoints from epochs {start_epoch} to {end_epoch}")
        
        # Initialize averaged state dictionaries for each recoverable
        averaged_states = {}
        for name in self.recoverables:
            for ckpt in checkpoints:
                if name in ckpt.paramfiles:
                    # Load state dict from first checkpoint for this recoverable
                    state = torch.load(ckpt.paramfiles[name], map_location=device)
                    averaged_states[name] = state
                    break
                    
            if name not in averaged_states:
                logger.warning(f"Could not find checkpoint for {name} to initialize averaging")
                continue
        
        # Accumulate state dicts from remaining checkpoints
        for ckpt in checkpoints[1:]:
            for name in averaged_states:
                if name in ckpt.paramfiles:
                    state = torch.load(ckpt.paramfiles[name], map_location=device)
                    for key in averaged_states[name]:
                        if key in state:
                            averaged_states[name][key] += state[key]
        
        # Compute average by dividing by number of checkpoints
        for name in averaged_states:
            for key in averaged_states[name]:
                averaged_states[name][key] /= len(checkpoints)
        
        # Create directory for averaged checkpoint
        ckpt_dir = self._custom_checkpoint_dirpath(save_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save metadata
        meta = {
            "unixtime": time.time(),
            "averaged_epochs": {
                "start": start_epoch,
                "end": end_epoch,
                "count": len(checkpoints)
            }
        }
        self._save_checkpoint_metafile(ckpt_dir / META_FILENAME, meta)
        
        # Save averaged state dicts
        for name, state in averaged_states.items():
            save_path = ckpt_dir / f"{name}{CHECKPOINT_EXT}"
            torch.save(state, save_path)
            logger.info(f"Saved averaged {name} to {save_path}")
            
            # Load the averaged state into the recoverable
            if name in self.recoverables:
                obj = self.recoverables[name]
                if isinstance(obj, torch.nn.Module):
                    obj.load_state_dict(state)
                elif hasattr(obj, 'load_state_dict'):
                    obj.load_state_dict(state)
        
        logger.info(f"Weight averaging complete. Saved to {ckpt_dir}")
        return True

    def _new_checkpoint_dirpath(self, epoch: Optional[int] = None) -> pathlib.Path:
        """
        Generate a path for a new checkpoint directory with automatic versioning.
        
        Args:
            epoch: If provided, include epoch number in the directory name
            
        Returns:
            pathlib.Path: Path to the new checkpoint directory
        """
        if epoch is not None:
            stamp = f"EPOCH-{epoch}"
        else:
            # Use timestamp for checkpoint directory
            t = time.time()
            stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(t))
            
        # Add suffix number to avoid overwriting existing directories
        suffix_num = 0
        while (
            self.checkpoints_dir / f"{CHECKPOINT_PREFIX}-{stamp}-{suffix_num:02d}"
        ).exists():
            suffix_num += 1
            
        return self.checkpoints_dir / f"{CHECKPOINT_PREFIX}-{stamp}-{suffix_num:02d}"

    def _custom_checkpoint_dirpath(self, name: str) -> pathlib.Path:
        """
        Generate a path for a checkpoint directory with custom name.
        
        Args:
            name: Custom name for the checkpoint
            
        Returns:
            pathlib.Path: Path to the custom checkpoint directory
        """
        return self.checkpoints_dir / f"{CHECKPOINT_PREFIX}+{name}"

    def _save_checkpoint_metafile(
        self, fpath: pathlib.Path, meta_to_include: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Save checkpoint metadata to a YAML file.
        
        Args:
            fpath: Path to save the metadata file
            meta_to_include: Additional metadata to include
            
        Returns:
            dict: Complete metadata that was saved
        """
        meta = {"unixtime": time.time()}  # Always include timestamp
        meta.update(meta_to_include)
        
        with open(fpath, "w") as fo:
            fo.write(yaml.dump(meta))
            
        return meta

    @staticmethod
    def _is_checkpoint_dir(path: pathlib.Path) -> bool:
        """
        Check if a directory is a valid checkpoint directory.
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if directory is a valid checkpoint directory
        """
        path = pathlib.Path(path)
        
        # Must be a directory
        if not path.is_dir():
            return False
            
        # Must have correct prefix
        if not path.name.startswith(CHECKPOINT_PREFIX):
            return False
            
        # Must contain metadata file
        return (path / META_FILENAME).exists()