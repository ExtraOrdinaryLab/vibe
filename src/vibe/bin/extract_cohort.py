"""
Extract speaker embeddings from a cohort dataset (e.g., VoxCeleb2) for score normalization.
This script efficiently processes large cohort datasets using multiple GPUs when available.
"""

import os
import sys
import argparse
import random
from typing import List, Dict

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from kaldiio import WriteHelper

from vibe.utils import build, get_logger, build_config, load_wav_scp


def parse_arguments():
    """Parse command line arguments for the cohort embedding extraction process."""
    parser = argparse.ArgumentParser(description='Extract cohort embeddings for score normalization.')
    parser.add_argument('--exp_dir', default='', type=str, 
                        help='Experiment directory containing model and configuration')
    parser.add_argument('--cohort_scp', default=None, type=str,
                        help='SCP file containing cohort audio paths')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save cohort embeddings')
    parser.add_argument('--use_gpu', action='store_true', 
                        help='Use GPU for extraction if available')
    parser.add_argument('--gpu', nargs='+', default=None,
                        help='Specific GPU IDs to use. If not provided, all available GPUs will be used.')
    parser.add_argument('--max_samples', default=10000, type=int, 
                        help='Maximum number of cohort samples to process')
    parser.add_argument('--seed', default=1016, type=int, 
                        help='Random seed for sample selection')
    parser.add_argument('--max_frames', default=None, type=int, 
                        help='Max number of frames per utterance to process')
    
    return parser.parse_args(sys.argv[1:])


def setup_device(args, rank, logger):
    """Configure and return the appropriate computing device (GPU/CPU).
    
    Handles both single and multi-GPU scenarios, with automatic detection
    of available GPUs if specific IDs are not provided.
    """
    if args.use_gpu:
        if torch.cuda.is_available():
            # Get number of available GPUs
            available_gpus = torch.cuda.device_count()
            
            if available_gpus == 0:
                logger.warning("No CUDA devices are available despite CUDA being installed. Using CPU.")
                return torch.device('cpu')
            
            # If no specific GPUs are specified, use all available ones
            if args.gpu is None:
                gpu_ids = list(range(available_gpus))
                if rank == 0:
                    logger.info(f"No specific GPU IDs provided. Using all {available_gpus} available GPUs.")
            else:
                gpu_ids = [int(gpu_id) for gpu_id in args.gpu]
                if rank == 0:
                    logger.info(f"Using specified GPU IDs: {gpu_ids}")
            
            # Handle case where only one GPU is specified
            if len(gpu_ids) == 1:
                # When only one GPU is specified, always use that GPU regardless of rank
                gpu = gpu_ids[0]
                if rank == 0:
                    logger.info(f"Single GPU mode: using GPU {gpu} for all processes")
            else:
                # When multiple GPUs are specified, distribute work across them
                gpu = gpu_ids[rank % len(gpu_ids)]
                
            # Verify the GPU ID is valid
            if gpu >= available_gpus:
                logger.warning(f"Specified GPU ID {gpu} exceeds available GPUs {available_gpus-1}. Using GPU 0 instead.")
                gpu = 0
                
            device = torch.device('cuda', gpu)
            if rank == 0:
                logger.info(f"Process {rank} using device: {device}")
        else:
            msg = 'No CUDA device is detected. Using the CPU device.'
            if rank == 0:
                logger.warning(msg)
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        if rank == 0:
            logger.info("Using CPU as requested.")
    
    return device


def sample_cohort(data: Dict[str, str], max_samples: int, seed: int) -> Dict[str, str]:
    """
    Sample a subset of the cohort data.
    
    Args:
        data: Dictionary mapping utterance IDs to audio file paths
        max_samples: Maximum number of samples to select
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with sampled utterances
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # If data is smaller than max_samples, return all data
    if len(data) <= max_samples:
        return data
    
    # Sample keys randomly
    sampled_keys = random.sample(list(data.keys()), max_samples)
    return {k: data[k] for k in sampled_keys}


def main():
    """Main function to extract cohort embeddings for score normalization."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config_file = os.path.join(args.exp_dir, 'config.yaml')
    config = build_config(config_file)

    # Get distributed processing information
    # Default to single-process mode if environment variables aren't set
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    # Set up output directory for embeddings
    if args.output_dir is None:
        output_dir = os.path.join(args.exp_dir, 'cohort_embeddings')
    else:
        output_dir = args.output_dir
        
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    logger = get_logger()
    if rank == 0:
        logger.info(f"Extracting cohort embeddings for score normalization")
        logger.info(f"Cohort SCP file: {args.cohort_scp}")
        logger.info(f"Output directory: {output_dir}")

    # Set up computation device (GPU/CPU)
    device = setup_device(args, rank, logger)

    # Build and initialize the embedding model
    embedding_model = build('embedding_model', config)

    # Recover model parameters from last checkpoint
    config.checkpointer['args']['checkpoints_dir'] = os.path.join(args.exp_dir, 'models')
    config.checkpointer['args']['recoverables'] = {'embedding_model': embedding_model}
    checkpointer = build('checkpointer', config)
    checkpointer.recover_if_possible(epoch=config.num_epochs, device=device)

    # Prepare model for inference
    embedding_model.to(device)
    embedding_model.eval()
    
    # Initialize feature extractor
    feature_extractor = build('feature_extractor', config)

    # Load cohort data
    if rank == 0:
        logger.info(f"Loading cohort data from {args.cohort_scp}")
    cohort_data = load_wav_scp(args.cohort_scp)
    
    # Sample cohort data if needed
    if args.max_samples < len(cohort_data):
        if rank == 0:
            logger.info(f"Sampling {args.max_samples} utterances from {len(cohort_data)} total cohort utterances")
        # Use the same seed across all ranks to get the same sampling
        cohort_data = sample_cohort(cohort_data, args.max_samples, args.seed)
    
    # Distribute data across workers
    data_keys = list(cohort_data.keys())
    local_keys = data_keys[rank::world_size]
    
    if rank == 0:
        logger.info(f"Total cohort utterances after sampling: {len(cohort_data)}")
        logger.info(f"Processing {len(cohort_data)} cohort utterances with {world_size} workers")
    
    # Check if there's work for this worker
    if len(local_keys) == 0:
        msg = "The number of threads exceeds the number of files"
        logger.info(msg)
        sys.exit()

    # Set up output files
    emb_ark = os.path.join(output_dir, f'cohort_{rank:02d}.ark')
    emb_scp = os.path.join(output_dir, f'cohort_{rank:02d}.scp')

    # Extract embeddings using the model
    with torch.no_grad():
        with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
            for key in tqdm(local_keys, desc=f"Worker {rank} processing", disable=rank!=0):
                # Get audio file path and load it
                wav_path = cohort_data[key]
                try:
                    wav, sample_rate = torchaudio.load(wav_path)
                    
                    # Verify sample rate matches expected rate
                    assert sample_rate == config.sample_rate, (
                        f"The sample rate of wav is {sample_rate} and inconsistent "
                        f"with that of the pretrained model."
                    )
                    
                    # Extract features and generate embedding
                    feat = feature_extractor(wav)
                    if args.max_frames is not None:
                        feat = feat[:args.max_frames, :]
                    feat = feat.unsqueeze(0).to(device)
                    emb = embedding_model(feat).detach().cpu().numpy()
                    
                    # Write embedding to output file
                    writer(key, emb)
                except Exception as e:
                    if rank == 0 or world_size == 1:
                        logger.warning(f"Error processing {wav_path}: {str(e)}")


if __name__ == "__main__":
    main()