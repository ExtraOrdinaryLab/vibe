"""
Extract speaker embeddings from audio files for evaluation purposes.
This script loads a trained model and processes audio files to generate speaker embeddings.
"""

import os
import sys
import argparse
from typing import List

import torch
import torchaudio
from tqdm import tqdm
from kaldiio import WriteHelper

from vibe.utils import build, get_logger, build_config, load_wav_scp


def parse_arguments():
    """Parse command line arguments for the embedding extraction process."""
    parser = argparse.ArgumentParser(description='Extract embeddings for evaluation.')
    parser.add_argument('--exp_dir', default='', type=str, 
                        help='Experiment directory containing model and configuration')
    parser.add_argument('--audio_scp', default=None, nargs='+',
                        help='One or more data files containing audio paths. Multiple files can be specified.')
    parser.add_argument('--use_gpu', action='store_true', 
                        help='Use GPU for extraction if available')
    parser.add_argument('--gpu', nargs='+', default=None,
                        help='Specific GPU IDs to use. If not provided, all available GPUs will be used.')
    
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


def load_multiple_wav_scp(audio_scp_files: List[str]):
    """Load audio data from multiple wav.scp files.
    
    Args:
        audio_scp_files: List of paths to wav.scp files
        
    Returns:
        Dictionary mapping utterance IDs to audio file paths
    """
    combined_data = {}
    for scp_file in audio_scp_files:
        data = load_wav_scp(scp_file)
        # Check for duplicates
        duplicates = set(combined_data.keys()) & set(data.keys())
        if duplicates:
            print(f"Warning: Found {len(duplicates)} duplicate keys in {scp_file}. Later files will override earlier ones.")
        
        combined_data.update(data)
    
    return combined_data


def main():
    """Main function to extract speaker embeddings from audio files."""
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
    embedding_dir = os.path.join(args.exp_dir, 'embeddings')
    os.makedirs(embedding_dir, exist_ok=True)

    # Initialize logger
    logger = get_logger()

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

    # Load audio data from multiple files and distribute across workers
    data = load_multiple_wav_scp(args.audio_scp)
    data_keys = list(data.keys())
    local_keys = data_keys[rank::world_size]
    
    # Check if there's work for this worker
    if len(local_keys) == 0:
        msg = "The number of threads exceeds the number of files"
        logger.info(msg)
        sys.exit()

    # Set up output files
    emb_ark = os.path.join(embedding_dir, f'xvector_{rank:02d}.ark')
    emb_scp = os.path.join(embedding_dir, f'xvector_{rank:02d}.scp')

    # Log start of extraction (only on rank 0)
    if rank == 0:
        logger.info(f'Start extracting embeddings for {len(data_keys)} utterances from {len(args.audio_scp)} scp files.')
    
    # Extract embeddings using the model
    with torch.no_grad():
        with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
            for key in tqdm(local_keys):
                # Get audio file path and load it
                wav_path = data[key]
                wav, sample_rate = torchaudio.load(wav_path)
                
                # Verify sample rate matches expected rate
                assert sample_rate == config.sample_rate, (
                    f"The sample rate of wav is {sample_rate} and inconsistent "
                    f"with that of the pretrained model."
                )
                
                # Extract features and generate embedding
                feat = feature_extractor(wav)
                feat = feat.unsqueeze(0).to(device)
                emb = embedding_model(feat).detach().cpu().numpy()
                
                # Write embedding to output file
                writer(key, emb)


if __name__ == "__main__":
    main()