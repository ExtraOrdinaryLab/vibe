import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm


def sha256_hash(data):
    """
    Generate SHA256 hash for the given data.
    
    Args:
        data: String or bytes to be hashed
        
    Returns:
        Hexadecimal string representation of the hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    sha256_hash_object = hashlib.sha256(data)
    return sha256_hash_object.hexdigest()


def fast_scandir(path: str, extensions: List[str], recursive: bool = False):
    """
    Scan directory for files with specific extensions faster than glob.
    
    Args:
        path: Directory path to scan
        extensions: List of file extensions to include (with dot, e.g., ['.wav'])
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        Tuple of (list of subdirectories, list of matching files)
    """
    # Scan files recursively faster than glob
    subfolders, files = [], []

    try:  # Try to avoid 'permission denied' errors
        for f in os.scandir(path):
            try:  # Try to avoid 'too many levels of symbolic links' errors
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in extensions:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, extensions, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)

    return subfolders, files


def main():
    """
    Main function to create SCP (script) files mapping hashed IDs to audio files.
    Parses command line arguments and processes audio files.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Create SCP files that map SHA256 hashes to audio file paths'
    )
    parser.add_argument(
        '--audio-dir', 
        type=str, 
        default='/home/jovyan/corpus/audio/voxceleb1',
        help='Directory containing audio files to process'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='/home/jovyan/workspace/vibe/manifests/voxceleb1.scp',
        help='Output SCP file path'
    )
    parser.add_argument(
        '--extensions', 
        type=str, 
        nargs='+', 
        default=['.wav'],
        help='Audio file extensions to include (default: .wav)'
    )
    parser.add_argument(
        '--recursive', 
        action='store_true', 
        default=True,
        help='Scan directories recursively (default: True)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Scan for audio files
    print(f"Scanning for audio files in {args.audio_dir}...")
    _, audio_files = fast_scandir(
        args.audio_dir, 
        extensions=args.extensions, 
        recursive=args.recursive
    )
    print(f"Found {len(audio_files)} audio files")
    
    # Write SCP file
    print(f"Writing SCP file to {args.output}...")
    with open(args.output, 'w') as f:
        for audio_file in tqdm(audio_files):
            # Each line has format: "<hash> <filepath>"
            f.write(f"{sha256_hash(audio_file)} {audio_file}\n")
    
    print("Done!")


if __name__ == '__main__':
    main()