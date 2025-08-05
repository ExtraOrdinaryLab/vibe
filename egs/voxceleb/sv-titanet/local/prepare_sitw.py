#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to prepare SITW (Speakers in the Wild) dataset for speaker verification.
Converts FLAC files to WAV format and generates trial files.

https://github.com/ElsevierSoftwareX/SOFTX-D-20-00038
"""

import os
import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm


def convert_flac_to_wav(file_path: str) -> str:
    """
    Convert FLAC audio file to WAV format with 16kHz sampling rate.
    
    Args:
        file_path: Path to the FLAC file
        
    Returns:
        Path to the converted WAV file
    """
    src_path = Path(file_path)
    dst_path = src_path.with_suffix('.wav')
    
    # Skip conversion if WAV file already exists
    if dst_path.exists():
        return str(dst_path)
    
    # Ensure the output directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare FFmpeg command
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(src_path),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        str(dst_path)
    ]
    
    # Execute the command
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    return str(dst_path)


def prepare_sitw(sitw_folder: str, output_folder: str) -> None:
    """
    Prepare SITW dataset by generating trial files and converting audio to WAV format.
    
    Args:
        sitw_folder: Path to SITW dataset folder
        output_folder: Path to output folder for trial files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define paths for input and output files
    enroll_file = os.path.join(sitw_folder, 'eval', 'lists', 'enroll-core.lst')
    core_trials_file = os.path.join(sitw_folder, 'eval', 'keys', 'core-core.lst')
    multi_trials_file = os.path.join(sitw_folder, 'eval', 'keys', 'core-multi.lst')
    output_core_trials_file = os.path.join(output_folder, 'sitw_core_core.txt')
    output_multi_trials_file = os.path.join(output_folder, 'sitw_core_multi.txt')
    
    # Set to track all files for processing
    files = set()
    # Dictionary to map enrollment IDs to utterance IDs
    enroll2utt = {}
    
    print("Processing enrollment files...")
    # Read enrollment file and create enrollment to utterance mapping
    with open(enroll_file) as f:
        for line in f:
            parts = line.split()
            utt_id = parts[1].split('/')[1].strip()
            files.add(utt_id)
            enroll2utt[parts[0]] = utt_id
    
    # Process both core and multi trial files
    for trial_file, output_file in [
        (core_trials_file, output_core_trials_file),
        (multi_trials_file, output_multi_trials_file)
    ]:
        print(f"Processing {os.path.basename(trial_file)}...")
        
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(trial_file))
        
        with open(trial_file) as f, open(output_file, 'w') as outf:
            # Use tqdm for progress tracking
            for line in tqdm(f, total=total_lines, desc="Converting files"):
                parts = line.split()
                
                # Extract information from trial line
                utt_id = parts[1].split('/')[1]
                files.add(utt_id)
                
                # Convert label (imp=imposter, tar=target) to binary (0/1)
                label = 1 if parts[2].strip() == 'tar' else 0
                
                # Create full file paths
                enroll_filepath = os.path.join(sitw_folder, 'eval', 'audio', enroll2utt[parts[0]])
                test_filepath = os.path.join(sitw_folder, 'eval', 'audio', utt_id)
                
                # Convert files to WAV format
                enroll_wav = convert_flac_to_wav(enroll_filepath)
                test_wav = convert_flac_to_wav(test_filepath)
                
                # Write to output trial file
                outf.write(f'{label} {enroll_wav} {test_wav}\n')
    
    print(f"Trial files successfully generated in {output_folder}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare SITW dataset for speaker verification.'
    )
    parser.add_argument(
        '--sitw-folder', type=str, default='/home/jovyan/corpus/audio/sitw',
        help='Path to SITW dataset folder'
    )
    parser.add_argument(
        '--output-folder', type=str, default='/home/jovyan/workspace/vibe/trials',
        help='Path to output folder for trial files'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    prepare_sitw(args.sitw_folder, args.output_folder)


if __name__ == '__main__':
    main()