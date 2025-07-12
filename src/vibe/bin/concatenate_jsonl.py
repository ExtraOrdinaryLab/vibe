"""
This script concatenates multiple JSONL files, where each file may come from
different datasets (like Fisher, VoxCeleb, Switchboard). Since speaker IDs
might be duplicated across datasets, this script prepends a dataset-specific
prefix to each speaker ID.
"""

import os
import json
import argparse
from typing import List


def concatenate_jsonl(input_files: List[str], prefixes: List[str], output_file: str) -> None:
    """
    Concatenate multiple JSONL files, prepending dataset-specific prefixes to speaker IDs.
    
    Args:
        input_files: List of paths to input JSONL files
        prefixes: List of prefixes to prepend to speaker IDs, should match input_files
        output_file: Path to the output JSONL file
    """
    # Ensure we have at least 2 input files as required
    if len(input_files) < 2:
        raise ValueError("At least two input files are required for concatenation")
    
    # Check that number of prefixes matches number of input files
    if len(input_files) != len(prefixes):
        raise ValueError(f"Number of input files ({len(input_files)}) must match number of prefixes ({len(prefixes)})")
    
    # Validate that all input files exist
    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    total_entries = 0
    
    with open(output_file, 'w') as out_f:
        # Process each input file with its corresponding prefix
        for input_file, prefix in zip(input_files, prefixes):
            file_entries = 0
            
            with open(input_file, 'r') as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse the JSON entry
                        data = json.loads(line)
                        
                        # Prepend the prefix to the speaker ID
                        if 'spk_id' in data:
                            data['spk_id'] = f"{prefix}_{data['spk_id']}"
                        else:
                            print(f"Warning: Entry without 'spk_id' in {input_file}")
                            continue
                        
                        # Write the modified entry to the output file
                        out_f.write(json.dumps(data) + '\n')
                        file_entries += 1
                        
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {input_file}")
            
            print(f"Processed {file_entries} entries from {input_file}")
            total_entries += file_entries
    
    print(f"Successfully concatenated {len(input_files)} files with {total_entries} total entries into {output_file}")


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Concatenate JSONL files from different datasets and add prefixes to speaker IDs"
    )
    parser.add_argument(
        "--inputs", 
        nargs='+', 
        required=True, 
        help="Input JSONL files (e.g., voxceleb1.jsonl voxceleb2.jsonl switchboard.jsonl)"
    )
    parser.add_argument(
        "--prefixes", 
        nargs='+', 
        required=True, 
        help="Prefixes to prepend to speaker IDs (e.g., vox1 vox2 swbd)"
    )
    parser.add_argument(
        "--output", 
        default="concatenated.jsonl", 
        help="Output JSONL file (default: concatenated.jsonl)"
    )
    
    args = parser.parse_args()
    concatenate_jsonl(args.inputs, args.prefixes, args.output)


if __name__ == "__main__":
    main()