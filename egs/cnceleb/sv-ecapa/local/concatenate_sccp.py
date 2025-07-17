"""
This script concatenates multiple SCP files.
Each line in an SCP file contains a SHA256 ID followed by a filepath.
The script also detects and handles duplicate IDs across different files.
"""

import argparse
import os
from typing import List, Dict, Set


def concatenate_scp(input_files: List[str], output_file: str, handle_duplicates: str = "error") -> None:
    """
    Concatenate multiple SCP files into a single output file.
    
    Args:
        input_files: List of paths to input SCP files
        output_file: Path to the output SCP file
        handle_duplicates: How to handle duplicate IDs ('error', 'keep_first', 'keep_last')
    """
    # Ensure we have at least 2 input files
    if len(input_files) < 2:
        raise ValueError("At least two input files are required for concatenation")
    
    # Validate that all input files exist
    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Validate duplicate handling option
    valid_duplicate_options = ["error", "keep_first", "keep_last"]
    if handle_duplicates not in valid_duplicate_options:
        raise ValueError(f"handle_duplicates must be one of {valid_duplicate_options}")
    
    # Dictionary to keep track of IDs and their corresponding filepaths
    id_to_filepath: Dict[str, str] = {}
    # Set to keep track of duplicate IDs
    duplicate_ids: Set[str] = set()
    
    total_entries = 0
    
    # First pass: read all files and identify duplicates
    for input_file in input_files:
        file_entries = 0
        
        with open(input_file, 'r') as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Split the line into ID and filepath
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping invalid line in {input_file}: {line}")
                        continue
                    
                    sha256_id, filepath = parts
                    
                    # Check for duplicates
                    if sha256_id in id_to_filepath:
                        duplicate_ids.add(sha256_id)
                        if handle_duplicates == "keep_last":
                            id_to_filepath[sha256_id] = filepath
                    else:
                        id_to_filepath[sha256_id] = filepath
                    
                    file_entries += 1
                except Exception as e:
                    print(f"Warning: Error processing line in {input_file}: {line}")
                    print(f"         Error: {str(e)}")
            
            print(f"Processed {file_entries} entries from {input_file}")
            total_entries += file_entries
    
    # Handle duplicates according to the specified option
    if duplicate_ids and handle_duplicates == "error":
        raise ValueError(f"Found {len(duplicate_ids)} duplicate IDs across input files. Use --handle-duplicates to specify how to handle them.")
    
    # Write the entries to the output file
    with open(output_file, 'w') as out_f:
        for sha256_id, filepath in id_to_filepath.items():
            out_f.write(f"{sha256_id} {filepath}\n")
    
    print(f"Successfully concatenated {len(input_files)} files with {len(id_to_filepath)} unique entries into {output_file}")
    if duplicate_ids:
        print(f"Found and handled {len(duplicate_ids)} duplicate IDs according to '{handle_duplicates}' strategy")


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Concatenate SCP files containing SHA256 IDs and filepaths"
    )
    parser.add_argument(
        "--inputs", 
        nargs='+', 
        required=True, 
        help="Input SCP files to concatenate"
    )
    parser.add_argument(
        "--output", 
        default="concatenated.scp", 
        help="Output SCP file (default: concatenated.scp)"
    )
    parser.add_argument(
        "--handle-duplicates",
        choices=["error", "keep_first", "keep_last"],
        default="error",
        help="How to handle duplicate IDs: 'error' (raise error), 'keep_first' (keep first occurrence), "
             "'keep_last' (keep last occurrence)"
    )
    
    args = parser.parse_args()
    concatenate_scp(args.inputs, args.output, args.handle_duplicates)


if __name__ == "__main__":
    main()