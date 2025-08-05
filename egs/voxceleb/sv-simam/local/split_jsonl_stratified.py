#!/usr/bin/env python3
"""
Stratified JSONL Dataset Splitter

This script splits a JSONL dataset into training and validation sets,
ensuring that speaker IDs are stratified across splits.
"""

import json
import argparse
from collections import defaultdict

from sklearn.model_selection import train_test_split


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split JSONL dataset into train and validation sets with stratification"
    )
    
    parser.add_argument(
        "--input", "-i",
        default="your_input.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--train", "-t",
        default="train.jsonl",
        help="Path to output training set JSONL file"
    )
    parser.add_argument(
        "--valid", "-v",
        default="valid.jsonl",
        help="Path to output validation set JSONL file"
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def load_jsonl_data(filepath):
    """Load data from a JSONL file."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


def group_by_speaker(data):
    """Group entries by speaker ID."""
    spk_to_entries = defaultdict(list)
    for entry in data:
        spk_to_entries[entry["spk_id"]].append(entry)
    return spk_to_entries


def prepare_for_stratified_split(spk_to_entries):
    """Prepare data for stratified split by flattening and creating labels."""
    flattened = []
    spk_labels = []
    for spk_id, entries in spk_to_entries.items():
        for entry in entries:
            flattened.append(entry)
            spk_labels.append(spk_id)
    return flattened, spk_labels


def write_jsonl(data, filepath):
    """Write data to a JSONL file."""
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    """Main function to split JSONL dataset."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load data from JSONL file
    print(f"Loading data from {args.input}...")
    data = load_jsonl_data(args.input)
    
    # Group entries by speaker
    print("Grouping entries by speaker...")
    spk_to_entries = group_by_speaker(data)
    
    # Prepare data for stratified split
    print("Preparing for stratified split...")
    flattened, spk_labels = prepare_for_stratified_split(spk_to_entries)
    
    # Perform stratified split
    print(f"Performing {args.valid_size:.1%} stratified split with seed {args.seed}...")
    train_data, valid_data = train_test_split(
        flattened,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=spk_labels
    )
    
    # Write to output files
    print(f"Writing training data to {args.train}...")
    write_jsonl(train_data, args.train)
    
    print(f"Writing validation data to {args.valid}...")
    write_jsonl(valid_data, args.valid)
    
    # Print summary
    print(f"Split complete. Training set: {len(train_data)} items")
    print(f"Validation set: {len(valid_data)} items")


if __name__ == "__main__":
    main()