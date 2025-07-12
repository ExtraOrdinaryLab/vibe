"""
Script to split a JSONL file into training and validation sets.
"""

import os
import json
import random
import argparse
from collections import defaultdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Split a JSONL file into training and validation sets.")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--train-output", "-t", required=True, help="Output path for training set")
    parser.add_argument("--val-output", "-v", required=True, help="Output path for validation set")
    parser.add_argument("--ratio", "-r", type=float, default=0.95, 
                        help="Ratio of data to use for training (default: 0.95)")
    parser.add_argument("--split-by", type=str, default=None, 
                        help="Field to use for stratified splitting (e.g., 'spk_id'). If not provided, random splitting is used.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line}")
    return data


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def split_random(data, train_ratio, seed=42):
    """Split data randomly."""
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def split_stratified(data, split_field, train_ratio, seed=42):
    """
    Split data by stratifying on a specific field.
    This ensures that items with the same value for the specified field
    will not be split across training and validation sets.
    """
    random.seed(seed)
    
    # Group data by the specified field
    grouped_data = defaultdict(list)
    for item in data:
        # Check if the field exists in the item
        if split_field not in item:
            print(f"Warning: Field '{split_field}' not found in item: {item}")
            continue
            
        group_key = item[split_field]
        grouped_data[group_key].append(item)
    
    # Get list of unique values for the split field and shuffle them
    unique_values = list(grouped_data.keys())
    random.shuffle(unique_values)
    
    # Split unique values into training and validation sets
    split_idx = int(len(unique_values) * train_ratio)
    train_values = unique_values[:split_idx]
    val_values = unique_values[split_idx:]
    
    # Collect data for each set
    train_data = []
    val_data = []
    
    for value in train_values:
        train_data.extend(grouped_data[value])
    
    for value in val_values:
        val_data.extend(grouped_data[value])
    
    return train_data, val_data


def main():
    """Main function to split the JSONL file."""
    args = parse_arguments()
    
    # Load data
    print(f"Loading data from {args.input}")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} records")
    
    # Split data
    if args.split_by:
        print(f"Splitting data stratified by field: {args.split_by}")
        train_data, val_data = split_stratified(data, args.split_by, args.ratio, args.seed)
    else:
        print("Splitting data randomly")
        train_data, val_data = split_random(data, args.ratio, args.seed)
    
    print(f"Training set: {len(train_data)} records")
    print(f"Validation set: {len(val_data)} records")
    
    # Save data
    save_jsonl(train_data, args.train_output)
    save_jsonl(val_data, args.val_output)
    print(f"Training set saved to {args.train_output}")
    print(f"Validation set saved to {args.val_output}")


if __name__ == "__main__":
    main()