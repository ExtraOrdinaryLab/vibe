#!/bin/bash

# Script to download and prepare the VietnamCeleb dataset
# The dataset is downloaded from Google Drive in multiple parts and then combined

# Exit on error
set -e

# Install the Google Drive downloader tool
echo "Installing Google Drive downloader..."
pip install gdrive-downloader

# Create directory for the dataset if it doesn't exist
echo "Creating directory for the dataset..."
DATASET_DIR=/home/jovyan/corpus/audio/vietnamceleb
mkdir -p $DATASET_DIR
cd $DATASET_DIR

# Download dataset parts from Google Drive (only if they don't exist)
echo "Checking and downloading dataset parts..."

# Check and download part 1 (.zip file)
if [ -f "$DATASET_DIR/vietnam-celeb-part.zip" ]; then
    echo "File vietnam-celeb-part.zip already exists. Skipping download."
else
    echo "Downloading vietnam-celeb-part.zip..."
    gdrive-download --url https://drive.google.com/file/d/1pMuT3DFzSwib7SVcRS8VkDwPuLTsemSG/view?usp=share_link --output $DATASET_DIR/vietnam-celeb-part.zip
fi

# Check and download part 2 (.z01 file)
if [ -f "$DATASET_DIR/vietnam-celeb-part.z01" ]; then
    echo "File vietnam-celeb-part.z01 already exists. Skipping download."
else
    echo "Downloading vietnam-celeb-part.z01..."
    gdrive-download --url https://drive.google.com/file/d/1xayHt2HRqE1aJ4HvtUT40_9XlgvfDfRY/view?usp=share_link --output $DATASET_DIR/vietnam-celeb-part.z01
fi

# Check and download part 3 (.z02 file)
if [ -f "$DATASET_DIR/vietnam-celeb-part.z02" ]; then
    echo "File vietnam-celeb-part.z02 already exists. Skipping download."
else
    echo "Downloading vietnam-celeb-part.z02..."
    gdrive-download --url https://drive.google.com/file/d/1MIlM78EbN_J9cApkNes_2_BrFrf8XwMc/view?usp=share_link --output $DATASET_DIR/vietnam-celeb-part.z02
fi

# Check and download part 4 (.z03 file)
if [ -f "$DATASET_DIR/vietnam-celeb-part.z03" ]; then
    echo "File vietnam-celeb-part.z03 already exists. Skipping download."
else
    echo "Downloading vietnam-celeb-part.z03..."
    gdrive-download --url https://drive.google.com/file/d/1h6Na58DC03p-502B9QpC5Z_FadUwAdNA/view?usp=share_link --output $DATASET_DIR/vietnam-celeb-part.z03
fi

# Check if the combined file already exists
if [ -f "$DATASET_DIR/full-dataset.zip" ]; then
    echo "Full dataset archive already exists. Skipping combination step."
else
    # Combine the split zip files into one complete zip file
    echo "Combining zip parts into full dataset..."
    zip -F $DATASET_DIR/vietnam-celeb-part.zip --out $DATASET_DIR/full-dataset.zip
fi

# Check if the dataset has already been extracted
# Assuming that after extraction there will be some files in the directory
if [ "$(ls -A $DATASET_DIR/data 2>/dev/null)" ]; then
    echo "Dataset already appears to be extracted. Skipping extraction."
else
    # Extract the dataset
    echo "Extracting dataset..."
    unzip $DATASET_DIR/full-dataset.zip -d $DATASET_DIR
fi

echo "VietnamCeleb dataset has been successfully prepared at $DATASET_DIR"

find data -type f -name "*.wav" | wc -l