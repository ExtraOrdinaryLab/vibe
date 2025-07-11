#!/bin/bash

# Hugging Face token
TOKEN=""

# Output directory
OUTDIR="/home/jovyan/corpus/audio/voxceleb1"

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

# List of URLs
URLS=(
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb1/dev/vox1_dev_wav_partaa"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb1/dev/vox1_dev_wav_partab"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb1/dev/vox1_dev_wav_partac"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb1/dev/vox1_dev_wav_partad"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb1/test/vox1_test_wav.zip"
)

# Loop through URLs and download each file
for url in "${URLS[@]}"; do
  echo "Downloading: $url"
  wget --header="Authorization: Bearer $TOKEN" -P "$OUTDIR" "$url"
done
