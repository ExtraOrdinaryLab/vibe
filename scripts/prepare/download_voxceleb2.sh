#!/bin/bash

# Hugging Face token
TOKEN=""

# Output directory
OUTDIR="/home/jovyan/corpus/audio/voxceleb2"

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

# List of URLs
URLS=(
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partaa"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partab"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partac"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partad"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partae"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partaf"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partag"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/dev/vox2_dev_aac_partah"
  "https://huggingface.co/datasets/confit/voxceleb-full/resolve/main/voxceleb2/test/vox2_test_aac.zip"
)

# Loop through URLs and download each file
for url in "${URLS[@]}"; do
  echo "Downloading: $url"
  wget --header="Authorization: Bearer $TOKEN" -P "$OUTDIR" "$url"
done
