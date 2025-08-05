#!/bin/bash

# Hugging Face token
TOKEN=""

# Output directory
OUTDIR="/home/jovyan/corpus/audio/sitw"

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

# List of URLs
URLS=(
  "https://huggingface.co/datasets/confit/sitw-full/resolve/main/sitw_database.v4.tar.gz"
)

# Loop through URLs and download each file
for url in "${URLS[@]}"; do
  echo "Downloading: $url"
  wget --header="Authorization: Bearer $TOKEN" -P "$OUTDIR" "$url"
done
