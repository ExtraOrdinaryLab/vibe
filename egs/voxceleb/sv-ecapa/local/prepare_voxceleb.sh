#!/bin/bash

python local/prepare_voxceleb.py \
    --voxceleb_root /home/jovyan/corpus/audio/voxceleb1 \
    --manifest_root /home/jovyan/workspace/vibe/manifests \
    --manifest_prefix voxceleb1 \
    --max_workers 32

python local/prepare_voxceleb.py \
    --voxceleb_root /home/jovyan/corpus/audio/voxceleb2 \
    --manifest_root /home/jovyan/workspace/vibe/manifests \
    --manifest_prefix voxceleb2 \
    --max_workers 32