#!/bin/bash

python local/create_scp.py \
    --audio-dir /home/jovyan/corpus/audio/voxceleb1 \
    --output /home/jovyan/workspace/vibe/manifests/voxceleb1.scp \
    --extensions .wav