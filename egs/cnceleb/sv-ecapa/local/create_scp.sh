#!/bin/bash

python local/create_scp.py \
    --audio-dir /home/jovyan/corpus/audio/cnceleb1 \
    --output /home/jovyan/workspace/vibe/manifests/cnceleb1.scp \
    --extensions .wav