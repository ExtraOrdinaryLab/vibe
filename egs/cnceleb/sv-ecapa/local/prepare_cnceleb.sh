#!/bin/bash

python local/prepare_cnceleb.py \
    --cnceleb1_root /home/jovyan/corpus/audio/cnceleb1 \
    --cnceleb2_root /home/jovyan/corpus/audio/cnceleb2 \
    --manifest_root /home/jovyan/workspace/vibe/manifests \
    --manifest_prefix cnceleb \
    --max_workers 32
