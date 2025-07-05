#!/bin/bash

CHECKPOINT_PATH="/home/jovyan/workspace/vibe/outputs/ecapa_tdnn_aug"
TRIAL_FILE="voxceleb1_o_trial.txt"

accelerate launch --mixed_precision=fp16 verification.py \
    --manifest_paths manifests/voxceleb1_test.jsonl \
    --model_name_or_path $CHECKPOINT_PATH \
    --output_dir $CHECKPOINT_PATH \
    --trial_file $TRIAL_FILE \
    --device cuda \
    --batch_size 4 \
    --num_workers 32 \
    --label_column_name spk_id \
    --use_disk_cache \
    --score_norm none \
    --cohort_manifest manifests/voxceleb2_dev.jsonl manifests/voxceleb1_dev.jsonl \
    --cohort_size 200 \
    --num_cohort_samples 4000