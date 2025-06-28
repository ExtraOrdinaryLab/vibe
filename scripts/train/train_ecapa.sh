#!/bin/bash

accelerate launch train_ecapa.py \
    --manifest_paths manifests/voxceleb1_dev.jsonl manifests/voxceleb2_dev.jsonl \
    --label_column_name spk_id \
    --output_dir outputs/ecapa_tdnn_aug_concat \
    --max_length_seconds 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-3 \
    --weight_decay 2e-5 \
    --lr_scheduler_type cyclic \
    --max_train_steps 520000 \
    --logging_steps 10 \
    --checkpointing_steps 130000 \
    --num_workers 32 \
    --concat_augment \
    --rir_path /home/jovyan/corpus/audio/RIRS_NOISES \
    --musan_path /home/jovyan/corpus/audio/musan \
    --seed 914