#!/bin/bash

# learning_rate = 0.3 * batch_size / 256

accelerate launch train_simclr.py \
    --manifest_paths manifests/voxceleb1_dev.jsonl \
    --label_column_name spk_id \
    --output_dir outputs/ecapa_tdnn_simclr \
    --max_length_seconds 3 \
    --num_views 2 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.6 \
    --weight_decay 2e-5 \
    --lr_scheduler_type cyclic \
    --num_train_epochs 100 \
    --logging_steps 10 \
    --checkpointing_steps 10000 \
    --num_workers 4 \
    --rir_path /home/jovyan/corpus/audio/RIRS_NOISES \
    --musan_path /home/jovyan/corpus/audio/musan \
    --seed 914