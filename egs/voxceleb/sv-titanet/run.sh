#!/bin/bash

set -e
. ./path.sh || exit 1

stage=1
stop_stage=3

exp=exp
exp_name=titanet_small
gpus="0 1 2 3"

. parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Train the speaker embedding model.
    echo "Stage1: Training the speaker model..."
    num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
    torchrun --nproc_per_node=$num_gpu --master-port 29501 vibe/bin/train.py \
        --config conf/titanet_small.yaml \
        --gpu $gpus \
        --train_manifest /home/jovyan/workspace/vibe/manifests/voxceleb2_dev.jsonl \
        --noise_scp /home/jovyan/workspace/vibe/manifests/musan.scp \
        --reverb_scp /home/jovyan/workspace/vibe/manifests/rirs.scp \
        --exp_dir $exp_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Extract embeddings of test datasets.
    echo "Stage2: Extracting speaker embeddings..."
    num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
    torchrun --nproc_per_node=$num_gpu --master-port 29501 vibe/bin/extract.py \
        --exp_dir $exp_dir \
        --audio_scp /home/jovyan/workspace/vibe/manifests/voxceleb1.scp \
        --use_gpu \
        --gpu $gpus
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Output score metrics.
    echo "Stage3: Computing score metrics..."
    trials_dir="/home/jovyan/workspace/vibe/trials"
    trials="$trials_dir/voxceleb1_o.txt $trials_dir/voxceleb1_e.txt $trials_dir/voxceleb1_h.txt"
    python vibe/bin/sv_evaluation.py \
        --enrol_data $exp_dir/embeddings \
        --test_data $exp_dir/embeddings \
        --scores_dir $exp_dir/scores \
        --trials $trials
fi