#!/bin/bash

set -e
. ./path.sh || exit 1

stage=1
stop_stage=4

exp=exp
exp_name=ecapa_tdnn_v10
score_norm="" # z-norm t-norm s-norm as-norm
cohort_size="4000"
top_n="200"
gpus="0 1 2 3"

. parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Train the speaker embedding model.
    echo "Stage1: Training the speaker model..."
    num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
    torchrun --nproc_per_node=$num_gpu --master-port 29500 vibe/bin/train.py \
        --config conf/ecapa_tdnn_v10.yaml \
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
    manifests_dir="/home/jovyan/workspace/vibe/manifests"
    torchrun --nproc_per_node=$num_gpu --master-port 29500 vibe/bin/extract.py \
        --exp_dir $exp_dir \
        --audio_scp $manifests_dir/voxceleb1.scp $manifests_dir/sitw_eval.scp \
        --max_frames 100000 \
        --use_gpu \
        --gpu $gpus
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Extract cohort embeddings for score normalization.
    echo "Stage3: Extracting cohort embeddings..."
    if [ ! -z "$score_norm" ]; then
        num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
        cohort_scp="/home/jovyan/workspace/vibe/manifests/voxceleb2_dev.scp"
        cohort_emb_dir="$exp_dir/cohort_embeddings"
        torchrun --nproc_per_node=$num_gpu --master-port 29500 vibe/bin/extract_cohort.py \
            --exp_dir $exp_dir \
            --cohort_scp $cohort_scp \
            --output_dir $cohort_emb_dir \
            --max_samples $cohort_size \
            --use_gpu
    else
        echo "Score normalization disabled, skipping cohort embedding extraction."
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Output score metrics.
    echo "Stage4: Computing score metrics..."
    trials_dir="/home/jovyan/workspace/vibe/trials"
    trials="$trials_dir/voxceleb1_o.txt $trials_dir/voxceleb1_e.txt $trials_dir/voxceleb1_h.txt $trials_dir/sitw_core_core.txt $trials_dir/sitw_core_multi.txt"
    norm_args=""
    if [ ! -z "$score_norm" ]; then
        cohort_emb_dir="$exp_dir/cohort_embeddings"
        norm_args="--score_norm $score_norm --cohort_data ${cohort_emb_dir} --top_n ${top_n}"
    fi
    python vibe/bin/sv_evaluation.py \
        --enrol_data $exp_dir/embeddings \
        --test_data $exp_dir/embeddings \
        --scores_dir $exp_dir/scores \
        --trials $trials \
        ${norm_args}
fi