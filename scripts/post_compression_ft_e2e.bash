# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
python scripts/finetune_parallel.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --compressed_run_name 166 \
    --ft_train_mode \
    --ft_n_train 12 \
    --ft_n_val 4 \
    --seqlen 1024 \
    --ft_batch_size 1 \
    --ft_update_freq 8 \
    --ft_lr 1e-5 \
    --debug \
    --seed 42
