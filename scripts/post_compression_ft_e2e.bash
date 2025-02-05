# !/bin/bash

export CUDA_VISIBLE_DEVICES=3,4,5,6
python scripts/finetune_parallel.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --compressed_run_name 166 \
    --ft_grad_checkpoint \
    --ft_train_mode
