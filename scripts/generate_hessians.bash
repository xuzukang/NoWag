#!/bin/bash

export seed=0
dataset="pajama"
MODELS=(
    "datasets/Llama-2-7b-hf"
    # "datasets/Llama-2-13b-hf"
    # "datasets/Llama-2-70b-hf"
    # "datasets/Meta-Llama-3-8B"
    # "datasets/Meta-Llama-3-70B"
)
N_SAMPLES=(128)

SAVE_PATH_PARENT="./models"

for n_samples in "${N_SAMPLES[@]}"; do
    for MODEL_PATH in "${MODELS[@]}"; do

        echo "===========Generating hessians for $MODEL_PATH with $n_samples samples=========="
        cmd="python -u scripts/generate_hessians.py $MODEL_PATH $dataset \
        --device cuda:1 \
        --nsamples $n_samples \
        --offload_activations --forward_pass_batch_size 8 \
        --hessianDiag_save_path "${SAVE_PATH_PARENT}/$MODEL_PATH/hessianDiags/seed_$seed/$dataset/$n_samples" \
        --weight_save_path "${SAVE_PATH_PARENT}/$MODEL_PATH/original_weights" \
        --seed $seed"
        echo $cmd
        conda run -n NoWag --live-stream $cmd

        # conda run -n NoWAC-VQ --live-stream python -u scripts/generate_hessians.py $MODEL_PATH $dataset \
        # --device cuda:1 \
        # --nsamples $n_samples \
        # --offload_activations --forward_pass_batch_size 8 \
        # --hessianDiag_save_path "${SAVE_PATH_PARENT}/$MODEL_PATH/hessianDiags/seed_$seed/$dataset/$n_samples" \
        # --weight_save_path "${SAVE_PATH_PARENT}/$MODEL_PATH/original_weights" \
        # --seed $seed
    done
done


