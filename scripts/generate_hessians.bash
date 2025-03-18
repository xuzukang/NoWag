#!/bin/bash

# export MODEL_PATH=meta-llama/Meta-Llama-3-70B
# export MODEL_PATH=meta-llama/Meta-Llama-3-8B
# export MODEL_PATH=meta-llama/Llama-2-70b-hf
# export MODEL_PATH=meta-llama/Llama-2-13b-hf
# export MODEL_PATH=meta-llama/Llama-2-7b-hf
# export n_samples=256
# export seed=0
# dataset="pajama"
# MODELS=("meta-llama/Llama-2-7b-hf")
# N_SAMPLES=(6144)

# for n_samples in "${N_SAMPLES[@]}"; do
#     for MODEL_PATH in "${MODELS[@]}"; do
#         export SAVE_PATH="./models/$MODEL_PATH/hessians_new/seed_$seed"

#         echo "Generating hessians for $MODEL_PATH with $n_samples samples"
#         echo "Saving to $SAVE_PATH"
#         if [[ $MODEL_PATH == *"Llama-3"* ]]; then
#             export SEQLEN=8192
#             echo "Using Llama-3 so setting seqlen to 8192"
#         else
#             export SEQLEN=4096
#             echo "Using Llama-2 so setting seqlen to 4096"
#         fi

#         python -u scripts/generate_hessians.py $MODEL_PATH $dataset \
#         --seqlen $SEQLEN \
#         --device cuda:1 \
#         --nsamples_train $n_samples \
#         --nsamples_val 0 \
#         --save_path "$SAVE_PATH/$dataset/$n_samples" \
#         --seed $seed \
#         --offload_activations --forward_pass_batch_size 8 \
#         --stop_after_first_layer
#     done
# done



export seed=0
dataset="pajama"
MODELS=("meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-70b-hf"
    "meta-llama/Meta-Llama-3-8B"
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
        conda run -n NoWAC-VQ --live-stream $cmd

        # conda run -n NoWAC-VQ --live-stream python -u scripts/generate_hessians.py $MODEL_PATH $dataset \
        # --device cuda:1 \
        # --nsamples $n_samples \
        # --offload_activations --forward_pass_batch_size 8 \
        # --hessianDiag_save_path "${SAVE_PATH_PARENT}/$MODEL_PATH/hessianDiags/seed_$seed/$dataset/$n_samples" \
        # --weight_save_path "${SAVE_PATH_PARENT}/$MODEL_PATH/original_weights" \
        # --seed $seed
    done
done


