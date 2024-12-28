#!/bin/bash
# export MODEL_PATH=meta-llama/Meta-Llama-3-70B
# export MODEL_PATH=meta-llama/Meta-Llama-3-8B
export MODEL_PATH=meta-llama/Llama-2-70b-hf
# export MODEL_PATH=meta-llama/Llama-2-13b-hf
# export MODEL_PATH=meta-llama/Llama-2-7b-hf
export SAVE_PATH="./models/$MODEL_PATH/hessians_new"
export n_samples=128

echo "Generating hessians for $MODEL_PATH with $n_samples samples"
if [[ $MODEL_PATH == *"Llama-3"* ]]; then
    export SEQLEN=8192
    echo "Using Llama-3 so setting seqlen to 8192"
else
    export SEQLEN=4096
    echo "Using Llama-2 so setting seqlen to 4096"
fi

python -u scripts/generate_hessians.py $MODEL_PATH pajama \
--seqlen $SEQLEN \
--device cuda:0 \
--nsamples_train $n_samples \
--nsamples_val 0 \
--save_path "$SAVE_PATH/pajama/$n_samples" \
--offload_activations

