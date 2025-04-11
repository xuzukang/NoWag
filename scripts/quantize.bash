#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1 #uncomment to specify GPUs
enviroment="NoWAC-VQ"

models=("meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-70b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-70B"
)

for model in "${models[@]}"; do
    echo "===========Pruning $model with $pattern pattern=========="
    cmd="python -u NoWAG.py run_name=2bit_vq"
    echo "running command: $cmd"
    conda run -n $enviroment --live-stream $cmd
done