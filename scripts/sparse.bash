#!/bin/bash

DATASETS=("pajama" "c4")
N_SAMPLES=128
MODELS=("meta-llama/Llama-2-70b-hf" )
# "meta-llama/Llama-2-13b-hf")
PATTERNS=("2_4" "4_8")

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for PATTERN in "${PATTERNS[@]}"; do
            SAVE_PATH="./models/$MODEL/compressed/sparse/$PATTERN/$N_SAMPLES/$DATASET"
            kwargs_path=/data/lliu/huffman/scripts/1layer_compress/yamls/sparse_only/sparse_$PATTERN.yaml
            echo "Generating hessians for $MODEL with $n_samples samples"
            echo "kwargs_path: $kwargs_path"
            echo "Saving to $SAVE_PATH"
            if [[ $MODEL_PATH == *"Llama-3"* ]]; then
                export SEQLEN=8192
                echo "Using Llama-3 so setting seqlen to 8192"
            else
                export SEQLEN=4096
                echo "Using Llama-2 so setting seqlen to 4096"
            fi

            python -u /data/lliu/huffman/scripts/sparse.py \
                --base_model $MODEL \
                --sparse_kwargs_path $kwargs_path \
                --device cuda:0 \
                --save_path $SAVE_PATH \
                --seqlen $SEQLEN \
                --zero_shot_tasks None

        done
    done
done
