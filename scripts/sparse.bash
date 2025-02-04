#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
DATASETS=("pajama")
N_SAMPLES=128
MODELS=("meta-llama/Llama-2-70b-hf")
# "meta-llama/Llama-2-13b-hf")
PATTERNS=("2_4" "4_8")

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for PATTERN in "${PATTERNS[@]}"; do
            SAVE_PATH="./models/$MODEL/compressed/sparse/$PATTERN/$N_SAMPLES/$DATASET"
            kwargs_path=/data/lliu/huffman/scripts/1layer_compress/yamls/sparse_only/sparse_$PATTERN.yaml
            temp_save_path="temp/temp_model"
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
            
            mkdir -p $SAVE_PATH
            python -u /data/lliu/huffman/scripts/sparse.py \
                --hessian_dir "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/$DATASET/$N_SAMPLES" \
                --base_model $MODEL \
                --sparse_kwargs_path $kwargs_path \
                --device cuda:0 \
                --ppl_datasets wikitext2 c4 \
                --save_path $SAVE_PATH \
                --save_and_load_model \
                --save_and_load_temp_path $temp_save_path \
                --seqlen $SEQLEN > $SAVE_PATH/sparse_log.log

            rm -rf $temp_save_path
        done
    done
done
