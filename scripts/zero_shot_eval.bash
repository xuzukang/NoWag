#!/bin/bash

quantized_weights=('/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/166' \
                    '/data/lliu/huffman/models/meta-llama/Llama-2-13b-hf/compressed/166' \
                    '/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/compressed/167')
base_models=('meta-llama/Llama-2-7b-hf' \
             'meta-llama/Llama-2-13b-hf' \
             'meta-llama/Meta-Llama-3-8B')

for i in {0..2}
do
    echo "Running zero-shot eval for ${quantized_weights[$i]} which is a quantized version of ${base_models[$i]}"
    python -u zero_shot.py \
        --base_model ${base_models[$i]} \
        --quantized_weight_yaml "${quantized_weights[$i]}/checkpoints.yaml" \
        --tasks winogrande rte piqa arc_easy arc_challenge \
        --save > "${quantized_weights[$i]}/zero_shot_eval.log 2>&1"

done
