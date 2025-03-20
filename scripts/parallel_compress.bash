#!/bin/bash

#echo this pid 
echo "PID: $$"

CONDA_ENV="NoWAC-VQ"

models=(
        "meta-llama/Llama-2-7b-hf"
        "meta-llama/Llama-2-13b-hf"
        "meta-llama/Llama-2-70b-hf"
        "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-70B"
        )
bits=(
    2
    3
        )

#loop through models
for bit in "${bits[@]}"; do
    for model in "${models[@]}"; do
        echo "====================="
        echo "Compressing $model with $bit bits"
        config_file="/data/lliu/huffman/yamls/quantizer" #path to the config file
        # rules for the config file to use, 

        # #hard coded skipping for Llama-2 7b 2 and 3 bits
        # if [[ $model == *"Llama-2-7b"* ]] && [[ $bit -lt 4 ]]; then
        #     echo "Skipping Llama-2-7b with $bit bits"
        #     continue
        # fi


        #if we are doing 2 bits
        if [[ $bit == 2 ]]; then
            #if we are not doing a 70b model
            if [[ $model != *"70B"* ]]; then
                config_file+="/2bits_6d_basic.yaml"
            else
                config_file+="/2bits_7d_basic.yaml"
            fi
        #if we are doing 3 bits
        elif [[ $bit == 3 ]]; then
            #for all models use the 3 bits 4d basic config
            config_file+="/3bits_4d_basic.yaml"
        
        #if we are doing 4 bits
        elif [[ $bit == 4 ]]; then
            #for all models use the 4 bits 4d basic config
            config_file+="/4bits_3d_basic.yaml"
        fi

        echo "Using config file $config_file"

        #get the seqlen from the model name
        if [[ $model == *"Llama-3"* ]]; then
            echo "Using Llama-3 so setting seqlen to 8192"
            seqlen=8192
        else
            echo "Using Llama-2 so setting seqlen to 4096"
            seqlen=4096
        fi

        #run the compression script
        cmd="python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
            --models_to_compress $model --seqlens $seqlen \
            --hessian_path /data/lliu/huffman/models/{model_name}/hessianDiags/seed_0/pajama/128 \
            --self_attn_compression_algorithm "quantize" \
            --mlp_compression_algorithm "quantize" \
            --devices "cuda:7" "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2"\
            --yaml_path $config_file \
            --ppl_eval"

        echo "Running command: $cmd"
        mkdir -p /data/lliu/huffman/logs/compression_logs
        conda run -n $CONDA_ENV --live-stream $cmd #> /data/lliu/huffman/logs/compression_logs/${model//\//_}_compression_${bit}bits.log 2>&1
        # break
        echo "====================="
    done
    # break    
done




# python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessianDiags/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:7" "cuda:6" "cuda:5" "cuda:4" "cuda:3" \
#  --yaml_path "/data/lliu/huffman/yamls/quantizer/2bits_6d_basic.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval

#  python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-13b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessianDiags/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:7" "cuda:6" "cuda:5" "cuda:4" "cuda:3" \
#  --yaml_path "/data/lliu/huffman/yamls/quantizer/2bits_6d_basic.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval

#  python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessianDiags/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:7" "cuda:6" "cuda:5" "cuda:4" "cuda:3" \
#  --yaml_path "/data/lliu/huffman/yamls/quantizer/2bits_6d_basic.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval
