# python /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-70B" --seqlens 4096 4096 4096 8192 8192 \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune"


# python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Meta-Llama-3-8B" --seqlens 8192 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval

# python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "sparse" \
#  --mlp_compression_algorithm "sparse" \
#  --devices "cuda:7" "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/yamls/sparse_args4d.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval

python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
 --models_to_compress "meta-llama/Llama-2-70b-hf" --seqlens 4096 \
 --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
 --self_attn_compression_algorithm "quantize" \
 --mlp_compression_algorithm "quantize" \
 --devices "cuda:7" "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
 --yaml_path "/data/lliu/huffman/yamls/quantizer/2bits_6d_basic.yaml" \
 --use_wandb \
 --wandb_project "compression_no_finetune" \
 --ppl_eval

#  python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-70b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args2.yaml" \
#  --use_wandb \
#  --resume_wandb --wandb_id p9s2oq54 \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval


# python -u zero_shot.py \
#     --base_model meta-llama/Llama-2-70b-hf \
#     --quantized_weight_yaml "/data/lliu/huffman/models/meta-llama/Llama-2-70b-hf/compressed/169/checkpoints.yaml" \
#     --tasks winogrande rte piqa arc_easy arc_challenge \
#     --save > "/data/lliu/huffman/models/meta-llama/Llama-2-70b-hf/compressed/169/zero_shot_eval.log 2>&1"

#  python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Meta-Llama-3-8B" --seqlens 8192 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args2.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval

#  python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-13b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "sparse" \
#  --mlp_compression_algorithm "sparse" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/yamls/sparse_args7d.yaml" \
#  --use_wandb \
#  --resume_wandb --wandb_id oxiratbr \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval

 
# python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-70b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args_8d.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval




# python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf" "meta-llama/Meta-Llama-3-8B" --seqlens 4096 4096 4096 8192 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval


#  --resume_wandb --wandb_id gk0w13c2 \
# python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --discrete_update_hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_42/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune" \
#  --ppl_eval

