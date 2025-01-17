# python /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-70B" --seqlens 4096 4096 4096 8192 8192 \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune"


python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
 --models_to_compress "meta-llama/Llama-2-70b-hf" --seqlens 4096 \
 --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
 --self_attn_compression_algorithm "quantize" \
 --mlp_compression_algorithm "quantize" \
 --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
 --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
 --use_wandb \
 --wandb_project "compression_no_finetune" \
 --ppl_eval
# python -u /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-13b-hf" --seqlens 4096 \
#  --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128" \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --devices "cuda:6" "cuda:5" "cuda:4" "cuda:3" "cuda:2" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args_7d.yaml" \
#  --use_wandb \
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

