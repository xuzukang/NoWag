# python /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-70B" --seqlens 4096 4096 4096 8192 8192 \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
#  --use_wandb \
#  --wandb_project "compression_no_finetune"

python /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
 --models_to_compress "meta-llama/Llama-2-7b-hf" --seqlens 4096 \
 --hessian_path "/data/lliu/huffman/models/{model_name}/hessians_new/pajama/2048" \
 --self_attn_compression_algorithm "quantize" \
 --mlp_compression_algorithm "quantize" \
 --devices "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7" \
 --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
 --use_wandb \
 --resume_wandb \
 --wandb_id "hdrm7hyq" \
 --wandb_project "compression_no_finetune"
