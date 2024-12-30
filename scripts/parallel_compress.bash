# python /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
#  --models_to_compress "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-70B" --seqlens 4096 4096 4096 8192 8192 \
#  --self_attn_compression_algorithm "quantize" \
#  --mlp_compression_algorithm "quantize" \
#  --yaml_path "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml" \
#  --use_already_done \
#  --use_wandb \
#  --wandb_project "compression_no_finetune"

python /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
 --models_to_compress "meta-llama/Llama-2-7b-hf" --seqlens 4096 \
 --self_attn_compression_algorithm "joint" \
 --mlp_compression_algorithm "joint" \
 --yaml_path "/data/lliu/huffman/scripts/1layer_compress/joint_args.yaml" \
 --use_already_done \
 --use_wandb \
 --wandb_project "compression_no_finetune"
