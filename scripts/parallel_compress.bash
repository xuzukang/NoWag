python /data/lliu/huffman/scripts/layer_by_layer_parallel_compress.py \
 --models_to_compress "meta-llama/Llama-2-7b-hf" \
 --self_attn_compression_algorithm "joint" \
 --mlp_compression_algorithm "joint" \
 --yaml_path "/data/lliu/huffman/scripts/1layer_compress/joint_args.yaml" \
#  --use_wandb \
 --wandb_project "compression_no_finetune"

