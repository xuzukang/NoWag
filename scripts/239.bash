export MODEL_PATH="./meta-llama/Llama-2-7b-hf"
python -u scripts/layer_by_layer_parallel_compress.py "./models/$MODEL_PATH/hessians" "./models/$MODEL_PATH" \
--log_dir "./logs/parallel_compress/$MODEL_PATH/joint" \
--self_attn_compression_algorithm joint \
--mlp_compression_algorithm joint \
--devices cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--tensor_compress_kwargs_path ./scripts/1layer_compress/tensor_args.yaml \
--quantize_compress_kwargs_path ./scripts/1layer_compress/quantizer_args.yaml
echo "Finished compressing Llama-2-7b-hf"

export MODEL_PATH="./meta-llama/Llama-2-13b-hf"
python -u scripts/layer_by_layer_parallel_compress.py "./models/$MODEL_PATH/hessians" "./models/$MODEL_PATH" \
--log_dir "./logs/parallel_compress/$MODEL_PATH/quantize" \
--self_attn_compression_algorithm quantize \
--mlp_compression_algorithm quantize \
--devices cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--tensor_compress_kwargs_path ./scripts/1layer_compress/tensor_args.yaml \
--quantize_compress_kwargs_path ./scripts/1layer_compress/quantizer_args_2bpv.yaml
echo "Finished quantizing Llama-2-13b-hf"

python -u scripts/layer_by_layer_parallel_compress.py "./models/$MODEL_PATH/hessians" "./models/$MODEL_PATH" \
--log_dir "./logs/parallel_compress/$MODEL_PATH/joint" \
--self_attn_compression_algorithm joint \
--mlp_compression_algorithm joint \
--devices cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--tensor_compress_kwargs_path ./scripts/1layer_compress/tensor_args.yaml \
--quantize_compress_kwargs_path ./scripts/1layer_compress/quantizer_args.yaml 
echo "Finished compressing Llama-2-13b-hf"

export MODEL_PATH="./meta-llama/Meta-Llama-3-8B"
python -u scripts/layer_by_layer_parallel_compress.py "./models/$MODEL_PATH/hessians" "./models/$MODEL_PATH" \
--log_dir "./logs/parallel_compress/$MODEL_PATH/quantize" \
--self_attn_compression_algorithm quantize \
--mlp_compression_algorithm quantize \
--devices cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--tensor_compress_kwargs_path ./scripts/1layer_compress/tensor_args.yaml \
--quantize_compress_kwargs_path ./scripts/1layer_compress/quantizer_args_2bpv.yaml
echo "Finished quantizing Meta-Llama-3-8B"

python -u scripts/layer_by_layer_parallel_compress.py "./models/$MODEL_PATH/hessians" "./models/$MODEL_PATH" \
--log_dir "./logs/parallel_compress/$MODEL_PATH/joint" \
--mlp_compression_algorithm joint \
--devices cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--tensor_compress_kwargs_path ./scripts/1layer_compress/tensor_args.yaml \
--quantize_compress_kwargs_path ./scripts/1layer_compress/quantizer_args.yaml
echo "Finished compressing Meta-Llama-3-8B"


