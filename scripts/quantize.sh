export MODEL_PATH=meta-llama/Llama-2-7b-hf
export SAVE_PATH= ./quantized_models/llama-2-7b/2bpv

python -u llama_quantize.py $MODEL_PATH pajama \
--quantize \
--device cuda:7 \
--nsamples_train 2048 \
--subvector_dim 4 \
--n_bits_per_value 2 \
--save_path "$SAVE_PATH/quantized" \
--lr 0.001 \
--nsamples_val 0 \
--add_bias \
--lr_multiple 1 \
--offload_activations