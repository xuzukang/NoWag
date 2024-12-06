export MODEL_PATH=meta-llama/Llama-2-7b-hf
export SAVE_PATH="./models/$MODEL_PATH/2bpv"

python -u llama_quantize.py $MODEL_PATH pajama \
--seqlen 4096 \
--device cuda:6 \
--nsamples_train 128 \
--subvector_dim 4 \
--n_bits_per_value 2 \
--save_path "$SAVE_PATH/128/quantized" \
--lr 0.001 \
--nsamples_val 0 \
--add_bias \
--lr_multiple 1 \
--offload_activations \
--forward_pass_batch_size 4
python -u perplexity_eval.py \
--checkpoint_path "$SAVE_PATH/128/quantized" \
--datasets "wikitext2" "c4" \
--device cuda:6