export MODEL_PATH=meta-llama/Meta-Llama-3-70B
export SAVE_PATH="./models/$MODEL_PATH/hessians"
export n_samples=128

python -u scripts/generate_hessians.py $MODEL_PATH pajama \
--seqlen 8192 \
--device cuda:7 \
--nsamples_train $n_samples \
--nsamples_val 0 \
--save_path "$SAVE_PATH/pajama/$n_samples" \
--offload_activations

