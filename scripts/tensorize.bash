export MODEL_PATH=meta-llama/Llama-2-7b-hf
export SAVE_PATH="./models/$MODEL_PATH/tensorized"

python -u llama_tensorize.py $MODEL_PATH pajama \
--seqlen 4096 \
--device cuda:7 \
--nsamples_train 128 \
--N_qudits 3 \
--save_path "$SAVE_PATH/128/no_finetune2" \
--lr 1e-2 \
--nsamples_val 0 \
--add_bias \
--offload_activations \
--forward_pass_batch_size 4