export MODEL_PATH="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/tensorized/128"

python -u post_quantization_fine_tune.py \
--quantized_model_path="$MODEL_PATH/no_finetune2" \
--save_path="$MODEL_PATH/finetune_soft_labels" \
--device=cuda:7 \
--log_wandb \
--finetune_epochs=1 \
--finetune_lr=1e-5 \
--eval_every_samples=64 \
--update_every_n_tokens=4096 \
--finetune_nsamples_train=128 \
--finetune_nsamples_val=0 \
--finetune_adam_beta2=0.95 \
--soft_labels