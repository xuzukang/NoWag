python -u post_quantization_fine_tune.py \
--quantized_model_path="/data/lliu/huffman/quantized_models/llama-2-7b/2bpv/quantized" \
--save_path="/data/lliu/huffman/quantized_models/llama-2-7b/e2e_finetuned/quantized" \
--device=cuda:7 \
--log_wandb \
--finetune_epochs=10 \
--finetune_lr=1e-4 \
--eval_every_samples=64 \
--update_every_n_tokens=16384