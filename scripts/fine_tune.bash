export CUDA_VISIBLE_DEVICES=7,6,5,4

python -u /data/lliu/huffman/scripts/finetune_pl.py\
    --base_model meta-llama/Llama-2-7b-hf \
    --compressed_run_name 170 \
    --ft_n_train 32 \
    --ft_n_val 16
