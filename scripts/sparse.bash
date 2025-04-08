# we expect one argument, the name of the model being processed


model_name=$1

output_dir="models/${model_name}/sparse"
mkdir -p ${output_dir}


sparse_kwargs_dir="/data/lliu/huffman/yamls/sparse"

kwargs_to_run=(
    "50_2_4"
    "50_4_8"
    "50unstructured"
)
#for each set of kwargs, run the sparse script
for kwargs in "${kwargs_to_run[@]}"
do
    echo "Running sparse with kwargs ${kwargs}"
    output_dir_use="${output_dir}/${kwargs}"
    log_path="${output_dir_use}/log.log"
    echo "Logging to ${log_path}"
    mkdir -p ${output_dir_use}

    python scripts/sparse_parallel.py \
        --base_model ${model_name} \
        --sparse_kwargs_path "${sparse_kwargs_dir}/${kwargs}.yaml" \
        --save_path ${output_dir_use} \
        > ${log_path} 2>&1
        
done