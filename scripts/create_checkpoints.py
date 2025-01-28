# import yaml 

# model_name = "meta-llama/Meta-Llama-3-8B"
# allocation_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/allocation/test_allocation1/allocation.yaml"
# allocations_dict = yaml.load(open(allocation_path, "r"), Loader = yaml.FullLoader)


# sparse_checkpoints_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/compressed/151/checkpoints.yaml"
# sparse_checkpoints = yaml.load(open(sparse_checkpoints_path, "r"), Loader = yaml.FullLoader)

# non_sparse_checkpoints_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/compressed/154/checkpoints.yaml"
# non_sparse_checkpoints = yaml.load(open(non_sparse_checkpoints_path, "r"), Loader = yaml.FullLoader)

# new_checkpoints_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/compressed/test_sparse/checkpoints.yaml"
# new_checkpoints_dict = {}


# for key in sparse_checkpoints:
#     #get the allocation
#     # print(allocations_dict)
#     allocation_config = allocations_dict[key.replace(model_name + "/", "")+".pt"]
    
#     if allocation_config["sparse_frac"] == 0:
#         new_checkpoints_dict[key] = non_sparse_checkpoints[key]
#     else:
#         new_checkpoints_dict[key] = sparse_checkpoints[key]
        
        
# import os 
# os.makedirs(os.path.dirname(new_checkpoints_path), exist_ok = True)

# with open(new_checkpoints_path, "w") as f:
#     yaml.dump(new_checkpoints_dict, f)
    
    
# nohup_eval_command = f"nohup python -u perplexity_eval.py --base_model meta-llama/Meta-Llama-3-8B --seqlen 8192 --checkpoint_list_path {new_checkpoints_path} --device cuda:7 > {new_checkpoints_path.replace('/checkpoints.yaml', '/ppl_eval.log')} 2>&1 &"

# os.system(nohup_eval_command)




# import yaml 

# model_name = "meta-llama/Meta-Llama-3-8B"
# allocation_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/allocation/test_allocation1/allocation.yaml"
# allocations_dict = yaml.load(open(allocation_path, "r"), Loader = yaml.FullLoader)


# sparse_checkpoints_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/compressed/151/checkpoints.yaml"
# sparse_checkpoints = yaml.load(open(sparse_checkpoints_path, "r"), Loader = yaml.FullLoader)

# non_sparse_checkpoints_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/compressed/154/checkpoints.yaml"
# non_sparse_checkpoints = yaml.load(open(non_sparse_checkpoints_path, "r"), Loader = yaml.FullLoader)

# new_checkpoints_path = "/data/lliu/huffman/models/meta-llama/Meta-Llama-3-8B/compressed/test_sparse/checkpoints.yaml"
# new_checkpoints_dict = {}


# for key in sparse_checkpoints:
#     #get the allocation
#     # print(allocations_dict)
#     allocation_config = allocations_dict[key.replace(model_name + "/", "")+".pt"]
    
#     if allocation_config["sparse_frac"] == 0:
#         new_checkpoints_dict[key] = non_sparse_checkpoints[key]
#     else:
#         new_checkpoints_dict[key] = sparse_checkpoints[key]
        
        
# import os 
# os.makedirs(os.path.dirname(new_checkpoints_path), exist_ok = True)

# with open(new_checkpoints_path, "w") as f:
#     yaml.dump(new_checkpoints_dict, f)
    
    
# nohup_eval_command = f"nohup python -u perplexity_eval.py --base_model meta-llama/Meta-Llama-3-8B --seqlen 8192 --checkpoint_list_path {new_checkpoints_path} --device cuda:7 > {new_checkpoints_path.replace('/checkpoints.yaml', '/ppl_eval.log')} 2>&1 &"

# os.system(nohup_eval_command)



import yaml 

model_name = "meta-llama/Llama-2-7b-hf"
allocation_path = f"/data/lliu/huffman/models/{model_name}/allocation/test_allocation3/allocation.yaml"
allocations_dict = yaml.load(open(allocation_path, "r"), Loader = yaml.FullLoader)


sparse_checkpoints_path = f"/data/lliu/huffman/models/{model_name}/compressed/150/checkpoints.yaml"
sparse_checkpoints = yaml.load(open(sparse_checkpoints_path, "r"), Loader = yaml.FullLoader)

non_sparse_checkpoints_path = f"/data/lliu/huffman/models/{model_name}/compressed/166/checkpoints.yaml"
non_sparse_checkpoints = yaml.load(open(non_sparse_checkpoints_path, "r"), Loader = yaml.FullLoader)

new_checkpoints_path = f"/data/lliu/huffman/models/{model_name}/compressed/test_sparse_6d_not_Wanda_entropy/checkpoints.yaml"
new_checkpoints_dict = {}


for key in sparse_checkpoints:
    #get the allocation
    # print(allocations_dict)
    allocation_config = allocations_dict[key.replace(model_name + "/", "")+".pt"]
    
    if allocation_config["sparse_frac"] == 0:
        new_checkpoints_dict[key] = non_sparse_checkpoints[key]
    else:
        new_checkpoints_dict[key] = sparse_checkpoints[key]
        
        
import os 
os.makedirs(os.path.dirname(new_checkpoints_path), exist_ok = True)

with open(new_checkpoints_path, "w") as f:
    yaml.dump(new_checkpoints_dict, f)
    
    
nohup_eval_command = f"nohup python -u perplexity_eval.py --base_model meta-llama/Meta-Llama-3-8B --seqlen 4096 --checkpoint_list_path {new_checkpoints_path} --device cuda:7 > {new_checkpoints_path.replace('/checkpoints.yaml', '/ppl_eval.log')} 2>&1 &"

os.system(nohup_eval_command)