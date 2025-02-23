import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
print("pid", os.getpid())
import sys

import yaml
from typing import Tuple, Optional, Union, List, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


import src.quantize_compress as qc
import wandb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hessian_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/seed_0/pajama/128/layer_0/self_attn.q_proj.pt")
parser.add_argument("--weights_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/original_weights")
parser.add_argument("--save_path", type=str, default="/data/lliu/huffman/test/save_self_attn.q_proj.pt")
parser.add_argument("--device", type = str, default = "cuda:2",
                    help = "device to use for training")
parser.add_argument("--yaml_path", type = str, default = "/data/lliu/huffman/yamls/quantizer/quantizer_args.yaml")
# parser.add_argument("--d", type = int, default = 4,
#                     help = "subvector dimension")
# parser.add_argument("--n_bits", type = float, default = 2)
# parser.add_argument("--norm_order", type = int, nargs = "+", default = [0,1],)
# parser.add_argument("--cluster_ignore_norms", action="store_true",
#                     help = "optional flag to not compensate for the norms in the cluster assignment")

# #alignment parameters
# parser.add_argument("--lr", type = float, default = 1e-3)
# parser.add_argument("--lr_multiplier", type = float, default = 1,
#                     help = "optional learning rate multiplier to decay for learning rate scheduler")
# parser.add_argument("--n_iter", type = int, default = 100)
# parser.add_argument("--clip_grad", type = float, default = 1e-1,
#                     help = "optional gradient clipping if -1 no clipping is applied")
# parser.add_argument("--low_bound", type = float, default = 1e-5,
#                     help = "optional lower bound for the loss, stopping after reaching this value")
# parser.add_argument("--patience", type = int, default = 250,
#                     help = "optional patience for early stopping")
# parser.add_argument("--patience_scheduler", type = int, default = 50,
#                     help = "reduce learning rate on plateau scheduler patience")
# parser.add_argument("--verbose", type = int, default = 1,
#                     help = "print after this many iterations")
# parser.add_argument("--seed", type = int, default = 1,)


args = parser.parse_args()

kwargs = yaml.load(open(args.yaml_path, "r"), Loader = yaml.FullLoader)

dtype = torch.float32 if kwargs.get("dtype", "float32") == "float32" else torch.float16
print("dtype", dtype)
seed = kwargs["seed"]
torch.manual_seed(seed)    
np.random.seed(seed)
torch.cuda.manual_seed(seed)


weight = torch.load(os.path.join(args.weights_path,args.hessian_path.split("/")[-2], args.hessian_path.split("/")[-1]),
                    map_location = torch.device(args.device)
                    )["weight"]

original_dtype = weight.dtype

#create the compression module
if kwargs["quantizer_type"] == "LinearVQ":
    compression_module = qc.LinearVQ(
        weight.to(args.device).to(dtype),
        verbose=True
    )
elif kwargs["quantizer_type"] == "LinearVQ_Halving":
    compression_module = qc.LinearVQ_Halving(
        weight.to(args.device).to(dtype),
        verbose=True
    )
else: 
    raise ValueError("quantizer type not recognized")


#load the hessian
hessian = torch.load(args.hessian_path,
                    map_location = torch.device(args.device)
                     )
if "hessian" in hessian:
    compression_module.hessian = hessian["hessian"].to(dtype)
    if kwargs.get("hessian_regularization", 0) > 0:
        diag_mean = compression_module.hessian.diag().mean()
        compression_module.hessian += kwargs["hessian_regularization"] * torch.eye(compression_module.hessian.size(0)).to(args.device) * diag_mean
elif "hessiaDiag" in hessian:
    compression_module.hessianDiag = hessian["hessianDiag"].to(dtype)
    if kwargs.get("hessian_regularization", 0) > 0:
        diag_mean = compression_module.hessianDiag.mean()
        compression_module.hessianDiag += kwargs["hessian_regularization"] * diag_mean
else:
    raise ValueError("hessian not found in hessian file")
    
# print("hessian", compression_module.hessian)
# raise ValueError("stop here")

#hanlding of allocation based bits stuff
if "allocation_config" in kwargs:
    #load the allocation file 
    allocation = yaml.load(open(kwargs["allocation_config"], "r"), Loader = yaml.FullLoader)
    allocation_config = allocation[args.hessian_path.split("/")[-2] + "/" + args.hessian_path.split("/")[-1]]
    
    n_bits = allocation_config["n_bits"]
    #get the corresponding d that can be used and is less than max_d_prod
    d = 1
    print("n_bits", n_bits)
    while d * n_bits <= kwargs.get("max_d_prod", 12) and d <= kwargs.get("max_d", 6):
        print(d,kwargs.get("max_d", 6))
        print(n_bits*d)
        if n_bits*d % 1 < 1e-4 or n_bits*d % 1 > 1 - 1e-4:
            best_d = d 
        d += 1
    
    #update the quantizer kwargs
    kwargs["quantizer_kwargs"]["d"] = best_d
    kwargs["quantizer_kwargs"]["n_bits"] = n_bits
    print("best_d", best_d, "n_bits", n_bits)       
    

#quantize the weights with k-means
compression_module.compress(**kwargs["quantizer_kwargs"])

print("initial loss",compression_module.get_reconstruction_error(compression_module.hessian if hasattr(compression_module, "hessian") else compression_module.hessianDiag).item())

if "alignment_kwargs" in kwargs:
    raise ValueError("alignment not implemented")
else:
    print("best_loss",compression_module.get_reconstruction_error(compression_module.hessian if hasattr(compression_module, "hessian") else compression_module.hessianDiag).item())
print("n_params", compression_module.get_n_original_parameters())
print("n_bits", compression_module.get_n_bits())
print("bpv", compression_module.get_n_bits()/compression_module.get_n_original_parameters())

# compression_module.to(original_dtype)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(compression_module.state_dict(), args.save_path)
#save the args

current_filetype = args.save_path[args.save_path.rfind("."):]
args_save_path = args.save_path[:args.save_path.rfind(".")] + "_args.yaml"
# print("args_save_path", args_save_path)

#save the args as a yaml file
with open(args_save_path, "w") as f:
    #add a arg that these are quantized weights
    kwargs["compression_type"] = str(compression_module)
    yaml.dump(kwargs, f)

#try to creat a 
    


