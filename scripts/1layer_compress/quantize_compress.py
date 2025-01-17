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


import src.linear_compress as lc
import src.quantizers.vector_quantizer as vq
import src.quantizers.vq2 as vq2
import wandb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hessian_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/seed_0/pajama/128/layer_0/self_attn.q_proj.pt")
parser.add_argument("--weights_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/original_weights")
parser.add_argument("--save_path", type=str, default="/data/lliu/huffman/test/save_self_attn.q_proj.pt")
parser.add_argument("--discrete_update_hessian_path", type=str, default=None)
parser.add_argument("--device", type = str, default = "cuda:2",
                    help = "device to use for training")
parser.add_argument("--yaml_path", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml")
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

hessian = torch.load(args.hessian_path,
                    map_location = torch.device(args.device)
                     )["hessian"]
# print("hessian", hessian)
# raise ValueError("stop here")
weight = torch.load(os.path.join(args.weights_path,args.hessian_path.split("/")[-2], args.hessian_path.split("/")[-1]),
                    map_location = torch.device(args.device)
                    )["weight"]
# print("weight.device", weight.device)
# print("weight_loaded", weight[0])
original_dtype = weight.dtype
# print("original dtype", original_dtype)
# print("here")
# print("weight_loaded2", weight[0])
# print("weight.to(args.device)",weight.to(args.device)[0])
# print("weight.to(args.device).to(torch.float32)",weight.to(args.device).to(torch.float32)[0])
compression_module = lc.LinearQuantized(
    weight.to(args.device).to(dtype),
)

compression_module.hessian = hessian.to(args.device).to(dtype)
if kwargs.get("hessian_regularization", 0) > 0:
    diag_mean = compression_module.hessian.diag().mean()
    compression_module.hessian += kwargs["hessian_regularization"] * torch.eye(compression_module.hessian.size(0)).to(args.device) * diag_mean
    
# print("hessian", compression_module.hessian)
# raise ValueError("stop here")
compression_module.quantize(
    vq2.VectorQuantizer_1st_order if kwargs.get("quantizer_type", "original") == "1st_order" else vq.VectorQuantizer,
    **kwargs["quantizer_kwargs"]
)
# print(compression_module.get_reconstruction_error().item())
# raise ValueError("stop here")
# wandb.init(project="test_layer_compress", config=kwargs)
# wandb.watch(compression_module,log = "all",
#             log_freq = 1)

if kwargs.get("quantizer_type", "original") == "1st_order":
    # compression_module.quantizer:vq2.VectorQuantizer_1st_order
    if args.discrete_update_hessian_path is not None:
        discrete_hessian = torch.load(os.path.join(args.discrete_update_hessian_path,args.hessian_path.split("/")[-2], args.hessian_path.split("/")[-1]), map_location=torch.device(args.device))["hessian"].to(args.device).to(torch.float32)
        if kwargs.get("quantizer_type", "original") == "1st_order":
            discrete_hessian = compression_module.quantizer.pad_hessian(discrete_hessian)
        print("discrete_hessian", discrete_hessian)
        print("normal_hessian", compression_module.hessian)
    else:
        discrete_hessian = compression_module.hessian
        
        

    discrete_update_kwargs={"hessian":discrete_hessian, "n_parallel": kwargs["quantizer_kwargs"].get("n_parallel", -1)}
    if discrete_update_kwargs["n_parallel"] == -1:
        discrete_update_kwargs["n_parallel"] = compression_module.quantizer.determine_optimal_n_parallel(discrete_hessian)
    # do a discrete update out of the box
    if kwargs.get("initial_discrete_update", False):
        for i in range(kwargs.get("n_discrete_updates", 1)):
            compression_module.update_discrete(**discrete_update_kwargs)
            print("loss:",compression_module.get_reconstruction_error().item())
    # print(compression_module.get_reconstruction_error().item())
else:
    discrete_update_kwargs = {}
#get the initial loss
print("initial loss",compression_module.get_reconstruction_error().item())
if "alignment_kwargs" in kwargs:
    compression_module.set_additional_attributes_as_trainable()
    best_loss = compression_module.align(
        val_hessian=None,
        discrete_update_kwargs=discrete_update_kwargs,
        **kwargs["alignment_kwargs"],
        # lr = args.lr,
        # lr_multiplier = args.lr_multiplier,
        # n_iters = args.n_iter,
        # clip_grad = args.clip_grad,
        # low_bound = args.low_bound,
        # patience = args.patience,
        # patience_scheduler = args.patience_scheduler,
        # verbose = args.verbose,
    )
    if kwargs.get("final_discrete_update", False):
        compression_module.update_discrete(**discrete_update_kwargs)

    print("best_loss", compression_module.get_reconstruction_error().item())
else:
    print("best_loss",compression_module.get_reconstruction_error().item())
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
    kwargs["compression_type"] = "quantized"
    yaml.dump(kwargs, f)
    


