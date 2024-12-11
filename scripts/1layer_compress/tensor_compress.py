
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
print("pid", os.getpid())
import sys
from typing import Tuple, Optional, Union, List, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


import src.tensor_compress as tc

import argparse



parser = argparse.ArgumentParser()

parser.add_argument("--load_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians/pajama/128/layer_0/self_attn.q_proj.pt")
parser.add_argument("--save_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/tensorized/pajama/128/layer_0/self_attn.q_proj.pt")
parser.add_argument("--device", type = str, default = "cuda:0",
                    help = "device to use for training")
#qudit parameters
parser.add_argument("--N_qudits", type = int, default = 3)
parser.add_argument("--fixed_qudits_shapes", type = int, nargs = "+", default = [],
                    help = "optional option to fix on or more of the qudit shapes. If not provided, the shapes are automatically determined")
parser.add_argument("--norm_order", type = int, nargs = "+", default = [0,1],)
parser.add_argument("--sparse_frac", type = float, default = 0.0,
                    help = "fraction of the qudit shapes that are set to zero")
#alignment parameters
parser.add_argument("--lr", type = float, default = 1e-2)
parser.add_argument("--lr_norms", type = float, default = None,
                    help = "optional different learning rate for the norms")
parser.add_argument("--lr_multiplier", type = float, default = 1/3,
                    help = "optional learning rate multiplier to decay for learning rate scheduler")
parser.add_argument("--n_iter", type = int, default = 2500)
parser.add_argument("--n_iters_warmup_task", type = int, default = 100,
                    help = "number of iterations to warm up on aligning with the weight matrix only, better perfomance and faster")
parser.add_argument("--clip_grad", type = float, default = 1e-1,
                    help = "optional gradient clipping if -1 no clipping is applied")
parser.add_argument("--low_bound", type = float, default = 1e-5,
                    help = "optional lower bound for the loss, stopping after reaching this value")
parser.add_argument("--patience", type = int, default = 250,
                    help = "optional patience for early stopping")
parser.add_argument("--patience_scheduler", type = int, default = 50,
                    help = "reduce learning rate on plateau scheduler patience")
parser.add_argument("--verbose", type = int, default = 1e10,
                    help = "print after this many iterations")
parser.add_argument("--seed", type = int, default = 0,)

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

args = parser.parse_args()

data = torch.load(args.load_path)
original_dtype = data["weight"].dtype
print("original dtype", original_dtype)

print(data.keys())  

if args.sparse_frac > 0:
    compression_module = tc.LinearTensorizedWithSparse(
        data["weight"].to(args.device).to(torch.float32))
    
    compression_module.tensor_decompose(N_qudits=args.N_qudits, fixed_qudits_shapes=args.fixed_qudits_shapes,
                                        norm_order = args.norm_order, sparsity = args.sparse_frac)
    
else:
    compression_module = tc.LinearTensorized(
        data["weight"].to(args.device).to(torch.float32)
    )
    
    compression_module.tensor_decompose(N_qudits=args.N_qudits, fixed_qudits_shapes=args.fixed_qudits_shapes,
                                        norm_order = args.norm_order)

compression_module.set_additional_attributes_as_trainable()
print("bits", compression_module.get_n_bits()/compression_module.get_n_original_parameters())
compression_module.hessian = data["hessian"].to(args.device).to(torch.float32)
print(args.n_iters_warmup_task, args.n_iter)
best_loss = compression_module.align(None,
                        lr = args.lr,
                        lr_norms = args.lr_norms,
                        lr_multiplier=args.lr_multiplier,
                        n_iters = args.n_iter,
                            n_iters_warmup_task = args.n_iters_warmup_task,
                            clip_grad = args.clip_grad,
                            verbose = args.verbose,
                            low_bound = args.low_bound,
                            patience=args.patience,
                            patience_scheduler=args.patience_scheduler,
)

# compression_module.to(original_dtype)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(compression_module.state_dict(), args.save_path)
#save the args

print("best_loss", best_loss)
print("n_params", compression_module.get_n_original_parameters())
print("n_bits", compression_module.get_n_bits())



import yaml
current_filetype = args.save_path[args.save_path.rfind("."):]
args_save_path = args.save_path[:args.save_path.rfind(".")] + "_args.yaml"
# print("args_save_path", args_save_path)

#save the args as a yaml file
with open(args_save_path, "w") as f:
    args_dict = vars(args)
    #add a arg that these are tensorized weights
    args_dict["compression_type"] = "tensorized"
    yaml.dump(args_dict, f)





