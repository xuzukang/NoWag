import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os

print("pid", os.getpid())
import sys
import copy

import yaml
from typing import Tuple, Optional, Union, List, Callable

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)


import src.linear_compress as lc
import src.quantizers.vector_quantizer as vq
import src.quantizers.vq2 as vq2
import wandb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--hessian_path",
    type=str,
    default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/seed_0/pajama/128/layer_0/self_attn.q_proj.pt",
)
parser.add_argument(
    "--weights_path",
    type=str,
    default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/original_weights",
)
parser.add_argument("--save_path", type=str, default="IGNORE")
parser.add_argument("--discrete_update_hessian_path", type=str, default=None)
parser.add_argument(
    "--device", type=str, default="cuda:2", help="device to use for training"
)
parser.add_argument(
    "--yaml_path",
    type=str,
    default="/data/lliu/huffman/scripts/1layer_compress/sparse_args.yaml",
)

args = parser.parse_args()

if "cpu" in args.device:
    args.device = "cpu"


kwargs = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)


dtype = torch.float32 if kwargs.get("dtype", "float32") == "float32" else torch.float16
print("dtype", dtype)
seed = kwargs["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

hessian = torch.load(args.hessian_path, map_location=torch.device(args.device))[
    "hessian"
]
# print("hessian", hessian)
# raise ValueError("stop here")
weight = torch.load(
    os.path.join(
        args.weights_path,
        args.hessian_path.split("/")[-2],
        args.hessian_path.split("/")[-1],
    ),
    map_location=torch.device(args.device),
)["weight"]
# print("weight.device", weight.device)
# print("weight_loaded", weight[0])
original_dtype = weight.dtype
# print("original dtype", original_dtype)
# print("here")
# print("weight_loaded2", weight[0])
# print("weight.to(args.device)",weight.to(args.device)[0])
# print("weight.to(args.device).to(torch.float32)",weight.to(args.device).to(torch.float32)[0])
compression_module = lc.LinearQuantizedSparse(
    weight.to(args.device).to(dtype),
)
compression_module.hessian = hessian.to(args.device).to(dtype)


if "allocation_config" in kwargs:
    # load the allocation file
    allocation = yaml.load(
        open(kwargs["allocation_config"], "r"), Loader=yaml.FullLoader
    )
    allocation_config = allocation[
        args.hessian_path.split("/")[-2] + "/" + args.hessian_path.split("/")[-1]
    ]

    n_bits = allocation_config["n_bits"]
    sparse_frac = allocation_config["sparse_frac"]
    kwargs["sparsify_kwargs"]["frac_sparse"] = sparse_frac
    # get the corresponding d that can be used and is less than max_d_prod
    d = 1
    while d * n_bits <= kwargs.get("max_d_prod", 12) and d <= kwargs.get("max_d", 6):
        if n_bits * d % 1 == 0:
            best_d = d
        d += 1

    # update the quantizer kwargs
    kwargs["quantizer_args"]["quantizer_kwargs"]["d"] = best_d
    kwargs["quantizer_args"]["quantizer_kwargs"]["n_bits"] = n_bits
    print("best_d", best_d, "n_bits", n_bits)


sparse_kwargs = kwargs["sparsify_kwargs"]
normalizer_
compression_module.sparse_only(
    sparse_kwargs["sparse_types"],
    sparse_kwargs["frac_sparse"],
    quantize_minus_sparse=sparse_kwargs["quantize_minus_sparse"],
    sparse_after_norm=sparse_kwargs["sparse_after_norm"],
)


# x = torch.randn(1, weight.size(1), device = args.device, dtype = dtype)

# y1 = compression_module(x)
# print(y1)
# compression_module.cache_reconstruct()
# y2 = compression_module(x)
# print(y2)
# disagree_idxs = torch.where(~torch.isclose(y1, y2))
# print("disagree_idxs", disagree_idxs)
# # assert torch.allclose(y1, y2)
# print("y1", y1[0,disagree_idxs[1]])
# print("y2", y2[0,disagree_idxs[1]])
# save the quantizers
print("best_loss", compression_module.get_reconstruction_error().item())
print("n_params", compression_module.get_n_original_parameters())
print("n_bits", compression_module.get_n_bits())
print(
    "bpv",
    compression_module.get_n_bits() / compression_module.get_n_original_parameters(),
)
# print(compression_module.state_dict())
# compression_module.to(original_dtype)

# state_dict = copy.deepcopy(compression_module.state_dict())

# compression_module_from_scratch = lc.LinearQuantizedSparse(
#     weight.to(args.device).to(dtype),
# )

# compression_module_from_scratch.blank_recreate(
#     vq.VectorQuantizer if kwargs["quantizer_args"].get("quantizer_type", "original")  != "1st_order" else vq2.VectorQuantizer_1st_order,
#     quantizer_kwargs=kwargs["quantizer_args"]["quantizer_kwargs"],
#     sparse_kwargs=kwargs["sparsify_kwargs"]
# )

# compression_module_from_scratch.load_state_dict(state_dict)
# compression_module_from_scratch.hessian = hessian.to(args.device).to(dtype)
# print("from scratch loss",compression_module_from_scratch.get_reconstruction_error().item())


if args.save_path != "IGNORE":
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(compression_module.state_dict(), args.save_path)
    # save the args

    current_filetype = args.save_path[args.save_path.rfind(".") :]
    args_save_path = args.save_path[: args.save_path.rfind(".")] + "_args.yaml"
    # print("args_save_path", args_save_path)

    # save the args as a yaml file
    with open(args_save_path, "w") as f:
        # add a arg that these are quantized weights
        kwargs["compression_type"] = "sparse"
        yaml.dump(kwargs, f)
