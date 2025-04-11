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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)


import src.tensor_compress as tc

import argparse
import yaml


parser = argparse.ArgumentParser()

parser.add_argument(
    "--load_path",
    type=str,
    default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians/pajama/128/layer_0/mlp.up_proj.pt",
)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="device to use for training"
)
# qudit parameters
parser.add_argument(
    "--yaml_path",
    type=str,
    default="/data/lliu/huffman/scripts/1layer_compress/tensor_args.yaml",
)

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

args = parser.parse_args()

data = torch.load(args.load_path)
original_dtype = data["weight"].dtype
print("original dtype", original_dtype)

kwargs = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)

print(data.keys())

tensorize_kwargs = kwargs["tensorize_kwargs"]

if tensorize_kwargs["sparse_frac"] > 0:
    compression_module = tc.LinearTensorizedWithSparse(
        data["weight"].to(args.device).to(torch.float32)
    )

    compression_module.tensor_decompose(**tensorize_kwargs)

else:
    compression_module = tc.LinearTensorized(
        data["weight"].to(args.device).to(torch.float32)
    )

    compression_module.tensor_decompose(**tensorize_kwargs)

compression_module.set_additional_attributes_as_trainable()
print(
    "bits",
    compression_module.get_n_bits() / compression_module.get_n_original_parameters(),
)
compression_module.hessian = data["hessian"].to(args.device).to(torch.float32)

alignment_kwargs = kwargs["alignment_kwargs"]

best_loss = compression_module.align(None, **alignment_kwargs)
print("best_loss", best_loss)
print("n_params", compression_module.get_n_original_parameters())
print("n_bits", compression_module.get_n_bits())

if args.save_path is not None:
    # compression_module.to(original_dtype)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(compression_module.state_dict(), args.save_path)
    # save the args

    current_filetype = args.save_path[args.save_path.rfind(".") :]
    args_save_path = args.save_path[: args.save_path.rfind(".")] + "_args.yaml"
    # print("args_save_path", args_save_path)

    # save the args as a yaml file
    with open(args_save_path, "w") as f:
        # add a arg that these are tensorized weights
        kwargs["compression_type"] = "tensorized"
        yaml.dump(kwargs, f)
