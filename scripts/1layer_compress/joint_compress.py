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


import src.joint_compress as jc
import src.tensor_compress as tc
import src.linear_compress as lc
import src.quantizers.vector_quantizer as vq

import argparse
import yaml

parser = argparse.ArgumentParser()

parser.add_argument(
    "--load_path",
    type=str,
    default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_0/mlp.up_proj.pt",
)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="device to use for training"
)
parser.add_argument(
    "--yaml_path",
    type=str,
    default="/data/lliu/huffman/scripts/1layer_compress/joint_args.yaml",
)

# parser.add_argument("--quantizer_yaml", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml")
# parser.add_argument("--tensorizer_yaml", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/tensor_args.yaml")

args = parser.parse_args()


kwargs = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)
kwargs_alignment_method = kwargs["alignment_method"]
quantizer_kwargs = kwargs["quantizer_kwargs"]
tensorizer_kwargs = kwargs["tensorize_kwargs"]

torch.manual_seed(kwargs["seed"])
np.random.seed(kwargs["seed"])
torch.cuda.manual_seed(kwargs["seed"])

data = torch.load(args.load_path)
original_dtype = data["weight"].dtype
weight = data["weight"].to(args.device).to(torch.float32)
hessian = data["hessian"].to(args.device).to(torch.float32)

compression_module = jc.JointCompressor(weight=weight)
compression_module.hessian = hessian

quantizer_kwargs["quantizer_class"] = vq.VectorQuantizer
print(tensorizer_kwargs)
compression_module.initalize_2_compressors(
    quantization_compression_algorithm=lc.LinearQuantized,
    quantization_kwargs=quantizer_kwargs,
    tensor_compression_algorithm=(
        tc.LinearTensorizedWithSparse
        if tensorizer_kwargs["sparse_frac"] > 0
        else tc.LinearTensorized
    ),
    tensor_compression_kwargs=tensorizer_kwargs,
)
print(
    "bpv",
    compression_module.get_n_bits() / compression_module.get_n_original_parameters(),
)
if "warmup_kwargs" in kwargs:
    warmup_kwargs = kwargs["warmup_kwargs"]
    compression_module.warmup_tensorization(**warmup_kwargs)

alignment_kwargs = kwargs["alignment_kwargs"]
if kwargs["alignment_method"] == "joint":
    best_loss = compression_module.joint_align(**alignment_kwargs)

elif kwargs["alignment_method"] == "alternate":
    best_loss = compression_module.alternating_align(
        **alignment_kwargs,
    )

print("best_loss", best_loss)
print("n_params", compression_module.get_n_original_parameters())
print("n_bits", compression_module.get_n_bits())
print(
    "bpv",
    compression_module.get_n_bits() / compression_module.get_n_original_parameters(),
)

if args.save_path is not None:

    # compression_module.to(original_dtype)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(compression_module.state_dict(), args.save_path)
    # save the args

    import yaml

    current_filetype = args.save_path[args.save_path.rfind(".") :]
    args_save_path = args.save_path[: args.save_path.rfind(".")] + "_args.yaml"
    # print("args_save_path", args_save_path)

    # save the args as a yaml file
    with open(args_save_path, "w") as f:
        quantizer_kwargs["quantizer_class"] = "VectorQuantizer"

        kwargs["compression_type"] = "joint"
        yaml.dump(kwargs, f)
