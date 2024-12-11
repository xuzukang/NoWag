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


import src.joint_compress as jc
import src.tensor_compress as tc
import src.linear_compress as lc
import src.quantizers.vector_quantizer as vq

import argparse
import yaml

parser = argparse.ArgumentParser()

parser.add_argument("--load_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians/pajama/128/layer_3/self_attn.q_proj.pt")
parser.add_argument("--save_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/joint/pajama/128/layer_3/self_attn.q_proj.pt")
parser.add_argument("--device", type = str, default = "cuda:0",
                    help = "device to use for training")

parser.add_argument("--quantizer_yaml", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml")
parser.add_argument("--tensorizer_yaml", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/tensor_args.yaml")

args = parser.parse_args()

quantizer_kwargs = yaml.load(open(args.quantizer_yaml, "r"), Loader = yaml.FullLoader)
tensorizer_kwargs = yaml.load(open(args.tensorizer_yaml, "r"), Loader = yaml.FullLoader)

torch.manual_seed(quantizer_kwargs["seed"])
np.random.seed(quantizer_kwargs["seed"])
torch.cuda.manual_seed(quantizer_kwargs["seed"])

data = torch.load(args.load_path)
original_dtype = data["weight"].dtype
weight = data["weight"].to(args.device).to(torch.float32)
hessian = data["hessian"].to(args.device).to(torch.float32)

compression_module = jc.JointCompressor(
    weight = weight)
compression_module.hessian = hessian

quantizer_kwargs["quantizer_class"] = vq.VectorQuantizer
print(tensorizer_kwargs)
best_loss = compression_module.compress(
    quantization_compression_algorithm = lc.LinearQuantized,
    quantization_kwargs=quantizer_kwargs,
    quantization_align_kwargs=quantizer_kwargs,
    tensor_compression_algorithm = tc.LinearTensorizedWithSparse if tensorizer_kwargs["sparse_frac"] > 0 else tc.LinearTensorized,
    tensor_compression_kwargs=tensorizer_kwargs,
    tensor_compression_align_kwargs=tensorizer_kwargs,
)

print("best_loss", best_loss)
print("n_params", compression_module.get_n_original_parameters())
print("n_bits", compression_module.get_n_bits())

# compression_module.to(original_dtype)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(compression_module.state_dict(), args.save_path)
#save the args


import yaml
current_filetype = args.save_path[args.save_path.rfind("."):]
args_save_path = args.save_path[:args.save_path.rfind(".")] + "_args.yaml"
# print("args_save_path", args_save_path)

#save the args as a yaml file
with open(args_save_path, "w") as f:
    args_dict = {}
    quantizer_kwargs["quantizer_class"] = "VectorQuantizer"
    args_dict["quantizer_kwargs"] = quantizer_kwargs
    args_dict["tensorizer_kwargs"] = tensorizer_kwargs
    #add a arg that these are quantized weights
    args_dict["compression_type"] = "joint"
    yaml.dump(args_dict, f)