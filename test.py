import torch
import torch.nn as nn   


import src.joint_compress as joint_compress
import src.tensor_compress as tensor_compress
import src.linear_compress as linear_compress
import src.quantizers.vector_quantizer as vector_quantizer

import yaml
import src.alignment.hessian_general_align as hessian_general_align

checkpoint_args = yaml.load(open("/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/flowing-gorge-33/layer_0/mlp.up_proj/compressed_args.yaml", "r"), Loader = yaml.FullLoader)
device = "cuda:0"


new_layer = joint_compress.JointCompressor(
    torch.zeros((4096,11008),device = device), None, False
)
checkpoint_args["tensorize_kwargs"]["pad_method"] = "pad_larger"
print(checkpoint_args["tensorize_kwargs"])
checkpoint_args["quantizer_kwargs"]["quantizer_class"] = vector_quantizer.VectorQuantizer
new_layer.blank_recreate(
    linear_compress.LinearQuantized, checkpoint_args["quantizer_kwargs"],
    tensor_compress.LinearTensorizedWithSparse if checkpoint_args["tensorize_kwargs"]["sparse_frac"] > 0 else tensor_compress.LinearTensorized,
    checkpoint_args["tensorize_kwargs"],
)

new_layer.clean()
del new_layer.original_weight
# state_dict = torch.load("models/meta-llama/Llama-2-7b-hf/compressed/flowing-gorge-33/layer_0/mlp.up_proj/compressed.pt")
# print(state_dict.keys())
# print(state_dict["tensor_compressor.gates.0"][0,0,0])
# raise Exception
# new_layer.load_state_dict(state_dict, strict=False)
new_layer.to(torch.float32)
print(new_layer.tensor_compressor.gates[0][0,0,0])
# raise ValueError("stop")
new_layer.tensor_compressor.safe_forward = False
new_layer(torch.zeros((1,4096,11008)).to(device))

# data = torch.load("/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_0/mlp.up_proj.pt")

# hessian = data["hessian"].to(device).to(torch.float32)
# weight = data["weight"].to(device).to(torch.float32)
# print("weight", weight)
# print("new_layer.reconstruct()", new_layer.reconstruct())
# print("loss:", hessian_general_align.loss(new_layer.reconstruct(), weight, hessian).item())