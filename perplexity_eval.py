import os
print("pid", os.getpid())

CUDA_LAUNCH_BLOCKING = 1
import time

import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union

# from vector_quantizer import *
import tqdm

# from quant import *
import random
import numpy as np
import src.quantizers.vector_quantizer as vector_quantizer
import src.linear_compress as linear_compress
import src.tensor_compress as tensor_compress
import src.joint_compress as joint_compress
from src.utils.model_utils import find_layers, get_llama, inference_layer
import src.data as data
import src.utils.utils as utils
import yaml
import transformers.models.llama.modeling_llama as llama

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False

class args_load:
    def __init__(self, yaml_path:str):
        with open(yaml_path, "r") as f:
            self.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))

    @staticmethod
    def load(yaml_path:str):
        return args_load(yaml_path)

def load_model_from_checkpoints(
                                checkpoints:dict[str:str],
                                model:llama.LlamaForCausalLM,
                                add_bias:bool = False,
                                ) -> llama.LlamaForCausalLM:
    """
    Load a model from a checkpoint of each individual layer.

    Args:
        checkpoints (dict[str:str]): A dictionary of the form {layer_name: checkpoint_path}
        model (llama.LlamaForCausalLM): The model to load the checkpoints into.
        add_bias (bool, optional): Whether to add a bias to the model. Defaults to False. turn to true for 

    """

    # args = args_load(os.path.join(checkpoint_path, "args.yaml"))
    # if not hasattr(args,"compression_type"):
    #     #assume that the model is quantized
    #     args.compression_type = "quantized"
    
    n_bits = 0
    n_params = 0
    
    layers = model.model.layers
    original_dtype = next(iter(model.parameters())).dtype

    sublayer_names = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]

    for i in tqdm.tqdm(range(len(layers))):
        layer = layers[i]
        for name in tqdm.tqdm(sublayer_names, leave=False):
            parent_module = getattr(layer, name.split(".")[0])
            module = getattr(parent_module, name.split(".")[1])

            sublayer_full_name = f"layer_{i}/{name}"
            if sublayer_full_name not in checkpoints:
                raise ValueError(f"Checkpoint for {sublayer_full_name} not found in keys: {checkpoints.keys()}")
            checkpoint_path = checkpoints[sublayer_full_name]
            checkpoint_args = yaml.load(open(checkpoint_path.replace(".pt", "_args.yaml"),"r"), Loader=yaml.FullLoader)
            compression_type = checkpoint_args["compression_type"]
            if compression_type == "quantized":
                new_layer = linear_compress.LinearQuantized(
                    module.weight, module.bias, add_bias
                )
                new_layer.blank_recreate(
                    vector_quantizer.VectorQuantizer, **checkpoint_args["quantizer_kwargs"]
                )
            if compression_type == "tensorized":
                tensorized_kwargs = checkpoint_args["tensorize_kwargs"]
                if tensorized_kwargs["sparse_frac"] > 0:
                    new_layer = tensor_compress.LinearTensorizedWithSparse(
                        module.weight, module.bias, add_bias
                    )
                else:
                    new_layer = tensor_compress.LinearTensorized(
                        module.weight, module.bias, add_bias
                    )
                new_layer.blank_recreate(
                    **checkpoint_args["tensorize_kwargs"]
                )
            elif compression_type == "joint":
                new_layer = joint_compress.JointCompressor(
                    module.weight, module.bias, add_bias
                )
                checkpoint_args["quantizer_kwargs"]["quantizer_class"] = vector_quantizer.VectorQuantizer
                new_layer.blank_recreate(
                    linear_compress.LinearQuantized, checkpoint_args["quantizer_kwargs"],
                    tensor_compress.LinearTensorizedWithSparse if checkpoint_args["tensorize_kwargs"]["sparse_frac"] > 0 else tensor_compress.LinearTensorized,
                    checkpoint_args["tensorize_kwargs"],
                )
            
            new_layer.clean()
            del new_layer.original_weight
            new_layer.load_state_dict(torch.load(checkpoint_path), strict=False)
            n_bits += new_layer.get_n_bits()
            n_params += new_layer.get_n_original_parameters()
            delattr(parent_module, name.split(".")[1])
            setattr(parent_module, name.split(".")[1], new_layer)
            

    print("bpv", n_bits / n_params)
    if args.log_wandb:
        wandb.log({"bpv": n_bits / n_params})
    model.to(original_dtype)

    return model





@torch.no_grad()
def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False,
               offload_activations: bool = False, batch_size: int = 8):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    print("nsamples", nsamples)
    print("testenc.numel()", testenc.numel())
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev if not offload_activations else "cpu"
    )
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp if not offload_activations else inp.cpu()
            cache["i"] += 1
            cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    kwargs = cache["kwargs"]
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # print("inps", inps)
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    for name in kwargs:
        if isinstance(kwargs[name], torch.Tensor):
            kwargs[name] = kwargs[name].to(dev)

    for i in tqdm.tqdm(range(len(layers))):
        # print(i)
        layer = layers[i].to(dev)
        outs = inference_layer(layer, inps, outs, kwargs, dev, offload_activations, batch_size,
                               disable_tqdm=True)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm.tqdm(range(nsamples)):
        hidden_states = inps[i].unsqueeze(0) if not offload_activations else inps[i].unsqueeze(0).to(dev)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        # print(lm_logits)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        # print(shift_labels)
        # raise Exception("stop")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        # print("loss", loss)
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{args.base_model}/{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, help = "the base model")
    parser.add_argument("--checkpoint_list_path", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=["wikitext2", "c4", "ptb"],
                        help="The datasets to evaluate on.",
                        default=["wikitext2", "c4"])
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llama")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help = "the wandb id so we can resume the run to link it with the compression run")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--offload_activations", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help = "batch size for the activations, if not specified, we will perform a binary search to fine the optimal batch size")
    

    args = parser.parse_args()

    if args.log_wandb:
        wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow")

    model = get_llama(args.base_model)
    model.seqlen = args.seqlen
    model_name = args.base_model
    if args.checkpoint_list_path:
        
        checkpoints = yaml.load(open(args.checkpoint_list_path,"r"), Loader=yaml.FullLoader)
        print(checkpoints)
        model = load_model_from_checkpoints(checkpoints,
                                                        # lambda x: "joint2" if "self_attn" in x else "quantize",
                                                        model)

    model.seqlen = args.seqlen
    model.eval()
    #offload the model to cpu
    model = model.to("cpu")

    for dataset in args.datasets:

        testloader = data.get_loaders(
            dataset, nsamples = 0, seqlen = model.seqlen, model = model_name,
            train_test = "test")
        
        llama_eval(model, testloader, args.device, dataset, args.log_wandb,
                     args.offload_activations, args.batch_size)

    






