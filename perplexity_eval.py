import os
print("pid", os.getpid())

CUDA_LAUNCH_BLOCKING = 1
import time
import gc
import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Literal

# from vector_quantizer import *
import tqdm

# from quant import *
import random
import numpy as np
import src.quantizers.vector_quantizer as vector_quantizer
import src.quantizers.vq2 as vector_quantizer_2
import src.linear_compress as linear_compress
import src.tensor_compress as tensor_compress
import src.joint_compress as joint_compress
from src.utils.model_utils import find_layers, get_llama, inference_layer
from src.utils.quantized_model import load_model_from_checkpoints
import src.data as data
import src.utils.utils as utils
import yaml
import transformers.models.llama.modeling_llama as llama

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False





@torch.no_grad()
def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False,
               offload_activations: bool = False, batch_size: int = 8,
               ram_batch_size:int = -1,
               base_model: str = "llama",
               results_log_path: str = None,
               disable_tqdm: bool = False) -> float:
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    n_samples_at_once = ram_batch_size if ram_batch_size > 0 else nsamples
    print("nsamples", nsamples)
    print("testenc.numel()", testenc.numel())
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    nlls = []
    for j in tqdm.tqdm(range(0,nsamples,n_samples_at_once),desc="big_loop"):
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (n_samples_at_once, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev if not offload_activations else "cpu"
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
        for i in range(n_samples_at_once):
            batch = testenc[:, ((j+i) * model.seqlen) : ((j+i + 1) * model.seqlen)].to(dev)
            try:
                model(batch)
            except ValueError:
                pass
    
        kwargs = cache["kwargs"]
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        #print("-----------inps------------", inps)
        # print("----------------model---------------", model)
        #print("-----------layers-------------", layers)
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.rotary_emb = model.model.rotary_emb.cpu()
        model.model.norm = model.model.norm.cpu()
        # print("-----------rotary_emb------------", model.model.rotary_emb)
        # print("-----------norm------------", model.model.norm)
        torch.cuda.empty_cache()

        # outs = torch.zeros_like(inps)
        outs = None
        for name in kwargs:
            if isinstance(kwargs[name], torch.Tensor):
                kwargs[name] = kwargs[name].to(dev)

        for i in tqdm.tqdm(range(len(layers)), desc="Inference", disable=disable_tqdm):
            # print(i)
            inps = inference_layer(layers[i], inps, outs, kwargs, dev, offload_activations, batch_size,
                                disable_tqdm=True)
            # layers[i] = layer.cpu()
            # del layer
            torch.cuda.empty_cache()
            gc.collect()
            # inps = outs
            # break

        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)
        torch.cuda.empty_cache()
        testenc_use = testenc[:,j*model.seqlen:(j+n_samples_at_once)*model.seqlen].to(dev)
        print("shapes",testenc_use.shape,j)
        for i in range(n_samples_at_once):
            hidden_states = inps[i].unsqueeze(0) if not offload_activations else inps[i].unsqueeze(0).to(dev)
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            # print(lm_logits)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc_use[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
            # print(shift_labels)
            # raise Exception("stop")
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            # print("loss", loss)
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
            del lm_logits
            del shift_labels
            del shift_logits
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        print({f"/perplexity/{base_model}/{dataset}": ppl.item()})
        wandb.log({f"/perplexity/{base_model}/{dataset}": ppl.item()})
    # if results_log_path is not None:
    #     #assume that it is a yaml file
    #     results = yaml.load(open(results_log_path, "r"), Loader = yaml.FullLoader)
        
    #     results[f"{base_model}/{dataset}"] = ppl.item()

    model.config.use_cache = use_cache
    return ppl.item()


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
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--offload_activations", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help = "batch size for the activations, if not specified, we will perform a binary search to fine the optimal batch size")
    parser.add_argument("--results_log_path", type=str, default = None)

    args = parser.parse_args()

    if args.log_wandb:
        wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow")

    model = get_llama(args.base_model)
    model.seqlen = args.seqlen
    model_name = args.base_model
    model.to("cpu")
    if args.checkpoint_list_path:
        
        checkpoints = yaml.load(open(args.checkpoint_list_path,"r"), Loader=yaml.FullLoader)
        # print(checkpoints)
        model,n_bits,n_vals = load_model_from_checkpoints(checkpoints,
                                            args.base_model,
                                                        # lambda x: "joint2" if "self_attn" in x else "quantize",
                                                        model,
                                            log_wandb=args.log_wandb,
                                            device = args.device,
                                            cache_reconstruct = True
        )
        print("bits", n_bits, "params", n_vals)
        print("bps", n_bits/n_vals)
        if args.log_wandb:
            wandb.log({"bpv": n_bits/n_vals})

    model.seqlen = args.seqlen
    model.eval()
    #offload the model to cpu
    model = model.to("cpu")

    for dataset in args.datasets:

        testloader = data.get_loaders(
            dataset, nsamples = 0, seqlen = model.seqlen, model = model_name,
            train_test = "test")
        
        llama_eval(model, testloader, args.device, dataset, args.log_wandb,
                     args.offload_activations, args.batch_size,
                     base_model = args.base_model,
                     results_log_path = args.results_log_path)

    






