CUDA_LAUNCH_BLOCKING=1
import time

import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from vector_quantizer import *
from modelutils import *
from quant import *
import random 
import numpy as np
import fine_tune as lora_fine_tune
import src.finetune as finetune
import src.MHA_low_rank as MHA_low_rank
import src.MLP_pruner as MLP_pruner

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 4096
    print("Model loaded.", model)
    return model

def get_n_parameters_and_bits(layer):

    n_bits = 0
    n_params = 0
    for name in ["v_proj", "k_proj", "q_proj", "o_proj"]:
        bits, params =  getattr(getattr(layer, "self_attn"), name).get_n_bits()
        n_bits += bits
        n_params += params
    bits, params = getattr(layer, "mlp").get_n_bits()
    n_bits += bits
    n_params += params
    return n_bits, n_params

def turn_on_batch_add(layer):
    for name in ["v_proj", "k_proj", "q_proj", "o_proj"]:
        getattr(getattr(layer, "self_attn"), name).turn_on_batch_add()
    getattr(layer, "mlp").turn_on_batch_add()

def quantize_layer(layer,args):
    for name in ["v_proj", "k_proj", "q_proj", "o_proj"]:
        getattr(getattr(layer, "self_attn"), name).quantize(d = args.subvector_dim_mha,
                                                            n_centriods = 2**(int(args.bits_per_value_mha*args.subvector_dim_mha)),
                                                            n_iter = args.n_iters_quantize,
                                                            normalize_rowwise = args.normalize_rowise_mha,
                                                            normalize_columnwise = args.normalize_columnwise_mha, 
                                                            diagonal_only = args.diagonal_only_mha)
    getattr(layer, "mlp").quantize(d = args.subvector_dim_mlp,
                                n_centriods = 2**(args.bits_per_value*args.subvector_dim_mlp),
                                n_iter = args.n_iters_quantize,
                                normalize_rowwise = args.normalize_rowise_mlp,
                                normalize_columnwise = args.normalize_columnwise_mlp,
                                diagonal_only = args.diagonal_only_mlp)





@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    print("position_ids", position_ids.shape)
    print("attention_mask", attention_mask.shape)

    print("Ready.")

    quantizers = {}

    total_bits = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer_dtype_orig = next(layer.parameters()).dtype
        print("layer original dtype", layer_dtype_orig)
        
        #first get the forward pass of the layer
        for j in range(args.nsamples):
                    # print("j", j)
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        
        #replace all the self attention modules with low rank modules
        self_attn = getattr(layer, "self_attn")
        #replace all the layers in the self attention module with low rank layers
        setattr(self_attn, "k_proj", MHA_low_rank.Low_Rank_linear(self_attn.k_proj.weight))
        setattr(self_attn, "v_proj", MHA_low_rank.Low_Rank_linear(self_attn.v_proj.weight))
        setattr(self_attn, "q_proj", MHA_low_rank.Low_Rank_linear(self_attn.q_proj.weight))
        setattr(self_attn, "o_proj", MHA_low_rank.Low_Rank_linear(self_attn.o_proj.weight))
        
        #replace the lowrank layers in the mlp module
        mlp = getattr(layer, "mlp")
        setattr(layer, "mlp", MLP_pruner.pruned_feed_forward(mlp.gate_proj.weight, mlp.down_proj.weight, mlp.up_proj.weight)) 
        #turn on the batch addition
        turn_on_batch_add(layer)

        #check that we have replace the layers correctly
        out_new = torch.zeros_like(outs)
        with torch.no_grad():
            for j in range(args.nsamples):
                out_new[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        assert torch.allclose(outs, out_new), "The layers were not replaced correctly, expected the same output"
        
        #perform the low rank approximation
        print("Performing low rank approximation")
        perform_low_rank = lambda s: getattr(getattr(layer, "self_attn"), s).low_rank(
                                    low_rank = args.low_rank,
                                    sparse_rowise = args.keep_top_rowise if args.keep_top != 0 else 0,
                                    sparse_colwise = args.keep_top_colwise if args.keep_top != 0 else 0
                    )
        
        perform_low_rank("k_proj")
        perform_low_rank("v_proj")
        perform_low_rank("q_proj")
        perform_low_rank("o_proj")
        
        #perform the pruning
        print("Pruning ...")
        getattr(layer, "mlp").prune(keep_top = args.keep_top_frac
                                    , keep_bottom = args.keep_bottom_frac
                                    ,damping = args.percdamp
                                    ,add_bias = args.add_bias)
        
        #fine tune once
        layer.to(dtype=torch.float32)
        
        print("Fine tuning ...")
        print(layer)
        print("attempting to cast to float32")
        layer = layer.to(dtype=torch.float32)
        # layer = cast_to_dtype(layer ,torch.float32)
        kwargs = {"attention_mask": attention_mask,
                    "position_ids": position_ids}
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        for name in kwargs:
            if kwargs[name] is not None:
                kwargs[name] = kwargs[name].to(dev)
                if kwargs[name].dtype == torch.float16:
                    kwargs[name] = kwargs[name].to(dtype=torch.float32)
                # print(name, kwargs[name].shape, kwargs[name].device, kwargs[name].dtype)
        finetune.finetune_groupwise(
            layer = layer,
            train_inps = [inps.to(dev).to(dtype=torch.float32)],
            train_outs = [outs.to(dev).to(dtype=torch.float32)],
            devices= [dev],
            args = args,
            valid_inps = None,
            valid_outs = None,
            **kwargs
        )
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        print("trying to convert back to original dtype")
        # layer = cast_to_dtype(layer ,layer_dtype_orig)
        layer = layer.to(dtype=layer_dtype_orig)
        print("Fine tuned once")
        
        #get the number of bits and the number of parameters
        n_bits, n_params = get_n_parameters_and_bits(layer)
        print("total size: ", n_params, "total Megabytes: ", n_bits/(8*1024**2), "bits per value: ", n_bits/n_params)

        #quantize
        print("Quantizing ...")
        #do a forward pass to get the activations
        turn_on_batch_add(layer)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        
        #quantize the layer
        quantize_layer(layer, args)

        #fine tune again
        layer = layer.to(dtype=torch.float32)
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        finetune.finetune_groupwise(
            layer = layer,
            train_inps = [inps.to(dev).to(dtype=torch.float32)],
            train_outs = [outs.to(dev).to(dtype=torch.float32)],
            devices= [dev],
            args = args,
            valid_inps = None,
            valid_outs = None,
            **kwargs
        )
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        print("trying to convert back to original dtype")

        n_bits, n_params = get_n_parameters_and_bits(layer)
        print("total size: ", n_params, "total Megabytes: ", n_bits/(8*1024**2), "bits per value: ", n_bits/n_params)
        total_bits += n_bits
        total_params += n_params

        # layer = cast_to_dtype(layer ,layer_dtype_orig)
        layer = layer.to(dtype=layer_dtype_orig)

        
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        layers[i] = layer.to(torch.device("cpu"))
        del layer
        torch.cuda.empty_cache()
        print("after cast to cpu")
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

        inps, outs = outs, inps
        # break
    model.config.use_cache = use_cache

    print("Total bits:", total_bits, "Total params:", total_params)
    print("average bits per value:", total_bits / total_params)
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            raise Exception("GMP not supported.")
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--low_rank", type=int, default=196, help="Low rank fraction for MHA."
    )
    parser.add_argument(
        "--keep_top_rowise", type=float, default=0, help="Keep top rowise for MHA."
    )
    parser.add_argument(
        "--keep_top_colwise", type=float, default=0, help="Keep top colwise for MHA."
    )
    parser.add_argument(
        "--keep_top_frac", type=float, default=0.25, help="Keep top frac for MLP."
    )
    parser.add_argument(
        "--keep_bottom_frac", type=float, default=0, help="Keep bottom frac for MLP."
    )
    parser.add_argument(
        "--add_bias", action="store_true", help="Add bias for MLP pruning."
    )
    parser.add_argument(
        "--subvector_dim_mha", type=int, default=4, help="Subvector dimension."
    )
    parser.add_argument(
        "--bits_per_value_mha", type=int, default=2, help="Bits per value for MHA."
    )
    parser.add_argument(
        "--normalize_rowise_mha", action="store_true", help="Normalize rowise for MHA."
    )
    parser.add_argument(
        "--normalize_columnwise_mha", action="store_true", help="Normalize columnwise for MHA.",
    )
    parser.add_argument(
        "--diagonal_only_mha", action="store_true", help="Diagonal only for MHA."
    )
    parser.add_argument(
        "--subvector_dim_mlp", type=int, default=4, help="Subvector dimension."
    )
    parser.add_argument(
        "--bits_per_value_mlp", type=int, default=1.75, help="Bits per value for MHA."
    )
    parser.add_argument(
        "--normalize_rowise_mlp", action="store_true", help="Normalize rowise for MHA."
    )
    parser.add_argument(
        "--normalize_columnwise_mlp", action="store_true", help="Normalize columnwise for MHA.",
    )
    parser.add_argument(
        "--diagonal_only_mlp", action="store_true", help="Diagonal only for MHA."
    )  
    parser.add_argument(
        "--finetune_max_epochs",
        type=int,
        default=10,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--finetune_early_stop",
        type=int,
        default=3,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-5,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=1,
        help="(finetuning only) train on batches of this many sequences, globally across all GPUs",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--finetune_adam_beta1",
        type=float,
        default=0.9,
        help="Finetuning adam_beta1",
    )
    parser.add_argument(
        "--finetune_adam_beta2",
        type=float,
        default=0.95,
        help="Finetuning adam_beta2",
    )
    parser.add_argument("--finetune_keep_best", action="store_true")
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=None,
        help="(finetuning only) Per-device and per-forward-pass batch size used to accumulate global --batch_size",
    )
    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()
    print(model.seqlen)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    
    if args.quantize:
        tick = time.time()
        n_params = sum(p.numel() for p in model.parameters())
        llama_sequential(model, dataloader, args.device)
        print(time.time() - tick)

    for dataset in ["wikitext2"]: #, "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, args.device, dataset, args.log_wandb)

    if len(args.save)>0:
        model.save_pretrained(args.save)
