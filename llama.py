import time

import torch
import random
import torch.nn as nn

from vector_quantizer import *
from modelutils import *
from quant import *
import random 
import numpy as np
import fine_tune


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
    model.seqlen = 2048
    print("Model loaded.", model)
    return model


# @torch.no_grad()
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

    print("Ready.")

    quantizers = {}

    total_bits = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        if args.fine_tune:
            if i > 0:
                with torch.no_grad():
                    for j in range(args.nsamples):
                        layers_target_output[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    # layers_target_output = layer(layers_target_output, attention_mask=attention_mask)
                
                layer = fine_tune.finetune_module(layer, inps, layers_target_output, lora=args.lora_fine_tune, lora_kwargs={
                                            "rank": args.lora_rank, "alpha": args.lora_alpha}, n_iters=args.fine_tune_n_iters,
                                            lambda_regul=0.1)
                
                # return 
                    
            
            else:
                with torch.no_grad():
                    print("initial layer")
                    print(inps.shape)
                    print(inps.shape)
                    layers_target_output = torch.zeros_like(inps)
                    for j in range(args.nsamples):
                        layers_target_output[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    print(layers_target_output.shape)   
        # else:
        #     if i > 0:
        #         return
            
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = VectorQuantizerTemp(subset[name])
                if args.wbits < 16:
                    raise Exception("Quantization not supported.")
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            with torch.no_grad():
                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(args.nsamples):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                for h in handles:
                    h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                if args.structured_sparsity:
                    print("Structured sparsity")
                    n_bits, n_params = gpts[name].structured_sparse_quantize(
                        subvector_dim = args.subvector_dim,
                        k_codebook = args.k_cosine_codebook,
                        keep_top_rowise = args.keep_top_rowise if args.keep_top != 0 else 0,
                        keep_top_colwise = args.keep_top_colwise if args.keep_top != 0 else 0,
                        lr = args.lr,   
                        lr_multiple = args.lr_multiple,
                        n_iters = args.n_iters,
                        clamp_gradients = args.clamp_gradients
                )
                    
                elif args.normalized_clustering:
                    print("Normalized clustering")
                    n_bits, n_params = gpts[name].normalized_clustering(
                        subvector_dim = args.subvector_dim,
                        k_codebook = args.k_cosine_codebook,
                        keep_top_rowise = args.keep_top_rowise if args.keep_top != 0 else 0,
                        keep_top_colwise = args.keep_top_colwise if args.keep_top != 0 else 0,
                        lr = args.lr,
                        lr_multiple = args.lr_multiple,
                        n_iters = args.n_iters,
                        clamp_gradients = args.clamp_gradients
                    )
                else:
                    n_bits, n_params = gpts[name].fastquant(
                        subvector_dim = args.subvector_dim,
                        k_magnitude_codebook = args.k_magnitude_codebook,
                        k_cosine_codebook = args.k_cosine_codebook,
                        keep_top = args.keep_top,
                        keep_top_criterion = args.keep_top_criterion,
                        lr = args.lr,
                        lr_multiple = args.lr_multiple,
                        n_iters = args.n_iters,
                        clamp_gradients = args.clamp_gradients
                    )
                gpts[name].free()
                free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
                print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

                total_bits += n_bits
                total_params += n_params

                # return quantizers
        # return quantizers

        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

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
        "--subvector_dim", type=int, default=16, help="Subvector dimension."
    )
    parser.add_argument(
        "--k_magnitude_codebook", type=int, default=256, help="Magnitude codebook size."
    )
    parser.add_argument(
        "--k_cosine_codebook", type=int, default=256, help="Cosine codebook size."
    )
    parser.add_argument(
        "--keep_top", type=float, default=0.01, help="Keep top k subvectors."
    )
    parser.add_argument(
        "--keep_top_criterion", type=str, default=['magnitude', 'hessian'], help="Keep top criterion.", 
        nargs="+"
    )
    parser.add_argument(
        "--lr", type=float, default=10, help="Learning rate for quantization."
    )
    parser.add_argument(
        "--lr_multiple", type=float, default=0.9, help="Learning rate multiple."
    )
    parser.add_argument(
        "--n_iters", type=int, default=100, help="Number of iterations."
    )
    parser.add_argument(
        "--clamp_gradients", type=float, default=0.1, help="Clamp gradients."
    )

    parser.add_argument(
        "--quantize", action="store_true", help="Whether to quantize the model."
    )
    parser.add_argument(
        "--structured_sparsity", action="store_true", help="Whether to use structured sparsity."
    )
    parser.add_argument(
        "--keep_top_rowise", type=float, default=0.45, help="Keep top k rowise."
    )
    parser.add_argument(
        "--keep_top_colwise", type=float, default=0.9, help="Keep top k colwise."
    )
    parser.add_argument(
        "--fine_tune", action="store_true", help="Whether to fine tune the model."
    )
    parser.add_argument(
        "--lora_fine_tune", action="store_true", help="Whether to use LoRA for fine tuning."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=1, help="Rank for LoRA."
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=0.1, help="Alpha for LoRA."
    )
    parser.add_argument(
        "--fine_tune_n_iters", type=int, default=10, help="Number of iterations for fine tuning."
    )
    
    parser.add_argument(
        "--normalized_clustering", action="store_true", help="Whether to use normalized clustering."
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()

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
