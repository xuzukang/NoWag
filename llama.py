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

class QuantizedLinear(nn.Module):
    def __init__(self, quantized_parameters, bias: Optional[nn.Parameter]):
        super().__init__()
        codebooks, mappings, mask, rowise_norms, colwise_norms, sparse_parameters, weight_shape, subvector_assignment = quantized_parameters
        self.out_features, self.in_features = weight_shape
        self.codebooks = nn.Parameter(codebooks.clone(), requires_grad=True)
        self.mappings = nn.Parameter(mappings.clone(), requires_grad=False)
        self.mask = nn.Parameter(~mask.clone(), requires_grad=False)
        self.mask_sum = self.mask.sum().item()
        self.subvector_assignment = nn.Parameter(subvector_assignment.clone(), requires_grad=False)
        self.rowise_norms = nn.Parameter(rowise_norms.clone(), requires_grad=True)
        self.colwise_norms = nn.Parameter(colwise_norms.clone(), requires_grad=True)
        if self.mask_sum > 0:
            self.sparse_parameters = nn.Parameter(sparse_parameters.clone(), requires_grad=True)
        if bias is not None:
            print("here: bias", bias)
            self.bias = bias
        else:
            print("not here")
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=self.codebooks.dtype,
                                                    device=self.codebooks.device
                                                 ), requires_grad=True)
        self.use_checkpoint = False

    def _construct_quantized_weight(self):
        if self.mappings.dtype == self.codebooks.dtype:
            print("current dtypes:", self.mappings.dtype, self.codebooks.dtype, self.subvector_assignment.dtype)
            self.mappings = self.mappings.int()
            self.subvector_assignment = self.subvector_assignment.int()
            self.mask = self.mask.bool()
            print("new dtypes:", self.mappings.dtype, self.codebooks.dtype, self.subvector_assignment.dtype)
        
            
        quantized_weight = torch.zeros(self.out_features, self.in_features, device=self.codebooks.device,
                                       dtype=self.codebooks.dtype)
        
        quantized_weight[:,self.subvector_assignment] = self.codebooks[self.mappings,:].reshape(self.out_features, -1, self.codebooks.shape[-1])
        
        quantized_weight = self.rowise_norms.unsqueeze(0) * self.colwise_norms.unsqueeze(1) * quantized_weight
        if self.mask_sum > 0:
            quantized_weight[self.mask] = self.sparse_parameters
        return quantized_weight
        
            
        
    def _forward(self, input: torch.Tensor):
        # print(input.dtype,self.quantized_weight.dtype)
        # if self.bias is not None:
        #     print(self.bias.dtype)
        self.quantized_weight = self._construct_quantized_weight()
        return F.linear(input, self.quantized_weight, self.bias)

    def forward(self, input: torch.Tensor):
        return self._forward(input)
    
    def to(self, *args, **kwargs):
        print("to: args", args, "kwargs", kwargs)
        self.codebooks = self.codebooks.to(*args, **kwargs)
        self.mappings = self.mappings.to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)
        self.rowise_norms = self.rowise_norms.to(*args, **kwargs)
        self.colwise_norms = self.colwise_norms.to(*args, **kwargs)
        if self.mask_sum > 0:
            self.sparse_parameters = self.sparse_parameters.to(*args, **kwargs)
        if self.bias is not None:
            print(self.bias.dtype)
            print(self.bias)
            self.bias = self.bias.to(*args, **kwargs)
        return super().to(*args, **kwargs)

def cast_to_dtype(module, dtype):   
    #cast the quantized weights to the original dtype
    for name, param in module.named_children():
        if isinstance(param, QuantizedLinear):
            print("casting to dtype", name, dtype)
            print("current dtypes:", param.quantized_weight.dtype)
            if param.bias is not None:
                print(param.bias.dtype)
            param.to(dtype = dtype)
        else:
            cast_to_dtype(param, dtype)
    return module

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
                gpts[name].set_n_samples(args.nsamples)
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
                    # print("j", j)
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
                    weights_new, n_bits, n_params = gpts[name].normalized_clustering(
                        subvector_dim = args.subvector_dim,
                        k_codebook = args.k_cosine_codebook,
                        keep_top_rowise = args.keep_top_rowise if args.keep_top != 0 else 0,
                        keep_top_colwise = args.keep_top_colwise if args.keep_top != 0 else 0,
                        lr = args.lr,
                        lr_multiple = args.lr_multiple,
                        n_iters = args.n_iters,
                        clamp_gradients = args.clamp_gradients
                    )
                    new_linear = QuantizedLinear(weights_new, subset[name].bias)
                    new_linear.to(dev)
                    module = getattr(layer, name.split(".")[0])
                    setattr(module, name.split(".")[1], new_linear)
                    del weights_new
                    # print(name)
                    # # print(getattr(layer, name))
                    # print(layer)
                    
                elif args.low_rank:
                    print("Low rank")
                    n_bits, n_params = gpts[name].low_rank(
                        low_rank_frac = args.low_rank_frac,
                        n_bits = args.lora_n_bits,
                        sparse_rowise = args.keep_top_rowise if args.keep_top != 0 else 0,
                        sparse_colwise = args.keep_top_colwise if args.keep_top != 0 else 0,
                        lr = args.lr,
                        lr_multiplier = args.lr_multiple,
                        n_iters = args.n_iters,
                        grad_clip = args.clamp_gradients
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
        if args.fine_tune:
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
                    print(name, kwargs[name].shape, kwargs[name].device, kwargs[name].dtype)
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
            print("Fine tuned")
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        layers[i] = layer.to(torch.device("cpu"))
        del layer
        del gpts
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
        "--lr", type=float, default=1e-3, help="Learning rate for quantization."
    )
    parser.add_argument(
        "--lr_multiple", type=float, default=0.9, help="Learning rate multiple."
    )
    parser.add_argument(
        "--n_iters", type=int, default=1000, help="Number of iterations."
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
        "--keep_top_rowise", type=float, default=0, help="Keep top k rowise."
    )
    parser.add_argument(
        "--keep_top_colwise", type=float, default=0, help="Keep top k colwise."
    )
    parser.add_argument(
        "--next_layer_finetune", action="store_true", help="Whether to fine tune the model."
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
        "--fine_tune", action="store_true", help="Whether to fine tune the model."
    )
    
    parser.add_argument(
        "--normalized_clustering", action="store_true", help="Whether to use normalized clustering."
    )
    parser.add_argument(
        "--low_rank", action="store_true", help="Whether to use low rank approximation."
    )
    parser.add_argument(
        "--low_rank_frac", type=float, default = 1/16, help="Low rank dimension."
    )
    parser.add_argument(
        "--lora_n_bits", type=int, default=8, help="Number of bits for LoRA."
    )

    parser.add_argument(
        "--finetune_max_epochs",
        type=int,
        default=5,
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
