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
import src.finetune_amp as finetune_amp
import src.quantizers.vector_quantizer as vector_quantizer
import src.utils.alignment.hessian_general_align as hessian_general_align
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
    def __init__(self, quantizer:vector_quantizer.VectorQuantizer, bias: Optional[nn.Parameter],create_bias: bool = True):
        super().__init__()
        self.quantizer = quantizer
        if bias is not None:
            print("here: bias", bias)
            self.bias = bias
        else:
            # print("not here")
            if create_bias:
                self.bias = nn.Parameter(torch.zeros(self.quantizer.n_out,
                                                     dtype=self.quantizer.codebook.dtype,
                                                    device=self.quantizer.codebook.device
                                                 ), requires_grad=True)
            else:
                self.bias = None
        self.use_checkpoint = False
        self.grad_disabled = False
        
        
        
    def forward(self, input: torch.Tensor):
        if self.grad_disabled:
            # print(input.dtype, self.quantized_weight.dtype)
            return F.linear(input, self.quantized_weight, self.bias)
            
        return F.linear(input, self.quantizer(), self.bias)
    
    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        
        self.register_buffer("quantized_weight", self.quantizer())
        self.grad_disabled = True
        self.verbose = True
            
    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True
        
        self.grad_disabled = False
        self.verbose = False
        del self.quantized_weight
            
    def get_n_bits(self):
        n_bits = self.quantizer.get_n_bits()
        if self.bias is not None:
            n_bits += self.bias.numel() *16
        return n_bits
    


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

def finetune_fn(
    layer: nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    val_inps: Optional[torch.Tensor],
    val_outs: Optional[torch.Tensor],
    args,
    kwargs: dict,
    dev:str):
    
    
    layer.to(torch.float32)
    finetune_amp.finetune_amp_eps_wrapper(
        layer = layer,
        train_inps = inps.to(dev).to(dtype = torch.float32),
        train_outputs = outs.to(dev).to(dtype = torch.float32),
        val_inps = val_inps.to(dev).to(dtype = torch.float32) if val_inps is not None else None,
        val_outputs = val_outs.to(dev).to(dtype = torch.float32) if val_outs is not None else None,
        args = args,
        layer_kwargs = kwargs,
        early_stop_eps = 1e-6
        
    )
    layer.to(torch.float32)
    return layer
                

@torch.no_grad()
def llama_sequential(model, dataloader, dataloader_val, dev):
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

    train_cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[train_cache["i"]] = inp
            train_cache["i"] += 1
            train_cache["attention_mask"] = kwargs["attention_mask"]
            train_cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module


    if args.nsamples_val > 0:
        inps_val = torch.zeros(
            (args.nsamples_val, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )

        val_cache = {"i": 0, "attention_mask": None, "position_ids": None}

        class Catcher_val(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps_val[val_cache["i"]] = inp
                val_cache["i"] += 1
                val_cache["attention_mask"] = kwargs["attention_mask"]
                val_cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

        layers[0] = Catcher_val(layers[0])
        with torch.no_grad():
            for batch in dataloader_val:
                try:
                    model(batch[0].to(dev))
                except ValueError:
                    pass
        val_outs = torch.zeros_like(inps_val)

        layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = train_cache["attention_mask"]
    position_ids = train_cache["position_ids"]
    
    kwargs = {"attention_mask": attention_mask,
                      "position_ids": position_ids}
    free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

    for name in kwargs:
        if kwargs[name] is not None:
            kwargs[name] = kwargs[name].to(dev)
            if kwargs[name].dtype == torch.float16:
                kwargs[name] = kwargs[name].to(dtype=torch.float16)
            print(name, kwargs[name].shape, kwargs[name].device, kwargs[name].dtype)
                    
    print("position_ids", position_ids.shape)
    print("attention_mask", attention_mask.shape)

    print("Ready.")

    quantizers = {}

    total_bits = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        # print(layer)
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
        print("sequential", sequential)

        for l,names in enumerate(sequential):
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
            
            def add_batch_val(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch_val(inp[0].data, out.data)

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
                    
                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch_val(name)))

                for j in range(args.nsamples_val):
                    val_outs[j] = layer(inps_val[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    
                for h in handles:
                    h.remove()

            print("finished adding batches")
            print("subset", subset)
            for name in subset:
                print(i, name)
                print("Pruning ...")
                tick = time.time()
                H = gpts[name].get_H()
                H_val = gpts[name].get_H_val()
                W = gpts[name].get_W()
                bias = getattr(subset[name], "bias", None)
                
                n_params = W.numel()
                if bias is not None:
                    n_params += bias.numel()
                    
                vq = vector_quantizer.VectorQuantizer_SampleReassign.quantize(
                    W,
                    H,
                    args.subvector_dim,
                    args.n_bits_per_value,
                    args.n_iters,
                    args.norm_order,
                )
                if args.no_val or args.nsamples_val == 0:
                    H_val = None
                #turn on the additional parameters
                vq.set_additional_attributes_as_trainable()
                #align the quantizer
                vq = hessian_general_align.align(
                    vq, W, H, 
                    H_val, 
                    args.lr, args.lr_multiple, args.n_iters,
                    patience_scheduler= 1,
                    val_every = 1,
                    discrete_update_every= 1,
                    clip_grad = args.clamp_gradients,
                    eps = 1e-6,
                    verbose = args.n_iters//10,
                    patience= 10**6,
                )
                vq.clean()
                # quantizer.Quantize.quantize(W, H,
                #                                         args.subvector_dim,
                #                                         args.k_cosine_codebook,
                #                                         args.n_iters,
                #                                         args.normalize_rowwise,
                #                                         args.normalize_colwise,
                #                                         args.align_quantize,
                #                                         args.regularization,
                #                                         args.lr,
                #                                         args.lr_multiple,
                #                                         align_clip_grad=args.clamp_gradients,
                #                                         damping = args.percdamp,    
                #                                         seed = args.seed,                           
                # )
                
                # new_layer = QuantizedLinear(quantized, getattr(subset[name], "bias", None), create_bias = True)
                # new_layer.disable_grad()
                #first set the existing layer to none
                getattr(layer, name.split(".")[0]).__setattr__(name.split(".")[1], None)
                setattr(getattr(layer, name.split(".")[0]), name.split(".")[1], QuantizedLinear(vq, getattr(subset[name], "bias", None), create_bias = True))
                getattr(getattr(layer, name.split(".")[0]), name.split(".")[1]).disable_grad()                               
                print("quantized in", time.time() - tick, "seconds")                       
                gpts[name].free()
                free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
                print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
                del W
                del H
                total_bits += getattr(getattr(layer, name.split(".")[0]), name.split(".")[1]).get_n_bits()
                total_params += n_params
                # break
                # return quantizers
            print("finished:", subset)
            if args.fine_tune_quip_like and l != len(sequential) - 1:
                layer = finetune_fn(
                    layer = layer, 
                    inps = inps, outs = outs, 
                    val_inps = inps_val if args.nsamples_val > 0 else None, 
                    val_outs = val_outs if args.nsamples_val > 0 else None,
                    args = args, kwargs = kwargs, dev = dev
                )
                
        if args.fine_tune or args.fine_tune_quip_like:
            #set everything to trainable
            for sublists in sequential:
                for name in sublists:
                    getattr(getattr(layer, name.split(".")[0]), name.split(".")[1]).enable_grad()
            print("Fine tuning ...")
            #switch to float16
            layer = finetune_fn(
                layer = layer, 
                inps = inps, outs = outs, 
                val_inps = inps_val if args.nsamples_val > 0 else None, 
                val_outs = val_outs if args.nsamples_val > 0 else None,
                args = args, kwargs = kwargs, dev = dev
            )
            
            
            # layer = layer.to(dtype=torch.float32)
            
            
            # finetune_new.finetune_amp_eps_wrapper(
            #     layer = layer,
            #     train_inps = inps.to(dev).to(dtype = torch.float32),
            #     train_outputs = outs.to(dev).to(dtype = torch.float32),
            #     args = args,
            #     layer_kwargs = kwargs,
            #     early_stop_eps = 1e-6
            # )
            # free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        print("trying to convert back to original dtype, current dtype:", layer_dtype_orig)
        # layer = cast_to_dtype(layer ,layer_dtype_orig)
        layer = layer.to(dtype=layer_dtype_orig)
        # print("Fine tuned")
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for j in range(args.nsamples_val):  
                val_outs[j] = layer(inps_val[j].unsqueeze(0), attention_mask=attention_mask)[0]

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
        # return 
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
        "--nsamples_val",
        type=int,
        default=16,
        help="Number of validation data samples.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0,
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
        "--true_sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--subvector_dim", type=int, default=4, help="Subvector dimension."
    )
    parser.add_argument(
        "--n_bits_per_value", type=int, default=2, help="Number of bits per value."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for quantization."
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
        "--norm_order", type=int, nargs="+", default=[0,1], help="Norm order."
    )
    parser.add_argument(
        "--align_quantize", action="store_true", help="Whether to align quantization."
    )
    parser.add_argument(
        "--regularization", type=float, default=0, help="Regularization."
    )
    
    parser.add_argument(
        "--fine_tune", action="store_true", help="Whether to fine tune the model."
    )
    parser.add_argument(
        "--fine_tune_quip_like", action="store_true", help="Whether to fine tune the model."
    )
    parser.add_argument(
        "--fnn_device", type=str, default=None, help="Whether to put the mlp/fnn on a separate device."
    )
    parser.add_argument(
        "--no_val", action="store_true", help="Whether to use validation hessian."
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=5,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--finetune_early_stop",
        type=int,
        default=5,
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

    dataloader, valloader, testloader = get_loaders(
        args.dataset, nsamples_train=args.nsamples,
        nsamples_val=args.nsamples_val,
          seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    
    if args.quantize:
        tick = time.time()
        n_params = sum(p.numel() for p in model.parameters())
        llama_sequential(model, dataloader, valloader, args.device)
        print("total time taken:", time.time() - tick)

    for dataset in ["wikitext2"]: #, "ptb", "c4"]:
        dataloader, valloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, args.device, dataset, args.log_wandb)

    if len(args.save)>0:
        model.save_pretrained(args.save)
