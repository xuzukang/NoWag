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
import src.finetune as finetune
import src.finetune as finetune
import os
import yaml
import src.quantizers.vector_quantizer as vector_quantizer
import src.tensor_compress as tensor_compress
from src.utils.model_utils import find_layers, get_llama, inference_layer
import src.data as data
import src.utils.utils as utils


try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def finetune_fn(
    layer: nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    val_inps: Optional[torch.Tensor],
    val_outs: Optional[torch.Tensor],
    args,
    discrete_update_fn,
    kwargs: dict,
    dev: str,
):
    layer.to(torch.float32)
    finetune.finetune_amp(
        layer=layer,
        train_inps=inps.to(dev).to(dtype=torch.float32),
        train_outputs=outs.to(dev).to(dtype=torch.float32),
        val_inps=val_inps.to(dev).to(dtype=torch.float32)
        if val_inps is not None
        else None,
        val_outputs=val_outs.to(dev).to(dtype=torch.float32)
        if val_outs is not None
        else None,
        args=args,
        layer_kwargs=kwargs,
        early_stop_eps=1e-6,
        discrete_update_fn=discrete_update_fn,
    )
    # layer.to(torch.float32)
    return layer


@torch.no_grad()
def quantize(model, dataloader, dataloader_val, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples_train,
         model.seqlen, 
         model.config.hidden_size), 
        dtype=dtype, 
        device=dev if not args.offload_activations else "cpu"
    )

    train_cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # print(kwargs)
            # raise Exception("stop")
            inps[train_cache["i"]] = inp if not args.offload_activations else inp.cpu()
            train_cache["i"] += 1
            train_cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="getting inputs", miniters=len(dataloader)//100):
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    if dataloader_val is not None:
        inps_val = torch.zeros(
            (args.nsamples_val, model.seqlen, model.config.hidden_size),
            dtype=dtype,
            device=dev if not args.offload_activations else "cpu",
        )

        val_cache = {"i": 0}
        class Catcher_val(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps_val[val_cache["i"]] = inp if not args.offload_activations else inp.cpu()
                val_cache["i"] += 1
                raise ValueError

        layers[0] = Catcher_val(layers[0])
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader_val, desc="getting val inputs", miniters=len(dataloader_val)//100):
                # model(batch[0].to(dev))
                try:
                    model(batch[0].to(dev))
                except ValueError:
                    pass
        val_outs = torch.zeros_like(inps_val)

        layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()   
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    kwargs = train_cache["kwargs"]
    free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

    for name in kwargs:
        if isinstance(kwargs[name], torch.Tensor):
            kwargs[name] = kwargs[name].to(dev)
            # print(name, kwargs[name].device, kwargs[name].dtype, kwargs[name].shape)


    print("Ready.")


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
            # raise Exception("Not supported yet.")
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]
        print("sequential", sequential)

        for l, names in enumerate(sequential):
            for name in names:
                parent_module = getattr(layer, name.split(".")[0])
                sublayer: nn.Linear = getattr(parent_module, name.split(".")[1])
                print("sublayer", sublayer)
                new_layer = tensor_compress.LinearTensorized(
                    sublayer.weight, sublayer.bias, add_bias=args.add_bias
                )
                new_layer.enable_hessian_logging()
                new_layer.to(dev)
                new_layer.to(sublayer.weight.dtype)

                # delete the old layer
                delattr(parent_module, name.split(".")[1])
                setattr(parent_module, name.split(".")[1], new_layer)

            # garbage collect
            torch.cuda.empty_cache()

            # pass the inputs through the models:
            inference_layer(layer, inps, outs, 
                            layer_kwargs=kwargs, 
                            dev=dev, offload_activations=args.offload_activations,
                            batch_size=args.forward_pass_batch_size)

            # get the hessians
            train_hessians = {}
            for name in names:
                train_hessians[name] = getattr(
                    getattr(layer, name.split(".")[0]), name.split(".")[1]
                ).dump_hessian()[0]
            # raise Exception("stop")
            if dataloader_val is not None:
                print("val")
                # turn back on the hessian logging
                for name in names:
                    getattr(
                        getattr(layer, name.split(".")[0]), name.split(".")[1]
                    ).enable_hessian_logging()

                # pass the inputs through the models:
                inference_layer(
                    layer,
                    inps_val,
                    val_outs,
                    layer_kwargs=kwargs,
                    dev=dev,
                    offload_activations=args.offload_activations,
                    batch_size=args.forward_pass_batch_size,
                )

                # get the hessians
                val_hessians = {}
                for name in names:
                    val_hessians[name] = getattr(
                        getattr(layer, name.split(".")[0]), name.split(".")[1]
                    ).dump_hessian()[0]

            # put the train hessians back in
            for name in names:
                getattr(
                    getattr(layer, name.split(".")[0]), name.split(".")[1]
                ).hessian = train_hessians[name]

            # quantize
            for name in names:
                print(f"layer{i}: {name}")
                new_layer = getattr(
                    getattr(layer, name.split(".")[0]), name.split(".")[1]
                )
                new_layer.tensor_decompose(
                    N_qudits = args.N_qudits,
                    norm_order = args.norm_order,
                )
                new_layer.set_additional_attributes_as_trainable()
                new_layer.to(torch.float32)
                # align
                # print("val_hessians", val_hessians)
                new_layer.align(
                    val_hessian=val_hessians[name]
                    if dataloader_val is not None
                    else None,
                    lr=args.lr,
                    lr_multiplier=args.lr_multiple,
                    n_iters=args.n_iters,
                    val_every=1,
                    discrete_update_every=1,
                    clip_grad=args.clamp_gradients,
                    eps=1e-6,
                    verbose=args.n_iters//10,
                    patience=-1,
                    patience_scheduler = args.patience_scheduler,
                )
                new_layer.clean()
                total_bits += new_layer.get_n_bits()
                total_params += new_layer.get_n_original_parameters()
                # raise Exception("stop")
            if args.fine_tune_quip_like and l != len(sequential) - 1:
                raise Exception("Not supported anymore")

        if args.fine_tune or args.fine_tune_quip_like:
            # set everything to trainable
            # for sublists in sequential:
            #     for name in sublists:
            #         getattr(getattr(layer, name.split(".")[0]), name.split(".")[1]).enable_grad()
            print("Fine tuning ...")
            # switch to float16
            layer = finetune_fn(
                layer=layer,
                inps=inps,
                outs=outs,
                val_inps=inps_val if dataloader_val is not None else None,
                val_outs=val_outs if dataloader_val is not None else None,
                args=args,
                kwargs=kwargs,
                dev=dev,
                discrete_update_fn=utils.update_discrete
                if args.update_discrete_finetune
                else None,
            )

        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        print(
            "trying to convert back to original dtype, current dtype:", layer_dtype_orig
        )
        # layer = cast_to_dtype(layer ,layer_dtype_orig)
        layer = layer.to(dtype=layer_dtype_orig)
        # print("Fine tuned")

        inference_layer(layer, 
                        inps, 
                        outs, 
                        layer_kwargs=kwargs, 
                        dev=dev, 
                        offload_activations=args.offload_activations,
                        batch_size=args.forward_pass_batch_size)   
        if dataloader_val is not None:
            inference_layer(
                layer,
                inps_val,
                val_outs,
                layer_kwargs=kwargs,
                dev=dev,
                offload_activations=args.offload_activations,
                batch_size=args.forward_pass_batch_size,
            )

        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        if args.save_path is not None:
            save_path = os.path.join(args.save_path, f"layer_{i}.pt")
            torch.save(layer.state_dict(), save_path)
            print(f"Saved layer {i} to {save_path}")
        
        layers[i] = layer.to(torch.device("cpu"))
        del layer
        torch.cuda.empty_cache()
        print("after cast to cpu")
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

        inps, outs = outs, inps
        # return
        # break
        print("Done with layer", i, "total_time elapsed:", round(time.time() - tick), "estimated time left:", round((time.time() - tick) * (len(layers) - i - 1) / (i + 1)))
    model.config.use_cache = use_cache

    print("Total bits:", total_bits, "Total params:", total_params)
    print("average bits per value:", total_bits / total_params)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "pajama"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on."
    )
    parser.add_argument(
        "--seqlen", type=int, default=1024, help="Sequence length."
    )
    parser.add_argument(
        "--nsamples_train", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--nsamples_val",
        type=int,
        default=16,
        help="Number of samples to dedicate to validation.",
    )
    parser.add_argument(
        "--forward_pass_batch_size",
        type=int,
        default=4,
        help="Batch size for forward pass, parallel process these many sequences.",
    )
    parser.add_argument("--save_path", 
                        type=str, 
                        default=None, 
                        help="Path to saved model."
    )
    parser.add_argument(
        "--true_sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--N_qudits", type=int, default=4, help="Number of qudits.")
    parser.add_argument(
        "--add_bias", action="store_true", help="Whether to add bias.")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for quantization."
    )
    parser.add_argument(
        "--lr_multiple", type=float, default=1/3, help="Learning rate multiple."
    )
    parser.add_argument(
        "--patience_scheduler",
        type=int,
        default=100,
        help="Patience for the learning rate scheduler.",
    )
    parser.add_argument(
        "--n_iters", type=int, default=2500, help="Number of iterations."
    )
    parser.add_argument(
        "--clamp_gradients", type=float, default=0.1, help="Clamp gradients."
    )
    parser.add_argument(
        "--norm_order", type=int, nargs="+", default=[0, 1], help="Norm order."
    )
    parser.add_argument(
        "--regularization", type=float, default=0, help="Regularization."
    )
    parser.add_argument(
        "--fine_tune", action="store_true", help="Whether to fine tune the model."
    )
    parser.add_argument(
        "--fine_tune_quip_like",
        action="store_true",
        help="Whether to fine tune the model.",
    )
    parser.add_argument(
        "--update_discrete_finetune",
        action="store_true",
        help="Whether to update discrete variables during finetuning.",
    )
    parser.add_argument(
        "--fnn_device",
        type=str,
        default=None,
        help="Whether to put the mlp/fnn on a separate device.",
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
    model.seqlen = args.seqlen
    model.eval()
    # print("n samples val", args.nsamples_val)
    # raise Exception("stop")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        #save the args as a yaml file
        with open(os.path.join(args.save_path, "args.yaml"), "w") as f:
            args_dict = vars(args)
            #add a arg that these are tensorized weights
            args_dict["compression_type"] = "tensorized"
            yaml.dump(args_dict, f)

    dataloader = data.get_loaders(
        args.dataset,
        nsamples=args.nsamples_train + args.nsamples_val,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        train_test="train",
    )
    tick = time.time()
    n_params = sum(p.numel() for p in model.parameters())
    if args.nsamples_val > 0:
        indexs = np.random.permutation(len(dataloader))
        train_idx, val_idx = indexs[args.nsamples_val :], indexs[: args.nsamples_val]
        train_loader = [dataloader[i] for i in train_idx]
        val_loader = [dataloader[i] for i in val_idx]
    else:
        train_loader = dataloader
        val_loader = None
    quantize(model, train_loader, val_loader, args.device)
    print("total time taken:", time.time() - tick)
