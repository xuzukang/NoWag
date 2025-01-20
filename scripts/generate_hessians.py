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
import os
import sys
if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())
    
    
import yaml
import src.quantizers.vector_quantizer as vector_quantizer
import src.linear_compress as linear_compress
from src.utils.model_utils import find_layers, get_llama, inference_layer
import src.data as data
import src.utils.utils as utils


try:
    import wandb

    has_wandb = True
except:
    has_wandb = False



@torch.no_grad()
def generate_hessians(model, dataloader, dataloader_val, dev, stop_after_first_layer=False):
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


    # raise Exception("stop")

    for name in kwargs:
        if isinstance(kwargs[name], torch.Tensor):
            kwargs[name] = kwargs[name].to(dev)
            # print(name, kwargs[name].device, kwargs[name].dtype, kwargs[name].shape)

    import gc
    # data_ptrs = []   
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             if obj.device == dev:
    #                 data_ptrs.append(obj.data_ptr())
    #     except:
    #         pass

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

        sequential = [list(full.keys())]

        for l, names in enumerate(sequential):
            # if all([os.path.exists(os.path.join(args.save_path, f"layer_{i}/{name}.pt")) for name in names]):
            #     print("skipping layer", i)
            #     inference_layer(layer, inps, outs, 
            #                 layer_kwargs=kwargs, 
            #                 dev=dev, offload_activations=args.offload_activations,
            #                 batch_size=args.forward_pass_batch_size)
                
            #     if dataloader_val is not None:
                    
            #         inference_layer(
            #             layer,
            #             inps_val,
            #             val_outs,
            #             layer_kwargs=kwargs,
            #             dev=dev,
            #             offload_activations=args.offload_activations,
            #             batch_size=args.forward_pass_batch_size,
            #         )
            #     continue
            
            
            for name in names:
                parent_module = getattr(layer, name.split(".")[0])
                sublayer: nn.Linear = getattr(parent_module, name.split(".")[1])
                print("sublayer", sublayer)
                new_layer = linear_compress.LinearQuantized(
                    sublayer.weight, sublayer.bias,
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
                data = {"hessian": train_hessians[name],
                        # "weight": getattr(getattr(layer, name.split(".")[0]), name.split(".")[-1]).original_weight,
                }
                if dataloader_val is not None:
                    data["val_hessian"] = val_hessians[name]
                    del val_hessians[name]
                
                save_path = os.path.join(args.save_path, f"layer_{i}/{name}.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(data, save_path)
                
                train_hessians[name] = None
                if dataloader_val is not None:
                    val_hessians[name] = None
                    del val_hessians[name]
                
                data["hessian"] = None
                data["val_hessian"] = None
                # data["weight"] = None
                del data
                del train_hessians[name]
                torch.cuda.empty_cache()

        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             if obj.data_ptr() not in data_ptrs:
        #                 print(type(obj), obj.shape)
        #             # print(type(obj), obj.shape)
        #     except:
        #         pass

        # layers[i] = layer.to(torch.device("cpu"))
        layer.to(torch.device("cpu"))
        for name in names:
            # print(name)
            del utils.recursive_find(layer, name).original_weight
            del utils.recursive_find(layer, name).original_bias
        del layer
        torch.cuda.empty_cache()
        
        print("stop after first layer", stop_after_first_layer)
        if stop_after_first_layer:
            return
        
        
        print("after cleaning up", i)
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

        # print(torch.cuda.memory_summary(device=args.device, abbreviated=False))

        # import gc
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             if obj.device == dev:
        #                 print(type(obj), obj.shape)
        #             # if obj.data_ptr() not in data_ptrs:
        #             #     print(type(obj), obj.shape)
        #             # print(type(obj), obj.shape)
        #     except:
        #         pass

        # raise Exception("stop")

        inps, outs = outs, inps
        # return
        # break
        print("Done with layer", i, "total_time elapsed:", round(time.time() - tick), "estimated time left:", round((time.time() - tick) * (len(layers) - i - 1) / (i + 1)))
    model.config.use_cache = use_cache

    # print("Total bits:", total_bits, "Total params:", total_params)
    # print("average bits per value:", total_bits / total_params)



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
        "--offload_activations",
        action="store_true",
        help="Offload activations to CPU to save memory.",
    )
    parser.add_argument(
        "--stop_after_first_layer",
        action="store_true",
        help = "Stop after the first layer, used for debugging."
    )
    args = parser.parse_args()
    # init W&B logging


    

    model = get_llama(args.model)
    model.seqlen = args.seqlen
    model.eval()
    model.to("cpu")
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
            yaml.dump(vars(args), f)

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
    generate_hessians(model, train_loader, val_loader, args.device,
                        stop_after_first_layer=args.stop_after_first_layer)
    print("total time taken:", time.time() - tick)
