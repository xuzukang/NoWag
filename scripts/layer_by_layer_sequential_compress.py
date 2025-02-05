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
from perplexity_eval import load_model_from_checkpoints, load_layer_from_checkpoint

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False



@torch.no_grad()
def compress_model(model, dataloader, dev,
                   checkpoint_yaml_path: str):
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
        (sum(args.nsamples),
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
        if not args.true_sequentail:
            full = find_layers(layer)

            sequential = [list(full.keys())]
        else:
            #this is hardcoded for laziness
            sequential = [
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                ["self_attn.o_proj"],
                ["mlp.gate_proj", "mlp.up_proj"],
                ["mlp.down_proj"]
            ]    

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

            #do a pass to get the hessians for each partition
            i_start = 0
            train_hessians: dict[str: List[torch.Tensor]] = {name: [] for name in names}
            for partition in args.nsamples:
                for name in names:
                    getattr(getattr(layer, name.split(".")[0]), name.split(".")[1]).enable_hessian_logging()
                inference_layer(layer, inps[i_start: i_start + partition], outs[i_start: i_start + partition],
                                layer_kwargs=kwargs,
                                dev=dev, offload_activations=args.offload_activations,
                                batch_size=args.forward_pass_batch_size)
                for name in names:
                    train_hessians[name].append(
                        getattr(
                            getattr(layer, name.split(".")[0]), name.split(".")[1]
                        ).dump_hessian()[0]
                    )
                i_start += partition
            
            #for each partition and each name, save the hessian
            save_paths = {name: [] for name in names}
            for partition in range(len(args.nsamples)):
                for name in names:
                    save_path = os.path.join(args.temp_path, f"partition_{partition}/layer_{i}/{name}.pt")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({"hessian": train_hessians[name][partition]}, save_path)
                    save_paths[name].append(save_path)
            

            
            #use our layer_by_layer_parallel_compress code to compress this layer
            
            parallel_compress_command = "python scripts/layer_by_layer_parallel_compress.py"
            parallel_compress_command += f" --model_to_compress {args.model} --hessian_path {os.path.join(args.temp_path, f'partition_0/')} --weights_path {args.weights_path} --save_path {args.save_path}"

            parallel_compress_command += " --devices"
            for device in args.devices[1:]:
                parallel_compress_command += f" {device}"
            
            if len(args.n_samples) > 1:
                parallel_compress_command += f" --discrete_update_hessian_path {os.path.join(args.temp_path, f'partition_1/')}"
                if len(args.n_samples) > 2:
                    print("warning: only 2 partitions supported the additional partitions will be ignored")
            if args.use_wandb:
                parallel_compress_command += f" --use_wandb --resume_wandb --wandb_id {wandb.run.id} --wandb_project {wandb.run.project} --no_config_update"
            
            print("running command", parallel_compress_command)
            os.system(parallel_compress_command)


            #load the checkpoints.yaml and put it back in the model 

            checkpoint_dict_full = yaml.load(checkpoint_yaml_path, Loader=yaml.FullLoader)
            checkpoint_dict_use = {}
            for checkpoint_name, checkpoint_path in checkpoint_dict_full.items():
                for name in names:
                    if f"layer_{i}/{name}" in checkpoint_name:
                        checkpoint_dict_use[checkpoint_name] = checkpoint_path

            layer, layer_n_bits, layer_n_params = load_layer_from_checkpoint(checkpoint_dict_use,
                                                                             layer,
                                                                             i,

                                                    add_bias = False,
                                                    base_model=args.model,
                                                    key_no_exist_handling="ignore")
        
            layer = model.model.layers[i]


            # pass the inputs through the models:
            outs = inference_layer(layer, inps, outs, 
                            layer_kwargs=kwargs, 
                            dev=dev, offload_activations=args.offload_activations,
                            batch_size=args.forward_pass_batch_size)
            
            total_bits += layer_n_bits
            total_params += layer_n_params

            #delete the hessians
            for name in names:
                for partition in range(len(args.nsamples)):
                    os.remove(save_paths[name][partition])


        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        
        layer.to(torch.device("cpu"))
        for name in names:
            # print(name)
            del utils.recursive_find(layer, name).original_weight
            del utils.recursive_find(layer, name).original_bias
        del layer
        torch.cuda.empty_cache()
        
        
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
        "--devices", type=str, default=["cuda:0"], help="Device to run on.",
        nargs = "+"

    )
    parser.add_argument(
        "--seqlen", type=int, default=1024, help="Sequence length."
    )
    parser.add_argument(
        "--nsamples", type=int, default=[128], help="Number of calibration data samples, can add more to create partitions for a seperate hessian for discrete updates to reduce overfitting.",
        nargs = "+"
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
        "--compression_yaml_params_path",
        type=str,
        default=None,
        help="Path to the compression yaml file.",
    )
    parser.add_argument(
        "--true_sequentail",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use wandb for logging.",
    )
    parser.add_argument(
        "--temp_path",
        type=str,
        default="temp/compression",
        help="Path to save temporary files.",
    )
    parser.add_argument("--weights_path", type = str, default = "/data/lliu/huffman/models/{model_name}/original_weights",)
    parser.add_argument("--save_path", type = str, default = "/data/lliu/huffman/models/{model_name}/compressed",
                        help = "path to save the compressed models")
    parser.add_argument()
    args = parser.parse_args()
    # init W&B logging

    args.base_model = args.model

    if args.use_wandb:
        wandb.init(project="layer_by_layer_compress", reinit=True)
        config = vars(args)
        config["compression_args"] = yaml.load(open(args.compression_yaml_params_path, "r"), Loader = yaml.FullLoader)
        wandb.config.update(args)
        run_name = wandb.run.name
    else:
        run_name = "test"
        

    
    #move the compression_yaml_params to the temp
    #add the run name to the temp path
    args.temp_path = os.path.join(args.temp_path, run_name)
    os.makedirs(args.temp_path, exist_ok=True)
    yaml.dump(yaml.load(open(args.compression_yaml_params_path, "r"), Loader = yaml.FullLoader), open(os.path.join(args.temp_path, "compression_params.yaml"), "w"))

    args.compression_yaml_params_path = os.path.join(args.temp_path, "compression_params.yaml")
    #this way we can modify the compression yaml params without changing the params for the next run

    

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
        raise Exception("validation no longer supported")
        indexs = np.random.permutation(len(dataloader))
        train_idx, val_idx = indexs[args.nsamples_val :], indexs[: args.nsamples_val]
        train_loader = [dataloader[i] for i in train_idx]
        val_loader = [dataloader[i] for i in val_idx]
    else:
        train_loader = dataloader
        val_loader = None
    compress_model(model, train_loader, val_loader, args.devices[0], args.compression_yaml_params_path)
    print("total time taken:", time.time() - tick)
