import yaml 
import torch 
import sys 
import os 

if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import src.linear_compress as lc
import wandb 
import perplexity_eval as ppl_eval
import torch.nn as nn
import tqdm
import torch
import gc
import zero_shot as zs

@torch.no_grad()
def sparse_layer(
                            layer:nn.Module,
                            hessian_path:str,
                            sparse_kwargs:dict,
                            clean:bool = True,
                            device:str = "cpu",
                            device_store:str = "cpu",
                            cache_reconstruct:bool = False):
    
    sublayer_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]
    n_bits, n_params = 0, 0

    for name in tqdm.tqdm(sublayer_names, leave=True, disable=True):

        parent_module = getattr(layer, name.split(".")[0])
        module = getattr(parent_module, name.split(".")[1])
        module.to(device)
        original_dtype = module.weight.dtype
        module.to(torch.float32)

        new_layer = lc.LinearQuantizedSparse(module.weight, module.bias)
        
        hessian = torch.load(f"{hessian_path}/{name}.pt", map_location = torch.device(device))["hessian"]
        new_layer.hessian = hessian

        new_layer.sparse_only(**sparse_kwargs)

        del new_layer.hessian
        # del new_layer.original_weight
        if clean:
            new_layer.clean()

        if cache_reconstruct:
            new_layer.cache_reconstruct()
        else:
            raise ValueError("cache_reconstruct must be True")

        # new_layer(torch.randn(1, new_layer.in_features).to(device))
        
        module.to(device_store)
        del module.weight
        del module.bias
        del hessian

        delattr(parent_module, name.split(".")[1])
        new_layer.to(original_dtype).to(device_store)
        # print(new_layer.cached_reconstruct.device)
        setattr(parent_module, name.split(".")[1], new_layer)
        # getattr(parent_module, name.split(".")[1]).to(device_store)

        torch.cuda.empty_cache()
        #print what remains in the memory
        # print(torch.cuda.memory_allocated(device)/1024**3)
        # #print the tensors that are still in the memory
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             if obj.device == torch.device(device):
        #                 print(type(obj),
        #                     obj.size(), "dtype", obj.dtype,obj.device)
        #     except:
        #         pass
        # print("--------")
        # raise ValueError("stop here")

    return layer


def sparse_model(model, 
                 hessian_path:str, 
                 sparse_kwargs:dict, 
                 clean:bool = True, 
                 device:str = "cpu", 
                 device_store:str = "cpu", 
                 cache_reconstruct:bool = False):
    
    layers = model.model.layers
    original_dtype = next(iter(model.parameters())).dtype

    for i in tqdm.tqdm(range(len(layers)), desc="Sparsifying",disable=False):
        layer = sparse_layer(
            layer = layers[i],
            hessian_path = f"{hessian_path}/layer_{i}",
            sparse_kwargs = sparse_kwargs,
            clean = clean,
            device = device,
            device_store = device_store,
            cache_reconstruct = cache_reconstruct
        )
        layers[i] = layer

        del layer
        torch.cuda.empty_cache()
        #print the memory usage
        # print(torch.cuda.memory_allocated(device)/1024**3)

    model.to(original_dtype)

    return model    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, help = "the base model")
    parser.add_argument("--sparse_kwargs_path", type=str, help = "the path to the sparse kwargs")
    parser.add_argument("--hessian_dir", type=str, help = "the path to the hessian_dir",
                        default = "/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128")
    parser.add_argument("--ppl_datasets", type=str, nargs="+",
                        choices=["wikitext2", "c4", "ptb"],
                        help="The datasets to evaluate on.",
                        default=["wikitext2"])
    parser.add_argument("--zero_shot_tasks", type=str, nargs="+",
                        # choices=["boolq","rte","hellaswag","winogrande", "arc_easy", "arc_challenge","openbookqa"],
                        help="The zero shot tasks to evaluate on.",
                        default=["boolq","rte","hellaswag","winogrande", "arc_easy", "arc_challenge","openbookqa"])
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llama")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help = "the wandb id so we can resume the run to link it with the compression run")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--offload_activations", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help = "batch size for the activations, if not specified, we will perform a binary search to fine the optimal batch size")
    parser.add_argument("--save_path", type=str, default = None)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    if args.log_wandb:
        wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow")

    model = ppl_eval.get_llama(args.base_model)
    model.seqlen = args.seqlen
    model_name = args.base_model

    
    # print(checkpoints)
    model = sparse_model(model,
                            hessian_path = args.hessian_dir.replace("{model_name}", model_name),
                            sparse_kwargs = yaml.load(open(args.sparse_kwargs_path, "r"), Loader=yaml.FullLoader),
                                        device = args.device,
                                        device_store = "cpu",
                                        cache_reconstruct = True
    )

    model.seqlen = args.seqlen
    model.eval()
    #offload the model to cpu
    model = model.to("cpu")

    results = {}
    for dataset in args.ppl_datasets:
        print("Evaluating:", dataset)
        testloader = ppl_eval.data.get_loaders(
            dataset, nsamples = 0, seqlen = model.seqlen, model = model_name,
            train_test = "test")
        
        ppl = ppl_eval.llama_eval(model, testloader, args.device, dataset, args.log_wandb,
                     args.offload_activations, args.batch_size,
                     base_model = args.base_model)
        
        results[dataset] = ppl


    if "None" not in args.zero_shot_tasks:
        results["zero_shot"] = zs.zero_shot(args.base_model, model, device = args.device,
                                            tasks = args.zero_shot_tasks)

    if args.save_path is not None:
        save_path = args.save_path.replace("{model_name}", model_name)
        os.makedirs(save_path, exist_ok=True)

        
        with open(os.path.join(save_path, "results.yaml"), "w") as f:
            yaml.dump(results, f)

        if args.save_model:
            raise NotImplementedError("Saving the model is not implemented yet")



    






