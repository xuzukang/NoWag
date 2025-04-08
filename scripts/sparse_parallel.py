import yaml 
import torch 
import sys 
import os 

if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import src.sparse_compress as sparse
import wandb 
import perplexity_eval as ppl_eval
import torch.nn as nn
import tqdm
import torch
import gc
import numpy as np

@torch.no_grad()
def sparse_layer(
                            layer:nn.Module,
                            hessian_path:str,
                            sparse_kwargs:dict,
                            clean:bool = True,
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
        
        device = next(module.parameters()).device
        original_dtype = module.weight.dtype
        module.to(torch.float32)

        new_layer = sparse.SparseLinear(module.weight, module.bias)
        
        if hessian_path != "None":
            # print(f"loading hessian for {name}")
            hessian = torch.load(f"{hessian_path}/{name}.pt", map_location = torch.device(device))
            # print("hessian keys", hessian.keys())   
            if "hessian" in hessian:
                hessian = hessian["hessian"]
                new_layer.hessian = hessian
            elif "hessianDiag" in hessian:
                hessianDiag = hessian["hessianDiag"]
                new_layer.hessianDiag = hessianDiag
            else:
                raise ValueError(f"hessian not found in the hessian file, keys: {hessian.keys()}")
        else:
            hessianDiag = torch.ones(module.weight.shape[1], device = device)
            new_layer.hessianDiag = hessianDiag
            

        new_layer.compress(**sparse_kwargs)

        # new_layer(torch.randn(1, new_layer.in_features).to(device))
        # print(new_layer.reconstruct())
        # print("original", module.weight)   
        # raise ValueError("stop here")
        module.weight.data = new_layer.reconstruct()

        module.to(original_dtype)
        setattr(parent_module, name.split(".")[1], module)
        del new_layer
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
                 cache_reconstruct:bool = False):
    
    layers = model.model.layers
    original_dtype = next(iter(model.parameters())).dtype

    for i in tqdm.tqdm(range(len(layers)), desc="Sparsifying",disable=False):
        layer = sparse_layer(
            layer = layers[i],
            hessian_path = f"{hessian_path}/layer_{i}" if hessian_path != "None" else "None",
            sparse_kwargs = sparse_kwargs,
            clean = clean,
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
    import sys 
    
    sys.stderr = sys.stdout
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, help = "the base model")
    parser.add_argument("--sparse_kwargs_path", type=str, help = "the path to the sparse kwargs")
    parser.add_argument("--hessian_dir", type=str, help = "the path to the hessian_dir",
                        default = "/data/lliu/huffman/models/{model_name}/hessianDiags/seed_0/pajama/128")
    parser.add_argument("--ppl_datasets", type=str, nargs="+",
                        choices=["wikitext2", "c4", "ptb"],
                        help="The datasets to evaluate on.",
                        default=["wikitext2","c4"])
    parser.add_argument("--zero_shot_tasks", type=str, nargs="+",
                        # choices=["boolq","rte","hellaswag","winogrande", "arc_easy", "arc_challenge","openbookqa"],
                        help="The zero shot tasks to evaluate on.",
                        default=["winogrande", "rte", "piqa", "arc_easy", "arc_challenge"])
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llama")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help = "the wandb id so we can resume the run to link it with the compression run")
    parser.add_argument("--seqlen", type=int, default=-1)
    parser.add_argument("--offload_activations", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help = "batch size for the activations, if not specified, we will perform a binary search to fine the optimal batch size")
    parser.add_argument("--save_path", type=str, default = None)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    if args.log_wandb:
        wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow")

    model = ppl_eval.get_llama(args.base_model,
                               device_map = "balanced",
                               dtype = torch.float16)
    args.seqlen = args.seqlen if args.seqlen != -1 else model.config.max_position_embeddings
    model.seqlen = args.seqlen
    print("seqlen", model.seqlen)
    model_name = args.base_model
    # print(checkpoints)
    if args.sparse_kwargs_path is not None:
        model = sparse_model(model,
                                hessian_path = args.hessian_dir.replace("{model_name}", model_name),
                                sparse_kwargs = yaml.load(open(args.sparse_kwargs_path, "r"), Loader=yaml.FullLoader)["sparse_kwargs"],
        )
    else:
        print("Warning: no sparse kwargs path provided, skipping sparsification and evaluating the model as is.")

    model.seqlen = args.seqlen
    model.eval()

    results = {"ppl":{},
               "zero_shot":{}}
    for dataset in args.ppl_datasets:
        print("Evaluating:", dataset)
        testloader = ppl_eval.data.get_loaders(
            dataset, nsamples = 0, seqlen = model.seqlen, model = model_name,
            train_test = "test")
        
        ppl = ppl_eval.llama_eval2(model, testloader, dataset, args.log_wandb,
                     base_model = args.base_model)
        print(f"{dataset} ppl:", ppl)
        
        results["ppl"][dataset] = ppl


    if "None" not in args.zero_shot_tasks:
        import zero_shot as zs
        zero_shot_results = zs.zero_shot(args.base_model, model,
                                            tasks = args.zero_shot_tasks)
        
        #parse the results
        print("results to add to a table:")
        avg_acc = 0
        for task in args.zero_shot_tasks:
            print(task, zero_shot_results[task]["acc"])
            acc = zero_shot_results[task]["acc"]
            if not isinstance(acc, np.float64):
                print("acc is not a float, converting to float")
                acc = acc.item()
            avg_acc += acc
            results["zero_shot"][task] = acc
        print()
        results["zero_shot"]["avg_acc"] = avg_acc / len(args.zero_shot_tasks)
        print("avg acc:", round(avg_acc / len(args.zero_shot_tasks) * 100,2))
        
    print(results)
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        yaml.dump(results, open(os.path.join(args.save_path, "results.yaml"), "w"))
        
        if args.save_model:
            model.save_pretrained(os.path.join(args.save_path, "model"))
            
    
        
    # os.system("rm -rf temp/temp_model")


    