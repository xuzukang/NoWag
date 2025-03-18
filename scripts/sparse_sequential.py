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
import copy
import zero_shot as zs
from src.utils.model_utils import inference_layer
import src.data as data

@torch.no_grad()
def sparse_layer(
                            layer:nn.Module,
                            sparse_kwargs:dict,
                            inps:torch.FloatTensor,
                            outs:torch.FloatTensor,
                            layer_kwargs:dict = {},
                            device:str = "cpu",
                            offload_activations:bool = False,
                            cache_reconstruct:bool = False):

    print("device:", device)    
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

    original_devices_dtypes = {} #the original devices of the layers
    
    layer_copy = copy.deepcopy(layer)
    for name in sublayer_names:
        parent_module = getattr(layer_copy, name.split(".")[0])
        linear_layer = getattr(parent_module, name.split(".")[1])

        original_device = linear_layer.weight.device
        original_dtype = linear_layer.weight.dtype

        linear_layer.to(device)
        linear_layer.to(torch.float32)

        sparse_layer = sparse.SparseLinear(linear_layer.weight, linear_layer.bias)
        sparse_layer.enable_hessian_logging()
    
        #set the layer to be the new layer 
        setattr(parent_module, name.split(".")[1], sparse_layer)
        original_devices_dtypes[name] = (original_device, original_dtype)
    layer_copy.to(device)
    #log the hessian
    _ = inference_layer(layer_copy, inps, outs,
                        layer_kwargs=layer_kwargs,
                        dev = device, 
                        offload_activations = offload_activations,
                        batch_size = 8,
                        inplace=False
    )

    for name in sublayer_names:
        parent_module = getattr(layer_copy, name.split(".")[0])
        sparse_layer = getattr(parent_module, name.split(".")[1])
        sparse_layer.compress(**sparse_kwargs)
        sparse_layer.clean()
        
        original_layer = getattr(getattr(layer, name.split(".")[0]), name.split(".")[1])
        original_layer.weight.data = sparse_layer.reconstruct().to(original_devices_dtypes[name][1]).to(original_devices_dtypes[name][0])

    del layer_copy
        
    original_dtype = next(iter(layer.parameters())).dtype
    layer.to(torch.float32)
    layer.to(device)
    
    #inference the layer again
    outs = inference_layer(layer, inps, outs,
                        layer_kwargs=layer_kwargs,
                        dev = device, 
                        offload_activations = offload_activations,
                        batch_size = 8,
                        inplace=False
    )

    layer.to(original_dtype)
    layer.to("cpu")

    return layer, outs


def sparse_model(model, 
                 dataloader,
                 sparse_kwargs:dict, 
                 clean:bool = True, 
                 device:str = "cpu", 
                 cache_reconstruct:bool = False,
                 offload_activations:bool = False,
):
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader),
         model.seqlen, 
         model.config.hidden_size), 
        dtype=dtype, 
        device="cpu"
    )

    train_cache = {"i": 0, "kwargs": None}

    outs = torch.zeros_like(inps)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # print(kwargs)
            # raise Exception("stop")
            inps[train_cache["i"]] = inp.cpu()
            train_cache["i"] += 1
            train_cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="getting inputs", miniters=len(dataloader)//100):
            try:
                model(batch[0].to(device))
            except ValueError:
                pass
    layers[0] = layers[0].module
    kwargs = train_cache["kwargs"]
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()   
    torch.cuda.empty_cache()

    print("ready to sparsify")


    layers = model.model.layers
    original_dtype = next(iter(model.parameters())).dtype

    inps = inps.to(torch.float32)
    outs = outs.to(torch.float32)

    for i in tqdm.tqdm(range(len(layers)), desc="Sparsifying",disable=False):
        layer, outs = sparse_layer(
            layer = layers[i],
            inps = inps,
            outs = outs,
            sparse_kwargs = sparse_kwargs,
            device = device,
            cache_reconstruct = cache_reconstruct,
            offload_activations = offload_activations,
            layer_kwargs = kwargs
        )
        layers[i] = layer

        del layer
        torch.cuda.empty_cache()
        inps = outs
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
    parser.add_argument("--calibration_dataset", type=str, help = "the calibration dataset",
                        choices=["wikitext2", "c4", "ptb","pajama"],
                        default="pajama")
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--nsamples", type=int, default = 128)
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
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--offload_activations", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help = "batch size for the activations, if not specified, we will perform a binary search to fine the optimal batch size")
    parser.add_argument("--save_path", type=str, default = None)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_and_load_model", action="store_true")
    parser.add_argument("--save_and_load_temp_path", type=str, default = "temp/temp_model")

    args = parser.parse_args()

    if args.log_wandb:
        wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow")

    model = ppl_eval.get_llama(args.base_model)
    model.seqlen = args.seqlen
    model_name = args.base_model
    model.to("cpu")

    dataloader = data.get_loaders(
        args.calibration_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.base_model,
        seqlen=model.seqlen,
        train_test="train",
    )

    # print(checkpoints)
    model = sparse_model(model,
                            dataloader,
                            sparse_kwargs = yaml.load(open(args.sparse_kwargs_path, "r"), Loader=yaml.FullLoader)["sparse_kwargs"],
                                        device = args.device,
                                        offload_activations = args.offload_activations, 
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

    if args.save_and_load_model:
        #save the model to a temp file
        #count the number of models in the temp folder
        # n_models = len(os.listdir("temp/temp_model"))
        # temp_path = "temp/temp_model/model_" + str(n_models)
        model.save_pretrained(args.save_and_load_temp_path)
        #load the model from the temp file
        model = ppl_eval.get_llama(args.base_model, model_path = args.save_and_load_temp_path,
                                   device_map = "auto")
        model.seqlen = args.seqlen
        model.eval()


    if "None" not in args.zero_shot_tasks:
        results["zero_shot"] = zs.zero_shot(args.base_model, model, device = args.device,
                                            tasks = args.zero_shot_tasks)
        
        #parse the results
        print("results to add to a table:")
        avg_acc = 0
        for task in args.zero_shot_tasks:
            print(round(results[task]["acc"] * 100,2), end = " & ")
            avg_acc += results[task]["acc"]
        print()
        print("avg acc:", round(avg_acc / len(args.zero_shot_tasks) * 100,2))
        

    if args.save_path is not None:
        save_path = args.save_path.replace("{model_name}", model_name)
        os.makedirs(save_path, exist_ok=True)

        
        with open(os.path.join(save_path, "results.yaml"), "w") as f:
            yaml.dump(results, f)

        if args.save_model:
            #just use the hf save function
            model.save_pretrained(save_path)
    # os.system("rm -rf temp/temp_model")


    






