import torch
import numpy as np
import argparse
from typing import List, Tuple, Literal, Optional, Union
import glob

#add the previous directory to the path
import sys
import tqdm
import os
import sys
if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())

import src.utils.normalizer as normalizer
import src.utils.sparse as sparse
import cvxpy as cp
import yaml

@torch.no_grad()
def weighted_kde_entropy(samples: torch.FloatTensor, #shape of (n_out, n_in)
                  weights: Optional[torch.FloatTensor] = None, #shape of (n_in)
                  mask: Optional[torch.BoolTensor] = None, #shape of (n_out, n_in)
                  x_range: Optional[Tuple[float, float]] = None,
                  n_samples: int = 1000,
                  bandwith_factor: float = 1.059) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Compute the entropy of a set of samples using the KDE method.
    """
                  
    bandwidth = bandwith_factor * (samples.std() * samples.numel() ** (-1 / 5))

    #if weights is not given, set it to 1
    if weights is None:
        weights = torch.ones(samples.shape[1], device=samples.device)
        
    def gaussian_pdf(x, mu  = 1, sigma = 1):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    if x_range is None:
        overall_range = (samples.min().item(), samples.max().item())
        #scale the range
        multiplier = 1+3*bandwidth
        x_range = (overall_range[0] - multiplier*bandwidth, overall_range[1] + multiplier*bandwidth)
    x = torch.linspace(x_range[0], x_range[1], n_samples, device=samples.device)

    estimated_pdf = torch.zeros_like(x)
    if mask is None:
        normalized_weights = weights/torch.sum(weights)
    
        for i in range(samples.shape[0]):
            estimated_pdf *= i/(i+1)
            estimated_pdf += torch.sum(gaussian_pdf(x.unsqueeze(-1), samples[i].unsqueeze(0), bandwidth) * normalized_weights.unsqueeze(0), dim=1) / (i+1)
    else:
        n_sum = 0
        for i in range(samples.shape[0]):
            row_mask = mask[i]
            weights_mask = weights[row_mask]
            estimated_pdf *= n_sum/(n_sum+weights_mask.sum())
            estimated_pdf += torch.sum(gaussian_pdf(x.unsqueeze(-1), samples[i,row_mask].unsqueeze(0), bandwidth) * weights_mask.unsqueeze(0), dim=1) / (n_sum+weights_mask.sum())
            n_sum += weights_mask.sum()
    # estimated_pdf = gaussian_pdf(x.unsqueeze(1), samples.unsqueeze(0), bandwidth).mean(dim=1)

    #get the entropy
    estimated_pdf = estimated_pdf / torch.trapz(estimated_pdf, x)
    entropy_base = estimated_pdf * torch.log(estimated_pdf)
    x_entropy = x[torch.isfinite(entropy_base)]
    entropy_base = entropy_base[torch.isfinite(entropy_base)]
    entropy = -torch.trapz(entropy_base, x_entropy)
    # print(torch.trapz( estimated_pdf, x))
    return entropy



parser = argparse.ArgumentParser(description='Entropy based allocation')
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--weight_path", type=str, default="/data/lliu/huffman/models/{model_name}/original_weights")
parser.add_argument("--hessian_path", type=str, default="/data/lliu/huffman/models/{model_name}/hessians_new/seed_0/pajama/128")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--importances_order", type=int, choices=[0,1,2], default=0, 
                    help="The order of the importances, 1 for l1, 2 for l2, 0 for just the raw values")
parser.add_argument("--normalizer_norm_order", type=int, nargs="+", default=[0,1])
parser.add_argument("--normalizer_zero", type=bool, nargs="+", default=[True,True])
parser.add_argument("--wanda_entropy", action="store_true"
                    , help="to calculate the entropy based on the wanda method")

parser.add_argument("--target_sparse_frac", type=float, default=0.0, help="The target sparsity fraction") 
parser.add_argument("--layer_max_sparse_frac", type=float, default=0.01, help="The maximum sparsity fraction for each layer")
parser.add_argument("--sparsity_structure", type=str, choices=["unstructured", "block",
                                                               "row", "column"], default="unstructured")
parser.add_argument("--block_size", type=int, nargs="+", default=[1,1], 
                    help="The block size for block sparsity structure, ignored for unstructured or row/column, if only one value is given, square blocks are used")
# parser.add_argument("--n_sparse_increment", type=float, default = 0.001, help="Increase the sparsity by this amount each time")

parser.add_argument("--target_bpv", type=int, default=2, help="The target number of bits")
parser.add_argument("--possible_bpv", type=float, default=[1,2,3,4], nargs="+", help="The possible number of bits")

parser.add_argument("--output_path", type=str, default="/data/lliu/huffman/models/{model_name}/allocation")
parser.add_argument("--output_name", type=str, default="allocation")
parser.add_argument("--debug", action="store_true", help="Debug mode")
args = parser.parse_args()

SAVE_PATH = args.output_path.format(model_name=args.model_name) + f"/{args.output_name}"
os.makedirs(SAVE_PATH, exist_ok=True)
                    

#load the data 
weight_dir = args.weight_path.format(model_name=args.model_name)
weights_path = glob.glob(f"{weight_dir}/**/*.pt", recursive=True)
print(f"Found {len(weights_path)} weights")
print(weights_path)

hessians_dir = args.hessian_path.format(model_name=args.model_name)

device = torch.device(args.device)


normalizer_kwargs = {"norm_order":args.normalizer_norm_order
                     ,"zero":args.normalizer_zero}


entropies_dict = {}
importances_wanda = {}
hessian_weights = {}
n_numel_dict = {}


if os.path.exists(f"{SAVE_PATH}/temp.yaml"):
    data = yaml.load(open(f"{SAVE_PATH}/temp.yaml", "r"), Loader=yaml.FullLoader)
    entropies_dict = data["entropies"]
    n_numel_dict = data["n_numel"]
    # importances_wanda = data["importances"]
    # hessian_weights = data["hessian_weights"]
    
    print("Loaded the previous data")
# else:
with torch.no_grad():
    for w_path in tqdm.tqdm(weights_path):
        # print(f"Processing {w_path}")
        w = torch.load(w_path, weights_only=True, map_location=device
                    )["weight"].to(torch.float32)
        _,w = normalizer.Normalizer.normalize_init(w,**normalizer_kwargs)
        
        diag_weigths = torch.diag(torch.load(w_path.replace(weight_dir, hessians_dir), weights_only=True, map_location=device)["hessian"]).to(torch.float32)

        # if args.importances_order == 0:
        #     importances = w * torch.sqrt(torch.diag(torch.load(w_path.replace(weight_dir, hessians_dir), weights_only=True, map_location=device)["hessian"].to(torch.float32))).unsqueeze(0) #.flatten()
        # if args.importances_order == 1:
        #     importances = (torch.abs(w) * torch.sqrt(torch.diag(torch.load(w_path.replace(weight_dir, hessians_dir), weights_only=True, map_location=device)["hessian"].to(torch.float32))).unsqueeze(0)) #.flatten()
        # if args.importances_order == 2:
        #     importances = (w**2) * torch.diag(torch.load(w_path.replace(weight_dir, hessians_dir), weights_only=True, map_location=device)["hessian"].to(torch.float32)).unsqueeze(0) #.flatten()


        name = w_path.replace(weight_dir,"")
        
        importances_wanda[name] = w * torch.sqrt(diag_weigths)
        if args.wanda_entropy:
            if name not in entropies_dict:
                entropies_dict[name] = weighted_kde_entropy(importances_wanda[name]).item()
        else:
            if name not in entropies_dict:
                entropies_dict[name] = weighted_kde_entropy(w, diag_weigths).item()
            hessian_weights[name] = diag_weigths
        n_numel_dict[name] = w.numel()
        del w
        del diag_weigths
        torch.cuda.empty_cache()
        
if args.debug:
    import matplotlib.pyplot as plt
    import copy

    entropies = copy.deepcopy([entropies_dict[key] for key in entropies_dict.keys()])
    plt.hist(entropies, bins=100)
    plt.savefig(f"{SAVE_PATH}/entropies_before.png")
    plt.close()
    
    if not os.path.exists(f"{SAVE_PATH}/temp.yaml"):
        yaml.dump({"entropies":entropies_dict, "n_numel":n_numel_dict,
                    #  "importances":importances_wanda, "hessian_weights":hessian_weights
                   }, open(f"{SAVE_PATH}/temp.yaml", "w"))

#allocate the sparsity
if args.target_sparse_frac > 0:
    n_sparse_target = args.target_sparse_frac * sum(n_numel_dict.values()) #the total number of sparse values 
    running_sparse = 0
    
    layer_sparsity = {key:0 for key in n_numel_dict.keys()}
    
    #handle the edge case of when the layer_max_sparse_frac is less or equal to the target_sparse_frac
    if args.layer_max_sparse_frac <= args.target_sparse_frac:
        #just sparsify all the layers to the maximum
        for key in n_numel_dict.keys():
            
            if args.sparsity_structure == "unstructured":
                n_sparse_layer = args.layer_max_sparse_frac * n_numel_dict[key]
                importances = importances_wanda[key]
                mask = sparse.UnstructuredSparse.generate_mask(importances, n_sparse_layer)
                
                if args.wanda_entropy:
                    entropies_dict[key] = weighted_kde_entropy(importances, mask=mask).item()
                else:
                    entropies_dict[key] = weighted_kde_entropy(importances,
                                                               hessian_weights[key], mask=mask).item()
            
                layer_sparsity[key] = n_sparse_layer  
            else:
                raise NotImplementedError("Only unstructured sparsity is supported")                           
            running_sparse += n_sparse_layer
    
    else:
        #otherwise we iterate through the layers and sparsify them
        bar = tqdm.tqdm(total=n_sparse_target)
        while running_sparse < n_sparse_target:
            sorted_keys = np.argsort([entropies_dict[key] for key in entropies_dict.keys()])[::-1]
            sorted_keys = [list(entropies_dict.keys())[i] for i in sorted_keys]
            # print(importances_wanda)
            for key in sorted_keys:
                print("sparsifying", key)
                #if we have already reached the maximum sparsity for this layer, skip
                if args.sparsity_structure == "unstructured":
                    #check if the layer is already at the maximum sparsity
                    if layer_sparsity[key] >= args.layer_max_sparse_frac * n_numel_dict[key]:
                        continue
                    
                    n_sparse_layer = args.layer_max_sparse_frac * n_numel_dict[key]
                    importances = importances_wanda[key]
                    mask = sparse.UnstructuredSparse.generate_mask(importances, int(n_sparse_layer))
                    
                    if args.wanda_entropy:
                        entropies_dict[key] = weighted_kde_entropy(importances, mask=mask).item()
                    else:
                        entropies_dict[key] = weighted_kde_entropy(importances,
                                                                   hessian_weights[key], mask=mask).item()

                    layer_sparsity[key] = n_sparse_layer
                running_sparse += n_sparse_layer
                bar.update(n_sparse_layer)
                if running_sparse >= n_sparse_target:
                    break
                
    #if we are debugging, plot the entropies
    if args.debug:

        _,bins,_ = plt.hist(entropies, bins=100, alpha=0.5, label="before")

        new_entropies = [entropies_dict[key] for key in entropies_dict.keys()]
        plt.hist(new_entropies, bins=bins, alpha=0.5, label="after")
        plt.savefig(f"{SAVE_PATH}/entropies_before_and_after.png")
        plt.yscale("log")
        plt.close()

else:
    print("No sparsity allocation")
    layer_sparsity = {key:0 for key in n_numel_dict.keys()}
#allocate the bits
if len(args.possible_bpv) == 1:
    assert args.possible_bpv[0] == args.target_bpv, "If only one possible bpv is given, it should be the target bpv"
    bpv_discrete = np.ones(len(entropies_dict)) * args.target_bpv
else:
    target_b_bits = args.target_bpv * sum(n_numel_dict.values())

    n_values = np.array(list(n_numel_dict.values()))
    entropies = np.array(list(entropies_dict.values()))

    bpv_layer_cont = cp.Variable(n_values.shape) #relaxed to be continuous

    #get the objective
    objective = cp.Minimize(cp.sum(cp.exp(2*entropies - 2*bpv_layer_cont)))

    constraints = [cp.sum(bpv_layer_cont * n_values) <= target_b_bits, 
                bpv_layer_cont >= 0
                ]

    problem = cp.Problem(objective, constraints)
    problem.solve()


    min_distortion = problem.value
    print("minimal_distortion:", problem.value)

    bpv_layer_cont = bpv_layer_cont.value
    print("n_bits", np.sum(bpv_layer_cont * n_values), "bpv", np.sum(bpv_layer_cont * n_values)/sum(n_values))

    possible_bpv = np.array(args.possible_bpv)

    #round the bits per value to the nearest possible value
    bpv_discrete = np.array([possible_bpv[np.argmin(np.abs(possible_bpv - bpv))] for bpv in bpv_layer_cont])
    # bpv_discrete = np.ones_like(bpv_discrete) * args.target_bpv
    #get the total number of bits
    n_bits = np.sum(bpv_discrete * n_values)
    print(n_bits, target_b_bits)
    print("post rounding distortion", np.sum(np.exp(2*entropies - 2*bpv_discrete)))
    layer_idx = np.argsort(entropies) #sort the layers by entropy in ascending order

    # bpv_discrete[layer_idx[0]] = possible_bpv[0]
    # bpv_discrete[layer_idx[-1]] = possible_bpv[-1]



        
    #check if the number of bits is larger or smaller than the target
    if n_bits > target_b_bits:
        print("The number of bits is larger than the target")
        
        #while the number of bits is larger than the target
        while n_bits > target_b_bits:
            print(np.sum(np.exp(2*entropies - 2*bpv_discrete)))
            #for each layer in the order of entropy
            for i in layer_idx:
                #if the number of bits is larger than the target
                if n_bits > target_b_bits:
                    #decrease the number of bits by 1
                    bpv_discrete[i] = np.max(possible_bpv[possible_bpv < bpv_discrete[i]])
                    #update the number of bits
                    n_bits = np.sum(bpv_discrete * n_values)
                else:
                    break
    elif n_bits < target_b_bits:
        print("The number of bits is smaller than the target")
        #while the number of bits is smaller than the target
        patience = 1
        while n_bits < target_b_bits:
            #for each layer in the order of entropy
            print(np.sum(np.exp(2*entropies - 2*bpv_discrete))<min_distortion)
            n_updates = 0
            for i in layer_idx[::-1]:
                #if the number of bits is smaller than the target
                # if n_bits < target_b_bits:
                    #increase the number of bits by 1
                old_bpv = bpv_discrete[i]
                if np.sum(possible_bpv > bpv_discrete[i]) == 0:
                    continue
                new_bpv = np.min(possible_bpv[possible_bpv > bpv_discrete[i]])
                n_bits_test = n_bits + n_values[i] * (new_bpv - old_bpv)
                if n_bits_test <= target_b_bits:
                    # print("accept", n_bits_test)
                    bpv_discrete[i] = new_bpv
                    n_bits = np.sum(bpv_discrete * n_values)
                    n_updates += 1   
                # else:
                    # print("current bits", n_bits,"if we add", n_values[i] * (new_bpv - old_bpv), "bits, we will have", n_bits_test)
                    # break
            
            if n_updates == 0:
                patience -= 1
                if patience == 0:
                    break
    else:
        print("The number of bits is equal to the target")      

    print("post allocation distortion", np.sum(np.exp(2*entropies - 2*bpv_discrete)))
    print("number of bits", n_bits, "bpv", n_bits/sum(n_values))
    if args.debug:
        import matplotlib.pyplot as plt

        plt.plot(entropies, bpv_discrete, "o")
        plt.plot(entropies, bpv_layer_cont, "x")
        plt.savefig(f"{SAVE_PATH}/bpv_vs_entropy.png")
        plt.close()
    

#save the allocation
    
allocation_dict = {}
for i,key in enumerate(entropies_dict.keys()):
    allocation_dict[key[1:]] = {"n_bits":float(bpv_discrete[i]), "sparse_frac":float(layer_sparsity[key]/n_numel_dict[key])}

# print(allocation_dict)
import yaml 
with open(f"{SAVE_PATH}/allocation.yaml", "w") as f:
    yaml.dump(allocation_dict, f)    
    