import argparse
import torch
import glob
import os
import tqdm

parser = argparse.ArgumentParser(description='Recorrect Hessians')
parser.add_argument("--prefix", type=str, help="prefix for the paths",
                    default = "/data/lliu/huffman/models")
parser.add_argument("--models", type = str, nargs='+', default = ["meta-llama/Llama-2-13b-hf","meta-llama/Llama-2-70b-hf"], help = "models to recorrect")
parser.add_argument('--hessians_suffixes', type=str, help='suffixes of the path to hessians',
                    nargs='+',
                    default = ["hessians_new/pajama/128","hessians_new/pajama/2048"])
parser.add_argument("--weight_save_suffix", type=str, help="suffix for the paths",
                    default = "original_weights")
args = parser.parse_args()


files = []
corresponding_weight_save_paths = []
for model in args.models:
    for suffix in args.hessians_suffixes:
        search_path = f"{args.prefix}/{model}/{suffix}/**/*.pt"
        print("search_path", search_path)
        found_files = glob.glob(search_path, recursive=True)
        files.extend(found_files)
        corresponding_weight_save_paths.extend([file.replace(suffix, args.weight_save_suffix) for file in found_files])

print("n_files", len(files))
# raise ValueError("stop here")
for i,file in enumerate(tqdm.tqdm(files)):
    data = torch.load(file)
    if sorted(list(data.keys())) == ["hessian"]:
        continue
    if sorted(list(data.keys())) != ["hessian", "weight"]:
        raise ValueError("unexpected keys", data.keys(), "expected", ["hessian", "weight"], "file", file)

    hessian = data["hessian"]
    weight = data["weight"]

    #save only the hessian to the file 
    torch.save({"hessian": hessian}, file)
    if os.path.exists(corresponding_weight_save_paths[i]):
        print("weight_already_exists", corresponding_weight_save_paths[i])
        continue
    
    os.makedirs(os.path.dirname(corresponding_weight_save_paths[i]), exist_ok=True)
    torch.save({"weight": weight}, corresponding_weight_save_paths[i])

