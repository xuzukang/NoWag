import torch
import os 
import glob
import yaml
import argparse
from perplexity_eval import *


parser = argparse.ArgumentParser()

parser.add_argument("--datasets", type=str, nargs="+",
                    choices=["wikitext2", "c4", "ptb"],
                    help="The datasets to evaluate on.",
                    default=["wikitext2", "c4"])
parser.add_argument("--log_wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="llama")
parser.add_argument("--wandb_id", type=str, default=None,
                    help = "the wandb id so we can resume the run to link it with the compression run")
parser.add_argument("--device", type=str, default="cuda:6")
parser.add_argument("--seqlen", type=int, default=4096)
parser.add_argument("--offload_activations", action="store_true")
parser.add_argument("--batch_size", type=int, default=1, 
                    help = "batch size for the activations, if not specified, we will perform a binary search to fine the optimal batch size")
parser.add_argument("--results_log_path", type=str, default = None)

args = parser.parse_args()

base_model = "meta-llama/Llama-2-7b-hf"
checkpoint_list_path = "/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/iconic-snowflake-26/checkpoints.yaml"
hessian_path = "/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/seed_0/pajama/2048"

model = get_llama(base_model)
model.seqlen = 4096


checkpoint_list = yaml.load(open(checkpoint_list_path, "r"), Loader = yaml.FullLoader)


model = load_model_from_checkpoints(checkpoint_list,base_model,
                                    model, 
                                    quantizer_type = "1st_order",
                                    clean = False)


model.to("cpu")
original_dtype = next(iter(model.parameters())).dtype
layers = model.model.layers


sublayer_names = [
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]

for i, layer in enumerate(tqdm.tqdm(layers)):
    # if i != 2:
    #     continue
    layer.to(args.device)
    layer.to(torch.float32)
    for sublayer in sublayer_names:
        # print("="*10)
        hessian_file = os.path.join(hessian_path, f"layer_{i}/{sublayer}.pt")
        # print("reference weight", getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).original_weight)
        # print("pre reconstruction_weight", getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).reconstruct())
        
        hessian = torch.load(hessian_file)["hessian"].to(args.device).to(torch.float32)
        assert torch.isclose(hessian, hessian.T, atol = 1e-5).all() 
        getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).hessian = hessian
        # for i in range(10):
        print("prev_recon_error", getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).get_reconstruction_error())
        print("n_changes:", getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).update_discrete(**{"hessian":hessian, "n_parallel": 4096, "temp":0.01}))
        print("post_recon_error", getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).get_reconstruction_error())
        
        # print("reconstructed_weight", getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).reconstruct())
        getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).clean()

        
        # getattr(getattr(layer, sublayer.split(".")[0]), sublayer.split(".")[1]).quantizer.precompute_weight()
    layer.to("cpu")
    break
model.seqlen = 4096
model.eval()
model.to(original_dtype)
#offload the model to cpu
model = model.to("cpu")

for dataset in ["wikitext2"]:

    testloader = data.get_loaders(
        dataset, nsamples = 0, seqlen = model.seqlen, model = base_model,
        train_test = "test")
    
    llama_eval(model, testloader, args.device, dataset,
                    args.offload_activations, args.batch_size,
                    base_model = base_model,
                    results_log_path = args.results_log_path)
