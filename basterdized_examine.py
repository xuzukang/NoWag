import yaml
import os
original_yaml_path = "/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/scarlet-fire-56/checkpoints.yaml"
original_yaml = yaml.load(open(original_yaml_path, "r"), Loader = yaml.FullLoader)
basterdizing_yaml = "/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/amber-serenity-59/checkpoints.yaml"
eps = 1e-5
base_model = "meta-llama/Llama-2-7b-hf"
print("="*10, "NON BASTERDIZED", "="*10)
os.popen(f"python perplexity_eval.py --base_model meta-llama/Llama-2-7b-hf --checkpoint_list_path {original_yaml_path} --results_log_path /data/lliu/huffman/test/non_basterdized_results.yaml")

# from quant import *
from perplexity_eval import *

model = get_llama(base_model)
model.seqlen = 4096
model_name = base_model

checkpoints = yaml.load(open(original_yaml_path, "r"), Loader = yaml.FullLoader)

model = get_llama(base_model)

class args:
    def __init__(self):
        pass
    
args = args()
args.base_model = base_model
args.log_wandb = False
args.seqlen = 4096

model = load_model_from_checkpoints(checkpoints,
                                    # lambda x: "joint2" if "self_attn" in x else "quantize",
                                    model,
                                    args = args,
                                    disable_tqdm=True)

model.seqlen = args.seqlen
model.eval()
#offload the model to cpu
model = model.to("cpu")

testloaders = {}

for dataset in ["wikitext2"]:

    testloader = data.get_loaders(
        dataset, nsamples = 0, seqlen = model.seqlen, model = model_name,
        train_test = "test")
    
    testloaders[dataset] = testloader

ppl_best = {}
for dataset, testloader in testloaders.items():
    ppl_best[dataset] = llama_eval(model, testloader, "cuda:1", dataset, False,
                     False, 1,
                     base_model = base_model,
                     disable_tqdm = True)



bad_layers = [] #layers from the bastardizing compression method that result in ppl increases, and thus should be ignored
for i in range(32):

    #read the bastardized yaml
    basterdizing_yaml_loaded = yaml.load(open(basterdizing_yaml, "r"), Loader = yaml.FullLoader)

    modules_to_change = {}
    for key, value in basterdizing_yaml_loaded.items():
        if f"layer_{i}/" in key:
            modules_to_change[key] = value
            
            
    model = load_model_from_checkpoints(modules_to_change,
                                    # lambda x: "joint2" if "self_attn" in x else "quantize",
                                    model,
                                    args = args,
                                    key_no_exist_handling="ignore",
                                    disable_tqdm=True)
    
    model.seqlen = args.seqlen
    model.eval()
    #offload the model to cpu
    model = model.to("cpu")
    
    ppl = {}
    for dataset, testloader in testloaders.items():
        ppl[dataset] = llama_eval(model, testloader, "cuda:1", dataset, False,
                         False, 1,
                         base_model = base_model,
                         disable_tqdm = True)
    
    is_layer_bad = False
    for key in ppl.keys():
        print(f"layer {i}, dataset {key}, ppl: {ppl[key]}, ppl_best: {ppl_best[key]}")
        if ppl[key] - ppl_best[key] > eps:
            is_layer_bad = True
            print("this layer is bad")
            break
    
    if is_layer_bad:
        #reload the original weights
        modules_to_correct = {}
        for key, value in checkpoints.items():
            if f"layer_{i}/" in key:
                modules_to_correct[key] = value
                
                
        model = load_model_from_checkpoints(modules_to_correct,
                                        # lambda x: "joint2" if "self_attn" in x else "quantize",
                                        model,
                                        args = args,
                                        key_no_exist_handling="ignore",
                                        disable_tqdm=True)
        
        model.seqlen = args.seqlen
        model.eval()
        #offload the model to cpu
        model = model.to("cpu")
        
        ppl = {}
        for dataset, testloader in testloaders.items():
            ppl[dataset] = llama_eval(model, testloader, "cuda:0", dataset, False,
                            False, 1,
                            base_model = base_model,
                            disable_tqdm = True)
        
        for key in ppl.keys():
            assert abs(ppl[key] - ppl_best[key]) < eps, f"I cannot recover the original weights for layer {i}, previous ppl: {ppl_best[key]}, current ppl: {ppl[key]} on dataset {key}"
            
    else:
        print("this layer is good")
        for key in ppl.keys():
            ppl_best[key] = ppl[key]
            
            


