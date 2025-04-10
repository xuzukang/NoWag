

import torch
import yaml 
import os 
import glob
import argparse
from src.model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama
from transformers import AutoConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoints_path", type=str, required=True)
    parser.add_argument("--hf_model_save_path", type=str, required=True)
    parser.add_argument("--add_bias", action="store_true")
    args = parser.parse_args()

    base_model = args.base_model
    checkpoints_path = args.checkpoints_path
    hf_model_save_path = args.hf_model_save_path
    add_bias = args.add_bias

    orig_config = AutoConfig.from_pretrained(base_model,dtype = "auto",
                                         device_map="cpu",
                                        attn_implementation='sdpa')
    orig_model = OrigLlama.from_pretrained(base_model, config=orig_config, torch_dtype="auto",
                                            device_map="cpu",
                                            low_cpu_mem_usage=True, attn_implementation='sdpa')

    checkpoints_dict = yaml.load(open(checkpoints_path, "r"), Loader=yaml.FullLoader)


    compression_kwargs = yaml.load(open((checkpoints_dict[list(checkpoints_dict.keys())[0]]).replace("compressed.pt", "compressed_args.yaml")),
                                    Loader=yaml.FullLoader)
    #check that all the other checkpoints have the same compression args
    for checkpoint in checkpoints_dict.values():
        assert compression_kwargs == yaml.load(open(checkpoint.replace("compressed.pt", "compressed_args.yaml"), "r"), Loader=yaml.FullLoader)

    #remove dtype from the compression kwargs
    compression_kwargs.pop("dtype", None)

    compression_type = compression_kwargs["compression_type"]


    compression_config = {"compression_kwargs": compression_kwargs, "compression_type": "LinearVQ", #compression_type,
                            "add_bias": add_bias, "skip_list":None}
    print(compression_config["compression_type"])
    orig_config.compress_config = compression_config



    model = LlamaForCausalLM(orig_config)
    model.to(orig_config.torch_dtype)
    model.load_state_dict(orig_model.state_dict(), strict=False)


    #for each checkpoint, load the right weight
    for checkpoint_name,checkpoint_path in checkpoints_dict.items():
        #first remove the base_model name from it
        checkpoint_name = checkpoint_name.replace(base_model, "")
        #now split by /
        checkpoint_name = checkpoint_name.split("/")[-2:]
        #from the first part, we can get which layer it is
        i_layer = int(checkpoint_name[0].replace("layer_", ""))
        #from the second part we can get which module (self_attn, mlp, etc) and which layer it is
        submodule_name, linear_name = checkpoint_name[1].split(".")
        
        #now we get the right module
        layer = getattr(getattr(model.model.layers[i_layer], submodule_name), linear_name)
        #record the original dtype
        orig_dtype = layer.codebook.dtype
        orig_device = layer.codebook.device
        #load the state dict
        state_dict = torch.load(checkpoint_path, map_location=orig_device)
        if "quantize" in compression_type: #to handle some older checkpoints
            state_dict = {s.replace("quantizer.",""):v for s,v in state_dict.items() if "reference" not in s}
            #rename "assignments" to "codes"
            state_dict = {s.replace("codes","assignments"):v for s,v in state_dict.items()}
            
            # state_dict["normalizer.original_shape"] = torch.tensor([new_layer.out_features, new_layer.in_features], dtype=torch.int32)
            # print(state_dict["normalizer.norms.0"])
            state_dict["normalizer.norms.0"] = state_dict["normalizer.norms.0"][:layer.in_features]
        layer.load_state_dict(state_dict)
        #convert to the right dtype
        layer.to(orig_dtype)        
    #save the model
    model.save_pretrained(hf_model_save_path)


if __name__ == "__main__":
    main()





