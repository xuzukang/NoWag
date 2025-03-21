

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

    orig_config = AutoConfig.from_pretrained(base_model)


    checkpoints_dict = yaml.load(open(checkpoints_path, "r"), Loader=yaml.FullLoader)


    compression_kwargs = yaml.load(open((checkpoints_dict[list(checkpoints_dict.keys())[0]]).replace("compressed.pt", "compressed_args.yaml")),
                                    Loader=yaml.FullLoader)
    #check that all the other checkpoints have the same compression args
    for checkpoint in checkpoints_dict.values():
        assert compression_kwargs == yaml.load(open(checkpoint.replace("compressed.pt", "compressed_args.yaml"), "r"), Loader=yaml.FullLoader)

    #remove dtype from the compression kwargs
    compression_kwargs.pop("dtype", None)

    compression_type = compression_kwargs["compression_type"]


    compression_config = {"compression_kwargs": compression_kwargs, "compression_type": compression_type,
                            "add_bias": add_bias, "skip_list":None}

    orig_config.compress_config = compression_config



    model = LlamaForCausalLM(orig_config)


    #for each checkpoint, load the right weight
    for checkpoint_name,checkpoint_path in checkpoints_dict.items():
        print(checkpoint_name)
        #first remove the base_model name from it
        checkpoint_name = checkpoint_name.replace(base_model, "")
        #now split by /
        checkpoint_name = checkpoint_name.split("/")[-2:]
        #from the first part, we can get which layer it is
        i_layer = int(checkpoint_name[0].replace("layer_", ""))
        #from the second part we can get which module (self_attn, mlp, etc) and which layer it is
        submodule_name, linear_name = checkpoint_name[1].split(".")
        
        #now we get the right module
        getattr(getattr(model.model.layers[i_layer], submodule_name), linear_name).load_state_dict(torch.load(checkpoint_path), strict=False)
        # raise ValueError("stop here")


    #save the model
    model.save_pretrained(hf_model_save_path,
                            safe_serialization=True,)


if __name__ == "__main__":
    main()





