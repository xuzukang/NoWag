CUDA_LAUNCH_BLOCKING = 1
import time
import torch
import sys 
import os 

if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

torch.autograd.set_detect_anomaly(True)
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple, Union, Any
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
import random
import numpy as np
import src.finetune as finetune
import src.utils.utils as utils
import src.data as data
import src.utils.model_utils as model_utils
import src.utils.quantized_model as quantized_model
import transformers.models.llama.modeling_llama as llama
import glob
import tqdm
import os
import yaml

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False
import transformers




#finetune the model with huggingface trainer

@torch.no_grad()
def get_target_model_outputs(
    target_model: llama.LlamaForCausalLM, inputs: list[torch.LongTensor], device: str, seqlen:int
):
    with torch.no_grad():
        target_model.eval()

        target_outs = torch.empty(
            len(inputs),
            seqlen,
            target_model.config.vocab_size,
            device="cpu",
        )

        for i in tqdm.tqdm(range(len(inputs))):
            teacher_out = target_model(inputs[i][0].to(device))[0]
            # print(teacher_out.shape)
            target_outs[i] = teacher_out.detach().cpu()

        return target_outs

@torch.no_grad()
def calculate_logits(model: llama.LlamaForCausalLM, devset, batch_size):
    logits = []
    for i in tqdm.tqdm(range(len(devset) // batch_size), desc = "Calculating logits"):
        logits.append(
            model(devset[i * batch_size:(i + 1) *
                         batch_size].cuda())['logits'].cpu())
    logits = torch.concat(logits, dim=0)
    return logits
    
class SimpleDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def make_datasets(X:torch.FloatTensor, Y:torch.FloatTensor, n_val:int, batch_size:int,
                 pin_memory:bool = True)->Tuple[DataLoader, DataLoader]:
    
    print("X", X.shape, "Y", Y.shape)
    #make the indices
    idxs = torch.randperm(len(X))
    train_idxs = idxs[:-n_val]

    train_ds = SimpleDataset(X[train_idxs], Y[train_idxs])
    valid_ds = SimpleDataset(X[idxs[-n_val:]], Y[idxs[-n_val:]])
    return train_ds, valid_ds

    # train_dl = DataLoader(train_ds,
    #                         batch_size=batch_size,
    #                         pin_memory=pin_memory,
    #                         shuffle=True)
    # val_dl = DataLoader(valid_ds,
    #                         batch_size=batch_size,
    #                         pin_memory=pin_memory,
    #                         shuffle=False)

    # return train_dl, val_dl

def llama_arg_fn(output, args, kwargs):
    return (output[0], *args[1:]), kwargs


def get_emb(args, kwargs):
    return args[0]


def save_finetuned_model_as_checkpoints(model:llama.LlamaForCausalLM,
                                      save_dir:str, 
                                      checkpoints_dict:str,
                                      base_model:str):
    #create a new directory
    os.makedirs(save_dir, exist_ok=True)

    #copy the params.yaml file from the model path over
    # os.system(f"cp {model_path}/args.yaml {save_path}/args.yaml")

    sublayer_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]
    checkpoints_yaml_new = {}

    for layer_idx, layer in enumerate(model.layers):
        for sublayer_name in sublayer_names:
            sublayer = getattr(getattr(layer, sublayer_name.split(".")[0]), sublayer_name.split(".")[1])
            save_path = os.path.join(save_dir, base_model, f"layer_{layer_idx}", sublayer_name, "compressed.pt")
            args_path = save_path.replace("compressed.pt", "compressed_args.yaml")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # print(f"Saving {sublayer_name} to {save_path}")
            os.system(f'cp {checkpoints_dict[f"{base_model}/layer_{layer_idx}/{sublayer_name}"].replace("compressed.pt", "compressed_args.yaml")} {args_path}')
            torch.save(sublayer.state_dict(), save_path)
            checkpoints_yaml_new[f"{base_model}/layer_{layer_idx}/{sublayer_name}"] = save_path
        layer_idx += 1

    yaml.safe_dump(checkpoints_yaml_new, open(os.path.join(save_dir, "checkpoints.yaml"), "w"))
    
#model wrapper that outputs the kl divergence loss
class Model_Wrapper(nn.Module):
    def __init__(self, model:llama.LlamaForCausalLM, position_ids:torch.LongTensor, attention_mask:torch.LongTensor):
        super().__init__()
        self.model = model
        self.position_ids = position_ids
        self.attention_mask = attention_mask
    
    def forward(self, x:torch.FloatTensor, y:torch.FloatTensor):
        
        out = self.model(x, position_ids = self.position_ids, attention_mask = self.attention_mask)
        
        logits = out[0][:, :-1].contiguous().softmax(dim=-1).float()
        
        #calculate the kl divergence loss
        loss = torch.nn.functional.kl_div(logits.log(), y, reduction = "batchmean")
        
        return loss
    



if __name__ == "__main__":

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", 
                        type=str, 
                        help="Base model to finetune.",
                        default = "meta-llama/Llama-2-7b-hf")
    parser.add_argument("--compressed_model_path", 
                        type=str, 
                        help="Path to the compressed model.",
                        default="models/{base_model}/compressed/{run_name}")
    parser.add_argument("--compressed_run_name", 
                        type=str, 
                        help="Name of the compressed run.")
    parser.add_argument("--finetune_save_path", 
                        type=str, 
                        help="Path to save the finetuned model.",
                        default="models/{base_model}/ft/{run_name}")
    parser.add_argument("--seqlen",
                        type=int,
                        help="Sequence length for the model.",
                        default=4096)
    parser.add_argument("--ft_dataset",
                        type=str,
                        choices=["wikitext2", "pajama", "c4"],
                        help="Dataset to fine tune on.",
                        default="pajama")
    parser.add_argument("--ft_n_train",
                        type=int,
                        help="Number of samples to fine tune on.",  
                        default=256)
    parser.add_argument("--ft_n_val",
                        type=int,
                        help="Number of samples to validate on.",
                        default=128)
    parser.add_argument("--seed",
                        type=int,
                        help="Seed for fine tuning.",
                        default=0)
    parser.add_argument("--use_wandb",
                        action="store_true",
                        help="Use wandb for logging.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Debug mode.")
    parser.add_argument("--seed",
                        type=int,
                        help="Seed for fine tuning.",
                        default=0)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    
    utils.seed(args.seed)
    
    

    #change the paths
    args.compressed_model_path = args.compressed_model_path.replace("{base_model}", args.base_model).replace("{run_name}", args.compressed_run_name)
    args.finetune_save_path = args.finetune_save_path.replace("{base_model}", args.base_model).replace("{run_name}", args.compressed_run_name)

    args.finetune_save_path = os.path.join(args.finetune_save_path,
                                    utils.find_run_num(args.finetune_save_path))
    
    #make the save folder
    os.makedirs(args.finetune_save_path, exist_ok=True)

    #if we are using wandb, log the run
    if args.use_wandb:
        wandb.init(project="post_compression_finetune", name="".join(args.finetune_save_path.split("/")[-2:]))


    #load and get the original model
    orig_model = model_utils.get_llama(args.base_model,device_map = "auto", dtype = torch.float32)

    #get the data
    train_data:list[torch.FloatTensor] = data.get_loaders(args.ft_dataset, nsamples = args.ft_n_train+args.ft_n_val
                                  , model = args.base_model, train_test = "train",
                                  seqlen=args.seqlen)
    
    train_data = torch.stack([_[0][0] for _ in train_data])
    print("train_data.shape", train_data.shape)

    #get the target model outputs
    target_out = calculate_logits(orig_model, train_data, args.ft_batch_size)[:, :-1].contiguous().softmax(dim=-1).float()

    del orig_model
    utils.clean()

    #load the compressed model
    deug_model_path = f"temp/{args.base_model}.pt"
    print("deug_model_path", deug_model_path, "exists", os.path.exists(deug_model_path))
    checkpoints = yaml.load(open(os.path.join(args.compressed_model_path, "checkpoints.yaml"),"r"), Loader=yaml.FullLoader)
    compressed_model, n_bits, n_values = quantized_model.load_model_from_checkpoints(checkpoints,
                                                                   args.base_model,
                                                                   add_bias = True,
                                                                   device=None,
                                                                   cache_reconstruct=False,
                                                                   load_checkpoints= not (args.debug and os.path.exists(deug_model_path)))
    
            
    print("n_bits", n_bits, "n_values", n_values)
    print("bpv", n_bits / n_values)
    emb = compressed_model.model.embed_tokens(train_data)
    position_ids = torch.arange(args.seqlen, dtype=torch.int32)[None, :] + \
        torch.zeros(args.ft_batch_size, args.seqlen, dtype=torch.int32)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.ft_batch_size, args.seqlen), emb[:args.ft_batch_size], 0)
    
    print("position_ids", position_ids.shape)
    print("attention_mask", attention_mask.shape)  



    trainset, valeset = make_datasets(emb, target_out, args.ft_n_val, args.ft_batch_size)

    #get one entry
    for item in trainset[0]:
        print("item", item.shape)
        

    #print the class of the model
    print(compressed_model)
    print("type(compressed_model)", type(compressed_model))
    torch.set_grad_enabled(True)
    
    #wrap the model
    model = Model_Wrapper(compressed_model, position_ids, attention_mask)

    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=args.finetune_save_path,
            per_device_train_batch_size=args.ft_batch_size,
            per_device_eval_batch_size=args.ft_batch_size,
            num_train_epochs=1,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            fp16=True,
            report_to="wandb" if args.use_wandb else "none",
            disable_tqdm=args.debug,
            logging_dir=os.path.join(args.finetune_save_path, "logs"),
        ),
    )
    
    #train the model
    trainer.train(train_dataset=trainset, eval_dataset=valeset)
    
                                     
                                     
                                     
                                                                   

    

