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
import src.utils.shard as shard
import src.utils.quantized_model as quantized_model
import transformers.models.llama.modeling_llama as llama
import glob
import tqdm
import os
import yaml
import lightning as L

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False
import transformers



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
    
def make_loaders(X:torch.FloatTensor, Y:torch.FloatTensor, n_val:int, batch_size:int,
                 pin_memory:bool = True)->Tuple[DataLoader, DataLoader]:
    
    print("X", X.shape, "Y", Y.shape)
    #make the indices
    idxs = torch.randperm(len(X))
    train_idxs = idxs[:-n_val]

    train_ds = SimpleDataset(X[train_idxs], Y[train_idxs])
    valid_ds = SimpleDataset(X[idxs[-n_val:]], Y[idxs[-n_val:]])

    train_dl = DataLoader(train_ds,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            shuffle=True)
    val_dl = DataLoader(valid_ds,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            shuffle=False)

    return train_dl, val_dl

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


class LightningLLM(L.LightningModule):

    def __init__(self, model: llama.LlamaForCausalLM,
                 position_ids: torch.LongTensor,
                 attention_mask: torch.FloatTensor,
                 lr: float = 1e-5,
                 optimizer_kwargs: dict = {},
                 lr_scheduler_name: str = "none",
                lr_scheduler_kwargs: dict = {},
    ):
        super().__init__()
        self.model = model
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.logger: L.loggers.WandbLogger  

        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_kwargs = lr_scheduler_kwargs


    def forward(self, inputs):  
        #return the inputs
        logits = self.model(inputs, position_ids=self.position_ids, attention_mask=self.attention_mask)['logits']
        return logits
    

    def step_(self, batch, mode:str):
        inputs, targets = batch
        logits = self(inputs)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log({f"{mode}_loss": loss})
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step_(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step_(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, **self.optimizer_kwargs)
        if self.lr_scheduler_name == "none":
            return {"optimizer": optimizer}
        else:
            scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(optimizer, **self.lr_scheduler_kwargs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    #custom save and load function using save_sharded_model_as_checkpoints
    def on_save_checkpoint(self, checkpoint):
        print(checkpoint.keys())
        raise NotImplementedError("on_save_checkpoint not implemented")
        





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
    parser.add_argument("--ft_epochs",
                        type=int,
                        help="Number of epochs to fine tune for.",
                        default=5),
    parser.add_argument("--ft_update_freq",
                        type=int,
                        help="Update the model every n batches.",
                        default=1)
    parser.add_argument("--ft_batch_size",
                        type=int,
                        help="Batch size for fine tuning.",
                        default=8)
    parser.add_argument("--ft_lr",
                        type=float,
                        help="Learning rate for fine tuning.",
                        default=1e-5)
    parser.add_argument("--ft_grad_checkpoint",
                        action="store_true",
                        help="Use gradient checkpointing for fine tuning.")
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
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    L.seed_everything(args.seed)

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
    orig_model = model_utils.get_llama(args.base_model,device_map = "auto")

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
                                                                   device="cpu",
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

    output_layer = {
        'layer': nn.Sequential(compressed_model.model.norm, compressed_model.lm_head),
        'fn': get_emb
    }


    trainloader,valloader = make_loaders(emb, target_out, args.ft_n_val, args.ft_batch_size)

    torch.set_grad_enabled(True)
    model = LightningLLM(compressed_model, position_ids, attention_mask, args.ft_lr)
    trainer = L.Trainer( logger = None,
                        accelerator = "gpu",
                        # devices = 4,
                        strategy = "fsdp",
                        precision="bf16-mixed",
                        gradient_clip_val=1.0,
                        gradient_clip_algorithm="norm",
                        accumulate_grad_batches = args.ft_update_freq,
    )
    trainer.fit(model, trainloader, valloader)

                                     
                                     
                                     
                                                                   

    

