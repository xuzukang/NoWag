import torch
import transformers
import yaml
import numpy as np
import sys 
import os 

if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.utils import utils
from src.model import llama
from transformers import LlamaForCausalLM as OrigLlama
import os
from src import data
import tqdm 
import torch
import argparse
import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset
from typing import Tuple
import torch.nn as nn
import wandb



class SimpleDataset(Dataset):

    def __init__(self, inputs, soft_labels):
        self.inputs = inputs
        self.soft_labels = soft_labels
    

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.soft_labels[idx],
        }
    
def make_datasets(X:torch.FloatTensor, Y:torch.FloatTensor, n_val:int) -> Tuple[Dataset, Dataset]:
    

    #make the indices
    idxs = torch.randperm(len(X))
    train_idxs = idxs[:-n_val]

    train_ds = SimpleDataset(X[train_idxs], Y[train_idxs])
    valid_ds = SimpleDataset(X[idxs[-n_val:]], Y[idxs[-n_val:]])
    return train_ds, valid_ds



@torch.no_grad()
def calculate_logits(model: llama.LlamaForCausalLM, devset, batch_size):
    logits = []
    for i in tqdm.tqdm(range(len(devset) // batch_size), desc = "Calculating logits"):
        logits.append(
            model(devset[i * batch_size:(i + 1) *
                         batch_size].cuda())['logits'].cpu())
    logits = torch.concat(logits, dim=0)
    return logits

#custom kld loss
def custom_kld_loss(outputs, labels, num_items_in_batch):
    # print(outputs['logits'].shape)
    logits = outputs['logits'][:,:-1,:].contiguous()
    # print(logits.shape, labels.shape, num_items_in_batch)
    
    #take the cross entropy loss along the last dimension
    #labels are of the same shape as logits
    loss = -torch.sum(labels * torch.log_softmax(logits, dim=-1), dim=-1)
    loss = loss.mean()
    return loss

@hydra.main(version_base=None, config_path="../config", config_name="quantized_ft_config")
def main(cfg: DictConfig):
    print("config:")
    print(OmegaConf.to_yaml(cfg))
    print("N_visible GPUs: ", torch.cuda.device_count())
    
    utils.seed(cfg.seed)
    
    #we expect the config to have a sub section called model with the name of the base model and the path to the quantized model
    base_model = cfg.model.base_model
    quantized_model_path = cfg.model.quantized_model_path
    seqlen = cfg.model.seqlen #this can be -1, in which case we switch to using the model's full sequence length

    base_model = OrigLlama.from_pretrained(base_model,
                                    device_map="auto", torch_dtype=torch.float16)
    if seqlen <= 0:
        seqlen = base_model.config.max_position_embeddings
        print(f"Using sequence length: {seqlen}")
        
    #load the overall data, we expect the config to have a dataset section with the name of the dataset
    dataset_cfg = cfg.dataset
    overall_data:list[torch.FloatTensor] = data.get_loaders(dataset_cfg.name, 
                                                            nsamples = dataset_cfg.ft_n_train + dataset_cfg.ft_n_val,
                                                            model = base_model.config.name_or_path,
                                                            train_test = "train",
                                                            seqlen=seqlen)
    

    overall_data = torch.stack([_[0][0] for _ in overall_data])

    if cfg.soft_labels:
        overall_out = calculate_logits(base_model,overall_data, cfg.logits_batch_size)
        print(overall_out)


        overall_out = overall_out[:, :-1].contiguous().to(torch.float32).softmax(dim=-1).float()
        print("overall_out\n",overall_out)
    del base_model
    utils.clean()
    
    quantized_model = llama.LlamaForCausalLM.from_pretrained(quantized_model_path,
                                                    device_map="auto",
                                                        torch_dtype=torch.float32,
                                                          low_cpu_mem_usage=True)
    if cfg.model.freeze_embeddings:
        quantized_model.model.embed_tokens.weight.requires_grad = False
        

    n_params = 0
    for name, param in quantized_model.named_parameters():
        if param.requires_grad:
            # print(name, param.numel())
            n_params += param.numel()
        else:
            print(name, "not requires grad, n_params: ", param.numel())
        
    print(f"Total number of parameters: {n_params}")
    
    if cfg.soft_labels:
        trainset,valset = make_datasets(overall_data,
                                    overall_out,
                                    dataset_cfg.ft_n_val)
    else:
        trainset,valset = make_datasets(overall_data,
                                    overall_data,
                                    dataset_cfg.ft_n_val
                                    )
    
    print(f"len(trainset): {len(trainset)}, len(valset): {len(valset)}")
    # wandb.init(project="llama-post_quantization_ft", name="llama-2-7b-hf-soft", config=cfg)
    
    args = instantiate(cfg.ft_args,
                       report_to="wandb",
                       run_name="llama-2-7b-hf-" + "soft" if cfg.soft_labels else "hard",
                       eval_on_start = True,
                       learning_rate = cfg.lr,
                       output_dir=cfg.output_dir)
    
    optimizer = None if not hasattr(cfg, "optimizer") else instantiate(cfg.optimizer, params=quantized_model.parameters())
    scheduler = None if not hasattr(cfg, "scheduler") else instantiate(cfg.scheduler, optimizer=optimizer)

    trainer = transformers.Trainer(
        model=quantized_model,
        args=args,
        train_dataset=trainset,
        eval_dataset=valset,
        compute_loss_func=custom_kld_loss if cfg.soft_labels else None,
        optimizers = (optimizer, scheduler)
    )
    trainer.train()
        
if __name__ == "__main__":
    main()
    