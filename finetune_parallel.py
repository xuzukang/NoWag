CUDA_LAUNCH_BLOCKING = 1
import time
import torch

torch.autograd.set_detect_anomaly(True)
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple, Union, Any
import random
import numpy as np
import src.finetune as finetune
import src.utils.utils as utils
import src.data as data
import src.utils.model_utils as model_utils
import src.quantizers.vector_quantizer as vector_quantizer
import src.utils.quantized_model as quantized_model
import transformers.models.llama.modeling_llama as llama
import glob
import os

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
    for i in range(len(devset) // batch_size):
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


def save_sharded_model_as_checkpoints(sharded_model, save_dir, checkpoints_dict, base_model):
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

    layer_idx = 0
    for shard in sharded_model.shards:
        for layer in shard.layers:
            for sublayer_name in sublayer_names:
                sublayer = getattr(getattr(layer, sublayer_name.split(".")[0]), sublayer_name.split(".")[1])
                save_path = os.path.join(save_dir, base_model, f"layer_{layer_idx}", sublayer_name, "compressed.pt")
                args_path = save_path.replace("compressed.pt", "compressed_args.yaml")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving {sublayer_name} to {save_path}")
                os.system(f'cp {checkpoints_dict[f"{base_model}/layer_{layer_idx}/{sublayer_name}"].replace("compressed.pt", "compressed_args.yaml")} {args_path}')
                torch.save(sublayer.state_dict(), save_path)
            layer_idx += 1


    

def add_optional_parameters(parser):
    
    parser.add_argument("--finetune_dataset", type=str, default = "pajama", help = "Dataset to fine tune on, if not provided, the dataset used for quantization will be used.")
    parser.add_argument("--finetune_nsamples_train", type=int, default = 256, help = "Number of samples to fine tune on, if not provided, the entire dataset will be used.")
    parser.add_argument("--finetune_nsamples_val", type=int, default = 0, help = "Number of samples to validate on, if not provided, the entire dataset will be used.")

if __name__ == "__main__":
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
                        default="models/{base_model}/finetuned/{run_name}")
    parser.add_argument("--save_folder", 
                        type=str, 
                        help="Path to save the finetuned model."
    )
    parser.add_argument("--finetune_dataset",
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
                        default=5)
    parser.add_argument("--ft_batch_size",
                        type=int,
                        help="Batch size for fine tuning.",
                        default=8)
    parser.add_argument("--ft_lr",
                        type=float,
                        help="Learning rate for fine tuning.",
                        default=1e-3)
    parser.add_argument("--ft_adam_beta1",
                        type=float,
                        help="Adam beta1 for fine tuning.",
                        default=0.9)
    parser.add_argument("--ft_adam_beta2",
                        type=float,
                        help="Adam beta2 for fine tuning.",
                        default=0.999)
    parser.add_argument("--ft_grad_checkpoint",
                        action="store_true",
                        help="Use gradient checkpointing for fine tuning.")
    parser.add_argument("--seed",
                        type=int,
                        help="Seed for fine tuning.",
                        default=0)
    

    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        choices=["wikitext2", "ptb", "c4"],
        help="Where to evaluate the model.",
        default=["wikitext2", "c4"],
    )
    parser.add_argument(
        "--eval_every_samples",
        type=int,
        default=-1,
        help="Evaluate the model every n samples.",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to the cpu.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on."
    )
    parser.add_argument(
        "--update_every_n_tokens",
        type=int,
        default=4096,
        help="Update the model every n tokens.",
    )
    parser.add_argument(
        "--update_discrete",
        action="store_true",
        help="Update the discrete weights.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="Exponential moving average decay.",
    )
    parser.add_argument(
        "--soft_labels",
        action="store_true",
        help="Use soft labels for training.",
    )
    parser.add_argument(
        "--train_seqlen",
        type=int,
        default=1024,
        help="Sequence length for training.",
    )
    parser.add_argument(
        "--eval_seqlen",
        type=int,
        default=4096,
        help="Sequence length for evaluation.",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=5,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--partition_size",
        type=int,
        default=1,
        help="Number of layers to put on the gpu at once.",
    )
    parser.add_argument(
        "--finetune_early_stop",
        type=int,
        default=5,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-3,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--finetune_adam_beta1",
        type=float,
        default=0.9,
        help="Finetuning adam_beta1",
    )
    parser.add_argument(
        "--finetune_adam_beta2",
        type=float,
        default=0.999,
        help="Finetuning adam_beta2",
    )
    parser.add_argument(
        "--embedding_grad",
        action="store_true",
        help="Whether to train the embedding layer.",
    )
    parser.add_argument(
        "--final_mlp_grad",
        action="store_true",
        help="Whether to train the final mlp layer.",
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        help="Whether to add bias to each layer.",
    )
    parser.add_argument(
        "--amp_finetuning",
        action = "store_true",  
        help = "Whether to use automatic mixed precision for finetuning."
    )
    parser.add_argument(
        "--finetune_grad_checkpoint",
        action = "store_true",
        help = "Whether to use gradient checkpointing for finetuning."
    )
    parser.add_argument("--finetune_keep_best", action="store_true")
    add_optional_parameters(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # init W&B logging

    #expect the checkpoints path to be of type:
    #/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/wandering-leaf-43/checkpoints.yaml
    
    load_dir = os.path.dirname(args.checkpoint_list_path)
    splitted_path = args.checkpoint_list_path.split("/")
    model_name = ""
    run_name = ""
    add = False
    for i in range(len(splitted_path)):
        if splitted_path[i] == "compressed":
            add = False
        if add:
            model_name += splitted_path[i] + "/"
        if splitted_path[i] == "models":
            add = True
        if splitted_path[i] == "compressed":
            run_name = splitted_path[i+1]
            break
    model_name = model_name[:-1]
    print("model_name", model_name)


    save_path = args.checkpoint_list_path.replace("compressed", "finetuned")
    #remove the last part of the path
    save_path = save_path[:save_path.rfind("/")]
    print("save_path", save_path)
    os.makedirs(save_path, exist_ok=True)

    #count the number of files in the directory
    n_prev_runs = len(glob.glob(os.path.join(save_path, "*")))
    run_name = f"{run_name}-{n_prev_runs}"
    print("run_name", run_name)
    save_path = os.path.join(save_path, run_name)

    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args, project="post_training_compression_training",
                     name = run_name)

    # quantization_args = args_load.load(os.path.join(args.quantized_model_path, "args.yaml"))


    model = model_utils.get_llama(model_name)
    args.base_model = model_name
    # model_name = 

    dataloader = data.get_loaders(
        args.finetune_dataset,
        # nsamples=128,
        nsamples=args.finetune_nsamples_train + (0 if args.finetune_nsamples_val is None else args.finetune_nsamples_val),
        seed=args.seed,
        model=model_name,
        seqlen=args.eval_seqlen,
        train_test="train"
    )
    dataloader = torch.stack([x[0] for x in dataloader]) #shape of dataloader is (nsamples, seqlen)

    if args.finetune_nsamples_val > 0:
        idxs = torch.random.shuffle(torch.arange(len(dataloader)))
        val_idxs = idxs[:args.finetune_nsamples_val]
        train_idxs = idxs[args.finetune_nsamples_val:]
        val_loader = dataloader[val_idxs]
        train_loader = dataloader[train_idxs]
    else:
        train_loader = dataloader
        val_loader = None


    if args.soft_labels:
        model.to(args.device)
        training_targets = get_target_model_outputs(model, dataloader, args.device, args.train_seqlen)
        model.to("cpu")
        training_targets.to("cpu")
    else:
        training_targets = None
    print("here")
    
    checkpoints_dict = yaml.load(open(args.checkpoint_list_path, "r"), Loader = yaml.FullLoader)
    model = load_model_from_checkpoints(checkpoints_dict, 
                                        model_name,
                                        model, 
                                        add_bias=args.add_bias)
    #save a new version of the checkpoint dict

    model.seqlen = args.train_seqlen
    model.to(args.device)
    model.train()

    # set the following parameters to not have a gradient computed
    model.model.embed_tokens.weight.requires_grad = False
    model.lm_head.weight.requires_grad = False

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(int(args.device.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
    model = model.to(torch.float16)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(int(args.device.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
    # training_inputs = training_inputs.to(torch.float16)
    # training_targets = training_targets.to(torch.float16)
    # if args.nsamples_val > 0:
    #     val_inputs = val_inputs.to(torch.float16)
    #     val_targets = val_targets.to(torch.float16)
    if args.update_discrete:
        utils.recursive_apply(
            model, "enable_importance_updates", {"decay": args.ema_decay}
        )

        utils.recursive_apply(model, "process_old_importances", {})
    else:
        try:
            utils.recursive_apply(model, "clean", {})
        except:
            pass
    model.to(args.device)
    model.gradient_checkpointing = args.finetune_grad_checkpoint
    free, total = torch.cuda.mem_get_info(int(args.device.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape, param.numel())
    print(f"Number of trainable parameters: {f'{n_params:,}'}")    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.finetune_lr,
        betas=(args.finetune_adam_beta1, args.finetune_adam_beta2),
        eps=1e-4,
    )

    # if args.log_wandb:
    #     wandb.watch(model, log_freq = 1)

    for epoch in range(args.finetune_epochs):

        for i in range(0,len(train_loader), args.eval_every_samples if args.eval_every_samples > 0 else len(train_loader)):
            model.train()
            # model.to(args.device)
            model.seqlen = args.train_seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.to(args.device)
            if args.amp_finetuning:
                finetune.finetune_end_to_end_amp(
                    model=model,
                    optimizer=optimizer,
                    train_tokens=train_loader[i:i+args.eval_every_samples],
                    train_soft_labels=training_targets[i:i+args.eval_every_samples] if args.soft_labels else None,
                    val_tokens=val_loader,
                    # partition_size = args.partition_size,
                    update_every_n_tokens=args.update_every_n_tokens,
                    log_wandb=args.log_wandb,
                    device=args.device,
                    discrete_update_fn=lambda model: utils.recursive_apply(
                        model, "update_discrete", {}
                    )
                    if args.update_discrete
                    else None,
                    use_tqdm=True,
                )
            else:

                finetune.finetune_end_to_end(
                        model=model,
                        optimizer=optimizer,
                        train_tokens=train_loader[i:i+args.eval_every_samples],
                        train_soft_labels=training_targets[i:i+args.eval_every_samples] if args.soft_labels else None,
                        val_tokens=val_loader,
                        # partition_size = args.partition_size,
                        update_every_n_tokens=args.update_every_n_tokens,
                        log_wandb=args.log_wandb,
                        device=args.device,
                        discrete_update_fn=lambda model: utils.recursive_apply(
                            model, "update_discrete", {}
                        )
                        if args.update_discrete
                        else None,
                        use_tqdm=True,
                    )
            torch.cuda.empty_cache()

            model.eval()
            model.to(args.device)
            model.seqlen = args.eval_seqlen
            model.config.use_cache = use_cache
            for dataset in args.eval_datasets:

                testloader = data.get_loaders(
                    dataset, nsamples = 0, seqlen = args.eval_seqlen, model = model_name,
                    train_test = "test")
                
                llama_eval(model, testloader, args.device, dataset, args.log_wandb,
                           args.offload_activations, args.inference_batch_size,
                           base_model = model_name)

            
            save_model_as_checkpoints(model,
                                      save_path,
                                        checkpoints_dict,
                                        base_model = model_name)
            
            new_checkpoints_dict = {}
            for key in checkpoints_dict:
                new_checkpoints_dict[key] = checkpoints_dict[key].replace(load_dir, save_path)
            yaml.dump(new_checkpoints_dict, open(args.checkpoint_list_path.replace(load_dir, save_path), "w"))