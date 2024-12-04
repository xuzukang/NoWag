CUDA_LAUNCH_BLOCKING = 1
import time
import torch

torch.autograd.set_detect_anomaly(True)
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Any
import random
import numpy as np
import src.finetune as finetune
import src.utils.utils as utils
import src.data as data
import src.utils.model_utils as model_utils
import src.quantizers.vector_quantizer as vector_quantizer
import transformers.models.llama.modeling_llama as llama
from llama_quantize import *
from perplexity_eval import  *
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
    target_model: llama.LlamaForCausalLM, inputs: list[torch.LongTensor], device: str
):
    with torch.no_grad():
        target_model.eval()

        target_outs = torch.empty(
            len(inputs),
            target_model.seqlen,
            target_model.config.vocab_size,
            device="cpu",
        )

        for i in tqdm.tqdm(range(len(inputs))):
            teacher_out = target_model(inputs[i][0].to(device))[0]
            # print(teacher_out.shape)
            target_outs[i] = teacher_out.detach().cpu()

        return target_outs
    

def resplit_loader(existing_loader:List[Tuple[torch.LongTensor, Any]],
                   new_seqlen:int,
                   )->List[Tuple[torch.LongTensor, Any]]:
    
    #do it in place
    prev_len = len(existing_loader)
    for i in range(prev_len):
        inp, tar = existing_loader.pop(0)
        assert inp.shape[1] % new_seqlen == 0, f"The sequence length {inp.shape[1]} is not divisible by {new_seqlen}"
        for j in range(0, inp.shape[1], new_seqlen):
            existing_loader.append((inp[:, j:j+new_seqlen], None))
    return existing_loader

def save_model_as_checkpoints(model, save_path, model_path):
    #create a new directory
    os.makedirs(save_path, exist_ok=True)

    #copy the params.yaml file from the model path over
    os.system(f"cp {model_path}/params.yaml {save_path}/params.yaml")

    
    for layer in model.model.layers:
        torch.save(layer.state_dict(), os.path.join(save_path, f"layer_{i}.pt"))
    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--quantized_model_path", type=str, help="Path to the quantized model.")
    parser.add_argument(
        "--save_path", type=str, help="Path to save the finetuned model."
    )
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
        "--soft_labels",
        action="store_true",
        help="Use soft labels for training.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="Exponential moving average decay.",
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
    parser.add_argument("--finetune_keep_best", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args, project="post_training_quantization")


    quantization_args = args_load.load(os.path.join(args.quantized_model_path, "args.yaml"))


    model = model_utils.get_llama(quantization_args.model)

    dataloader = data.get_loaders(
        quantization_args.dataset,
        # nsamples=128,
        nsamples=quantization_args.nsamples_train + quantization_args.nsamples_val,
        seed=args.seed,
        model=quantization_args.model,
        seqlen=quantization_args.seqlen,
        train_test="train",
    )

    if quantization_args.nsamples_val > 0:
        indexs = np.random.permutation(len(dataloader))
        train_idx, val_idx = indexs[quantization_args.nsamples_val :], indexs[: quantization_args.nsamples_val]
        train_loader = [dataloader[i] for i in train_idx]
        val_loader = [dataloader[i] for i in val_idx]
        val_loader = resplit_loader(val_loader, args.train_seqlen)
    else:
        train_loader = dataloader
        val_loader = None

    train_loader = resplit_loader(train_loader, args.train_seqlen)
    print("new train loader length", len(train_loader))
    args.eval_every_samples = int(args.eval_every_samples * quantization_args.seqlen/args.train_seqlen) if args.eval_every_samples > 0 else -1

    if args.soft_labels:
        model.to(args.device)
        training_targets = get_target_model_outputs(model, dataloader, args.device)
        model.to("cpu")
        training_targets.to("cpu")
    else:
        training_targets = None

    model, _ = load_model_from_checkpoints(args.quantized_model_path, model)

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
    free, total = torch.cuda.mem_get_info(int(args.device.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.finetune_lr,
        betas=(args.finetune_adam_beta1, args.finetune_adam_beta2),
        eps=1e-4,
    )

    for epoch in range(args.finetune_epochs):

        for i in range(0,len(train_loader), args.eval_every_samples if args.eval_every_samples > 0 else len(train_loader)):
            model.train()
            # model.to(args.device)
            model.seqlen = args.train_seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.to(args.device)
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
                    dataset, nsamples = 0, seqlen = args.eval_seqlen, model = quantization_args.model,
                    train_test = "test")
                
                llama_eval(model, testloader, args.device, dataset, args.log_wandb,
                           args.offload_activations, args.inference_batch_size)

            
            save_model_as_checkpoints(model, 
                                    os.path.join(args.save_path, f"epoch_{epoch}_iter_{i}"),
                                        args.quantized_model_path)