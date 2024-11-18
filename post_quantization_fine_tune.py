CUDA_LAUNCH_BLOCKING=1
import time
import torch
torch.autograd.set_detect_anomaly(True)
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from vector_quantizer import *
from modelutils import *
from quant import *
import random 
import numpy as np
import fine_tune as lora_fine_tune
import src.finetune as finetune
import src.finetune_amp as finetune_amp
import src.quantizers.vector_quantizer as vector_quantizer
import src.utils.alignment.hessian_general_align as hessian_general_align
from llama import *
import transformers.models.llama.modeling_llama as llama
import glob
import os
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False
import transformers
# def get_llama(model):
#     import torch
#     def skip(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip
#     from transformers import LlamaForCausalLM
#     model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
#     model.seqlen = 4096
#     print("Model loaded.", model)
#     return model

@torch.no_grad()
def swap_layers(model:llama.LlamaForCausalLM):

    layers = model.model.layers

    sublayer_names = ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj","self_attn.o_proj","mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]

    for i in range(len(layers)):
        layer = layers[i]
        for name in sublayer_names:
            parent_module = getattr(layer, name.split(".")[0])
            module:nn.Linear= getattr(parent_module, name.split(".")[1])

            vq = vector_quantizer.VectorQuantizer.blank_recreate(
                module.weight,
                d = 4,
                n_bits = 2,
            )
            new_layer = QuantizedLinear(vq,
                                        module.bias,
                                        True
                                        )
            delattr(parent_module, name.split(".")[1])
            setattr(parent_module, name.split(".")[1], new_layer)
    #clean the cuda cache
    torch.cuda.empty_cache()

    return model


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Model name.")
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128*4, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--nsamples_val",
        type=int,
        default=16,
        help="Number of validation data samples.",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=1,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
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
        default=1e-5,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=1024,
        help="(finetuning only) train on batches of this many sequences, globally across all GPUs",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
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
        default=0.95,
        help="Finetuning adam_beta2",
    )
    parser.add_argument("--finetune_keep_best", action="store_true")
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=None,
        help="(finetuning only) Per-device and per-forward-pass batch size used to accumulate global --batch_size",
    )
    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args,
                   project = "post_training_quantization")

    model = get_llama(args.model)
    target_model = get_llama(args.model)

    model.seqlen = 1024
    target_model.seqlen = 1024

    original_dtype = iter(model.parameters()).__next__().dtype
    original_device = iter(model.parameters()).__next__().device
    model = swap_layers(model)
    print(model)

    #set the following parameters to not have a gradient computed
    model.model.embed_tokens.weight.requires_grad = False
    model.lm_head.weight.requires_grad = False

    #for each bin in the model_path
    for path in glob.glob(os.path.join(args.model_path, "*.bin")):
        model.load_state_dict(torch.load(path), strict=False)
    print("original dtype", original_dtype)
    model = model.to(args.device).to(torch.float16)
    target_model = target_model.to(args.device).to(torch.float16)
    print(model)
    print(model.config.pretraining_tp)
    dataloader, valloader, testloader = get_loaders(
        args.dataset, nsamples_train=args.nsamples,
        nsamples_val=args.nsamples_val,
          seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.finetune_epochs > 0:
        finetune_amp.finetune_end_to_end(
            model, target_model,
            dataloader, args,
            val_inps = None,
        )
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model.seqlen = 4096


    for dataset in ["wikitext2"]: #, "ptb", "c4"]:
        dataloader, valloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, args.device, dataset, args.log_wandb)

    # if len(args.save)>0:
    #     model.save_pretrained(args.save)

