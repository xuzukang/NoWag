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
import src.utils.utils as utils
import src.quantizers.vector_quantizer as vector_quantizer
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
def swap_layers(model:llama.LlamaForCausalLM,
                quantizer_params:dict = {
                    "d": 4,
                    "n_bits": 2,
                    "norm_order": [0,1]
                }):

    layers = model.model.layers

    sublayer_names = ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj","self_attn.o_proj","mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]

    for i in range(len(layers)):
        layer = layers[i]
        for name in sublayer_names:
            parent_module = getattr(layer, name.split(".")[0])
            module:nn.Linear= getattr(parent_module, name.split(".")[1])

            new_layer = linear_compress.LinearQuantized(module.weight,
                                                        module.bias,
                                                        args.add_bias)
            new_layer.blank_recreate(
                vector_quantizer.VectorQuantizer,
                **quantizer_params
            )
            del new_layer.original_weight



            delattr(parent_module, name.split(".")[1])
            setattr(parent_module, name.split(".")[1], new_layer)
    #clean the cuda cache
    torch.cuda.empty_cache()

    return model

# @torch.no_grad()
# def update_direcete(module:nn.Module):
#     for name, child in module.named_children():
#         if isinstance(child, vector_quantizer.VectorQuantizer):
#             child.update_discrete()
#         update_direcete(child)

@torch.no_grad()
def get_target_model_outputs(target_model:llama.LlamaForCausalLM, inputs:list[torch.LongTensor],
                             device:str):
    
    with torch.no_grad():
        target_model.eval()

        target_outs = torch.empty(
            len(inputs), target_model.seqlen, target_model.config.vocab_size,
            device="cpu"
        )

        for i in tqdm.tqdm(range(len(inputs))):
            teacher_out = target_model(inputs[i][0].to(device))[0]
            # print(teacher_out.shape)
            target_outs[i] = teacher_out.detach().cpu()
        
        return target_outs




if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Model name.")
    parser.add_argument("model_path", type=str, help="Path to the quantized model.")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
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
        "--nsamples", type=int, default=512, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--nsamples_val",
        type=int,
        default=16,
        help="Number of validation data samples.",
    )
    parser.add_argument(
        "--update_every_n_tokens",
        type = int,
        default = 4096,
        help = "Update the model every n tokens."
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
        "--ema_decay", type=float, default=0.99, help="Exponential moving average decay."
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
        default=  1,
        help="Number of layers to put on the gpu at once.",
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        help="Add bias to the quantized layers.",
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
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args,
                   project = "post_training_quantization")

    model = get_llama(args.model)
    model.seqlen = args.train_seqlen
    dataloader, valloader, testloader = get_loaders(
        args.dataset, nsamples_train=args.nsamples,
        nsamples_val=args.nsamples_val,
          seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    
    # model.seqlen = args.train_seqlen

    # model.to(args.device)

    if args.soft_labels:
        model.to(args.device)
        training_targets = get_target_model_outputs(model, dataloader, args.device)
        model.to("cpu")
        training_targets.to("cpu")
    else:
        training_targets = None
    
    # if args.nsamples_val > 0:
    #     val_targets = get_target_model_outputs(model, valloader, args.device)


    # print("traing targets", training_targets.shape)

    # training_inputs, attention_mask = finetune.get_embedded(dataloader,
    #                                         model,
    #                                         args.device)
    # print("training inputs", training_inputs.shape)

    # if args.nsamples_val > 0:
    #     val_inputs, _ = finetune.get_embedded(valloader,
    #                                        model,
    #                                        args.device)
    #     print("val inputs", val_inputs.shape)
    

    #swap the layers of the model
    # model = swap_layers(model)
    # print(model)

    original_dtype = iter(model.parameters()).__next__().dtype
    original_device = iter(model.parameters()).__next__().device
    model = swap_layers(model)

    #set the following parameters to not have a gradient computed
    model.model.embed_tokens.weight.requires_grad = False
    model.lm_head.weight.requires_grad = False

    #for each bin in the model_path
    for path in glob.glob(os.path.join(args.model_path, "*.bin")):
        print("loading", path)
        model.load_state_dict(torch.load(path), strict=False)
    print("original dtype", original_dtype)
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
        utils.recursive_apply(model, "enable_importance_updates", 
                              {"decay": args.ema_decay})
        
        utils.recursive_apply(model, "process_old_importances", {})
    else:
        utils.recursive_apply(model, "clean", {})
    model.to(args.device)
    free, total = torch.cuda.mem_get_info(int(args.device.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total") 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr,
                                    betas=(args.finetune_adam_beta1, args.finetune_adam_beta2),
                                    eps = 1e-4)
    

    for epoch in range(args.finetune_epochs):
        model.train()
        # model.to(args.device)
        model.seqlen = args.train_seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.to(args.device)
        finetune.finetune_end_to_end(
            model = model,
            optimizer = optimizer,
            train_tokens = dataloader,
            train_soft_labels = training_targets,
            val_tokens = valloader if args.nsamples_val > 0 else None,
            # partition_size = args.partition_size,
            update_every_n_tokens =  args.update_every_n_tokens,
            log_wandb = args.log_wandb,
            device = args.device,
            discrete_update_fn = lambda model: utils.recursive_apply(model, "update_discrete", {}) if args.update_discrete else None
        )
        torch.cuda.empty_cache()

        model.eval()
        model.to(args.device)
        model.seqlen = args.eval_seqlen 
        model.config.use_cache = use_cache
        llama_eval(model, testloader, args.device, args.dataset, args.log_wandb)
    # model.config.use_cache = use_cache
    # model.to(args.device)
    # # raise NotImplementedError("This script is not yet implemented.")

    




    # target_model = get_llama(args.model)

    # model.seqlen = 1024
    # target_model.seqlen = 1024

    # original_dtype = iter(model.parameters()).__next__().dtype
    # original_device = iter(model.parameters()).__next__().device
    # model = swap_layers(model)
    # print(model)

    # #set the following parameters to not have a gradient computed
    # model.model.embed_tokens.weight.requires_grad = False
    # model.lm_head.weight.requires_grad = False

    # #for each bin in the model_path
    # for path in glob.glob(os.path.join(args.model_path, "*.bin")):
    #     model.load_state_dict(torch.load(path), strict=False)
    # print("original dtype", original_dtype)
    # model = model.to(args.device).to(torch.float16)
    # target_model = target_model.to(args.device).to(torch.float16)
    # print(model)
    # print(model.config.pretraining_tp)
    # dataloader, valloader, testloader = get_loaders(
    #     args.dataset, nsamples_train=args.nsamples,
    #     nsamples_val=args.nsamples_val,
    #       seed=args.seed, model=args.model, seqlen=model.seqlen
    # )

    # if args.finetune_epochs > 0:
    #     finetune.finetune_end_to_end(
    #         model, target_model,
    #         dataloader, args,
    #         val_inps = None,
    #         # discrete_update_fn = update_direcete,
    #     )

    # model.seqlen = 4096


    # for dataset in ["wikitext2"]: #, "ptb", "c4"]:
    #     dataloader, valloader, testloader = get_loaders(
    #         dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    #     )
    #     print("Dataset:", dataset)
    #     llama_eval(model, testloader, args.device, dataset, args.log_wandb)

    # # if len(args.save)>0:
    # #     model.save_pretrained(args.save)

