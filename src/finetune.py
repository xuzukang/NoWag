# a simpler version of fine tunining to work on one GPU


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import transformers.models.llama.modeling_llama as llama
import argparse
import tqdm
import numpy as np
from typing import List, Optional, Callable, Tuple, Union
from copy import deepcopy
import wandb
import gc

import src.utils.shard as shard
import src.utils.utils as utils


class NANError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


@torch.enable_grad()
def finetune_amp(
    layer: llama.LlamaDecoderLayer,
    train_inps: torch.Tensor,
    train_outputs: torch.Tensor,
    val_inps: torch.Tensor,
    val_outputs: torch.Tensor,
    args: argparse.Namespace,
    parameters_to_optimize: dict = None,
    discrete_update_fn: Optional[Callable] = None,
    layer_kwargs: dict = None,
    early_stop_eps: float = 1e-7,
    adam_eps: float = 1e-7,
):
    # if we want to put the fnn on a separate device
    device = args.device
    if args.fnn_device is not None:
        layer.mlp.to(args.fnn_device)
        # add a hook to transfer the inputs to the device and the outputs back to the original device
        layer.mlp.register_forward_pre_hook(
            lambda module, inputs: inputs.to(args.fnn_device)
        )
        layer.mlp.register_forward_hook(
            lambda module, inputs, outputs: outputs.to(device)
        )

    # get the parameters
    if parameters_to_optimize is None:
        parameters_to_optimize = {
            name: param
            for name, param in layer.named_parameters()
            if param.requires_grad
        }
        parameters_not_to_optimize = {
            name: param
            for name, param in layer.named_parameters()
            if not param.requires_grad
        }
        print(
            "the following parameters will not be optimized:",
            parameters_not_to_optimize.keys(),
        )
        print(
            "number of parameters to not optimize:",
            sum([param.numel() for param in parameters_not_to_optimize.values()]),
        )
    else:
        print("using the provided parameters")

    print("optimizing the following parameters:", parameters_to_optimize.keys())
    print(
        "total number of parameters to optimize:",
        sum([param.numel() for param in parameters_to_optimize.values()]),
    )
    optimizer = torch.optim.Adam(
        nn.ParameterList(parameters_to_optimize.values()),
        lr=args.finetune_lr,
        betas=(args.finetune_adam_beta1, args.finetune_adam_beta2),
        eps=adam_eps,
    )

    # print("initial parameter",list(parameters_to_optimize.keys())[0], parameters_to_optimize[list(parameters_to_optimize.keys())[0]])
    # set the model to train mode
    layer.train()

    local_batch_size = (
        args.local_batch_size
        if args.local_batch_size is not None
        else args.finetune_batch_size
    )
    # check that the local batch size is a multiple of the number of data points
    assert (
        len(train_inps) % local_batch_size == 0
    ), "the local batch size should be a multiple of the number of data points"

    n_accumulation_steps = args.finetune_batch_size // local_batch_size
    # check that the number of accumulation steps is a multiple of the local batch size
    assert (
        args.finetune_batch_size % local_batch_size == 0
    ), "the number of accumulation steps should be a multiple of the local batch size"

    n_batches = int(np.ceil(len(train_inps) / local_batch_size))
    if val_inps is not None:
        n_batches_val = int(np.ceil(len(val_inps) / local_batch_size))

    indexes = torch.randperm(len(train_inps), device=torch.device("cpu"))

    # move the train_inps and train_outputs to cpu to save gpu memory
    train_inps = train_inps.cpu()
    train_outputs = train_outputs.cpu()

    scaler = amp.GradScaler()
    prev_epoch_loss = np.inf
    patience = args.finetune_early_stop
    print("n accumulation steps:", n_accumulation_steps)
    print("n batches:", n_batches)
    for epoch in range(args.finetune_epochs):
        # print(f"epoch {epoch}")
        total_loss = 0
        n = 0
        for i in range(n_batches):
            # print(i)
            # optimizer.zero_grad()
            # get the batch
            batch_idx = indexes[i * local_batch_size : (i + 1) * local_batch_size]

            batch_inps = train_inps[batch_idx].to(device)
            batch_outputs = train_outputs[batch_idx].to(device)

            with amp.autocast():
                # forward pass
                out, *_ = layer(batch_inps, **layer_kwargs)
                loss = F.mse_loss(out, batch_outputs)
            # print(f"epoch {epoch} batch {i} loss: {loss.item()}")
            if not torch.isfinite(loss):
                raise NANError(
                    "NAN detected in the loss, suggesting increasing the epsilon value"
                )
            # loss.backward()
            scaler.scale(loss / n_accumulation_steps).backward()
            # print out the gradient for the first parameter
            # if i == 4:
            #     print(parameters_to_optimize[list(parameters_to_optimize.keys())[0]].grad)
            #     print(out)
            #     print(batch_outputs)

            total_loss += loss.item()
            if args.log_wandb:
                wandb.log({"loss": loss.item()})
            n += 1

            if (i + 1) % n_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if discrete_update_fn is not None:
                    discrete_update_fn(layer)
            #     scaler.step(optimizer)
            #     scaler.update()

        if val_inps is not None:
            total_loss_val = 0
            n_val = 0
            with torch.no_grad():
                for i in range(n_batches_val):
                    # get the batch
                    batch_inps = val_inps[
                        i * local_batch_size : (i + 1) * local_batch_size
                    ].to(device)
                    batch_outputs = val_outputs[
                        i * local_batch_size : (i + 1) * local_batch_size
                    ].to(device)

                    with amp.autocast():
                        # forward pass
                        out, *_ = layer(batch_inps, **layer_kwargs)
                        loss = F.mse_loss(out, batch_outputs)
                    total_loss_val += loss.item()
                    n_val += 1
            if args.log_wandb:
                wandb.log(
                    {"epoch_loss": total_loss / n, "val_loss": total_loss_val / n_val}
                )
            print(
                f"epoch {epoch} loss: {total_loss/n} val loss: {total_loss_val/n_val}"
            )

        else:
            if args.log_wandb:
                wandb.log({"epoch_loss": total_loss / n})
            print(f"epoch {epoch} loss: {total_loss/n}")
            total_loss_val = total_loss  # a hacky way to avoid the if statement
            n_val = n
        free, total = torch.cuda.mem_get_info(int(device.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        if total_loss_val / n_val > prev_epoch_loss - early_stop_eps:
            patience -= 1
            if patience == 0:
                print("early stopping")
                break
        # otherwise
        else:
            patience = args.finetune_early_stop
            prev_epoch_loss = total_loss_val / n_val
            if args.finetune_keep_best:
                best_weights = deepcopy(parameters_to_optimize)

    if args.finetune_keep_best:
        # print("best weight 1:",list(best_weights.keys())[0], best_weights[list(best_weights.keys())[0]])
        layer.load_state_dict(
            {name: param for name, param in best_weights.items()}, strict=False
        )
    return layer


def cross_entropy_loss(logits, target_logits):
    target_probs = F.softmax(target_logits, -1)
    return -torch.sum(target_probs * F.log_softmax(logits, -1), -1).mean()


@torch.enable_grad()
def finetune_end_to_end(
    model: llama.LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    train_tokens: List[torch.LongTensor],
    train_soft_labels: List[torch.FloatTensor],
    val_tokens: Optional[List[torch.LongTensor]] = None,
    val_soft_labels: Optional[List[torch.FloatTensor]] = None,
    discrete_update_fn: Optional[Callable] = None,
    update_every_n_tokens: int = 4096,
    log_wandb: bool = False,
    device: str = "cuda:0",
    use_tqdm: bool = True,
) -> Tuple[float, Optional[float]]:
    """finetune the model for one epoch"""
    # move everything to cpu first
    # assert model.seqlen % update_every_n_tokens == 0, "update_every_n_tokens must be a multiple of the sequence length"

    total_loss = 0
    n_tokens = 0
    for i in tqdm.tqdm(range(len(train_tokens)), disable=not use_tqdm, desc="Training", miniters=len(train_tokens) // 100):
        tokens = train_tokens[i][0].to(device)
        n_tokens += tokens.shape[1]
        # print("n_tokens", n_tokens, tokens.shape)

        if train_soft_labels is not None:
            labels = train_soft_labels[i].to(
                device
            )  # soft labels from the teacher model
            out = model(tokens)[0]
            loss = cross_entropy_loss(out, labels)
        else:
            loss = model(tokens, labels=tokens)[0]
        loss.backward()
        total_loss += loss.item()
        if log_wandb:
            wandb.log({"train_loss": loss.item()})
        if n_tokens >= update_every_n_tokens:
            print("updating")
            optimizer.step()
            optimizer.zero_grad()
            if discrete_update_fn is not None:
                discrete_update_fn(model)
            n_tokens = 0

    total_loss = total_loss / len(train_tokens)  # * train_tokens[0][0].shape[0])
    if val_tokens is not None:
        print("Warning: validation is not implemented yet")
    if log_wandb:
        wandb.log({"train_loss_batch": total_loss})
    return total_loss, None


@torch.no_grad()
def val_once(model:Union[llama.LlamaForCausalLM, shard.ShardTransformer],
             val_loader:torch.utils.data.DataLoader,
             position_ids:torch.Tensor,
            attention_mask:torch.Tensor,
            loss_fn:Callable) -> float:
    
    model.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in val_loader:
            output = model(
                source,
                position_ids=position_ids,
                attention_mask=attention_mask.float())[:, :-1].contiguous()
            total_loss += nn.CrossEntropyLoss()(
                output.view(-1, output.shape[-1]),
                target.to(0).view(-1, target.shape[-1]),
            )
            ct += 1
    model.train()
    return (total_loss / ct).cpu().item()

@torch.enable_grad()
def finetune_end_to_end_amp(
    model: llama.LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 1,
    log_wandb: bool = False,
    use_tqdm: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    update_freq: int = 1,
    save_fn: Optional[Callable] = None,
    patience: int = -1,
    **kwargs,
) -> Tuple[float, Optional[float]]:
    """finetune the model for one epoch"""
    # move everything to cpu first
    # assert model.seqlen % update_every_n_tokens == 0, "update_every_n_tokens must be a multiple of the sequence length"
    model.float()
    utils.recursive_apply(model, "cache_non_normalized")
    scaler = amp.GradScaler(enabled=True,
                            device = "cuda",
                            growth_factor=1.1,
                            backoff_factor=0.5)

    loss_fn = nn.CrossEntropyLoss()

    best_loss = val_once(model, val_loader,
                         position_ids, attention_mask,
                         loss_fn)
    print(f"initial val loss: {best_loss}")
    
    patience_used = 0

    for epoch in tqdm.tqdm(range(epochs), disable=not use_tqdm):
        count = 0
        train_loss = 0
        for i,(source, target) in enumerate(tqdm.tqdm(train_loader, disable=not use_tqdm, desc=f"epoch {epoch}")):
            tqdm.tqdm.write(f"batch {i}")
            tqdm.tqdm.write(f"source: {source}")
            with amp.autocast(device_type="cuda",
                                 dtype=torch.float16,
                                 enabled=True):
                 tqdm.tqdm.write("model call")
                 output = model(
                      source,
                      position_ids=position_ids,
                      attention_mask=attention_mask,
                 )[:, :-1].contiguous()
                 tqdm.tqdm.write("model call done")
                 tqdm.tqdm.write(f"output: {output}")
                 tqdm.tqdm.write(f"target: {target}")
                 loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                 target.to(0).view(-1, target.shape[-1]))
            # train_loss += loss.item()
            tqdm.tqdm.write(f"loss: {loss}")
            tqdm.tqdm.write("count: " + str(count))
            tqdm.tqdm.write("="*20)
            scaler.scale(loss).backward()
            count += 1
            if count % update_freq == update_freq - 1 or count == len(train_loader) - 1:
                tqdm.tqdm.write("updating")
                utils.recursive_apply(model, "propagate_gradients")
                scaler.step(optimizer)
                scaler.update()
                utils.recursive_apply(model, "cache_non_normalized")
                optimizer.zero_grad()

                #for each gpu
                for gpu in range(torch.cuda.device_count()):
                    tqdm.tqdm.write(f"GPU {gpu}: {utils.get_gpu_memory(gpu, return_str=True)}")



                tqdm.tqdm.write(f"codebook_grad {model.shards[0].layers[0].self_attn.q_proj.quantizer.codebook.grad}")
                tqdm.tqdm.write(f"cached_non_normalized {model.shards[0].layers[0].self_attn.q_proj.quantizer.cached_non_normalized.grad}")
                # raise ValueError

        raise ValueError
        
        print(f"epoch {epoch} train loss: {train_loss/count}")
        if log_wandb:
            wandb.log({"train_loss": train_loss/count})
        if val_loader is not None:
            val_loss = val_once(model, val_loader,
                                position_ids, attention_mask,
                                loss_fn)
            print(f"epoch {epoch} val loss: {val_loss}")
            if log_wandb:
                wandb.log({"val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                patience_used = 0
                if save_fn is not None:
                    save_fn(model)
            else:
                patience_used += 1
                if patience_used == patience:
                    print("early stopping")
                    break

            
