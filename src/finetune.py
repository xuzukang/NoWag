# a simpler version of fine tunining to work on one GPU


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import transformers.models.llama.modeling_llama as llama
import argparse
import tqdm
import numpy as np
from typing import List, Optional, Callable, Tuple
from copy import deepcopy
import wandb
import gc

class NANError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        
@torch.enable_grad()
def finetune_amp(layer: llama.LlamaDecoderLayer,
                                      train_inps: torch.Tensor,
                                      train_outputs: torch.Tensor,
                                      val_inps: torch.Tensor,
                                      val_outputs: torch.Tensor,
                                      args:argparse.Namespace,
                                      parameters_to_optimize:dict = None,
                                      discrete_update_fn: Optional[Callable] = None,
                                      layer_kwargs:dict = None, early_stop_eps:float = 1e-7,adam_eps:float = 1e-7):
    
    
    # if we want to put the fnn on a separate device
    device = args.device
    if args.fnn_device is not None:
        layer.mlp.to(args.fnn_device)
        #add a hook to transfer the inputs to the device and the outputs back to the original device
        layer.mlp.register_forward_pre_hook(lambda module, inputs: inputs.to(args.fnn_device))
        layer.mlp.register_forward_hook(lambda module, inputs, outputs: outputs.to(device))
        
    #get the parameters
    if parameters_to_optimize is None:
        parameters_to_optimize = {name: param for name, param in layer.named_parameters() if param.requires_grad}
        parameters_not_to_optimize = {name: param for name, param in layer.named_parameters() if not param.requires_grad}
        print("the following parameters will not be optimized:", parameters_not_to_optimize.keys())
        print("number of parameters to not optimize:", sum([param.numel() for param in parameters_not_to_optimize.values()]))
    else:
        print("using the provided parameters")
    
    print("optimizing the following parameters:",parameters_to_optimize.keys())
    print("total number of parameters to optimize:", sum([param.numel() for param in parameters_to_optimize.values()]))
    optimizer = torch.optim.Adam(nn.ParameterList(parameters_to_optimize.values()), lr=args.finetune_lr,
                                 betas = (args.finetune_adam_beta1, args.finetune_adam_beta2),
                                    eps=adam_eps)   
    
    # print("initial parameter",list(parameters_to_optimize.keys())[0], parameters_to_optimize[list(parameters_to_optimize.keys())[0]])
    #set the model to train mode
    layer.train()
    
    local_batch_size = args.local_batch_size if args.local_batch_size is not None else args.finetune_batch_size
    #check that the local batch size is a multiple of the number of data points
    assert len(train_inps) % local_batch_size == 0, "the local batch size should be a multiple of the number of data points"
    
    n_accumulation_steps = args.finetune_batch_size//local_batch_size
    #check that the number of accumulation steps is a multiple of the local batch size
    assert args.finetune_batch_size % local_batch_size == 0, "the number of accumulation steps should be a multiple of the local batch size"
    
    n_batches = int(np.ceil(len(train_inps)/local_batch_size))
    if val_inps is not None:
        n_batches_val = int(np.ceil(len(val_inps)/local_batch_size))
    
    indexes = torch.randperm(len(train_inps), device=torch.device('cpu'))
    
    #move the train_inps and train_outputs to cpu to save gpu memory
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
            #get the batch
            batch_idx = indexes[i*local_batch_size:(i+1)*local_batch_size]
            
            batch_inps = train_inps[batch_idx].to(device)
            batch_outputs = train_outputs[batch_idx].to(device)
            
            with amp.autocast():
                #forward pass
                out, *_ = layer(batch_inps, **layer_kwargs)
                loss = F.mse_loss(out, batch_outputs)
            # print(f"epoch {epoch} batch {i} loss: {loss.item()}")
            if not torch.isfinite(loss):
                raise NANError("NAN detected in the loss, suggesting increasing the epsilon value")
            # loss.backward()
            scaler.scale(loss/n_accumulation_steps).backward()
            #print out the gradient for the first parameter
            # if i == 4:
            #     print(parameters_to_optimize[list(parameters_to_optimize.keys())[0]].grad)
            #     print(out)
            #     print(batch_outputs)
            
            total_loss += loss.item()
            if args.log_wandb:
                wandb.log({"loss": loss.item()})
            n += 1
            
            if (i+1) % n_accumulation_steps == 0:
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
                    #get the batch
                    batch_inps = val_inps[i*local_batch_size:(i+1)*local_batch_size].to(device)
                    batch_outputs = val_outputs[i*local_batch_size:(i+1)*local_batch_size].to(device)
                    
                    with amp.autocast():
                        #forward pass
                        out, *_ = layer(batch_inps, **layer_kwargs)
                        loss = F.mse_loss(out, batch_outputs)
                    total_loss_val += loss.item()
                    n_val += 1
            if args.log_wandb:   
                wandb.log({"epoch_loss": total_loss/n, "val_loss": total_loss_val/n_val})
            print(f"epoch {epoch} loss: {total_loss/n} val loss: {total_loss_val/n_val}")

        else:
            if args.log_wandb:
                wandb.log({"epoch_loss": total_loss/n})    
            print(f"epoch {epoch} loss: {total_loss/n}")
            total_loss_val = total_loss # a hacky way to avoid the if statement
            n_val = n
        free, total = torch.cuda.mem_get_info(int(device.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
        if total_loss_val/n_val > prev_epoch_loss - early_stop_eps:
            patience -= 1
            if patience == 0:
                print("early stopping")
                break
        #otherwise
        else:
            patience = args.finetune_early_stop
            prev_epoch_loss = total_loss_val/n_val
            if args.finetune_keep_best:
                best_weights = deepcopy(parameters_to_optimize)
        
    
    if args.finetune_keep_best:
        # print("best weight 1:",list(best_weights.keys())[0], best_weights[list(best_weights.keys())[0]])
        layer.load_state_dict({name: param for name, param in best_weights.items()}, strict=False)
    return layer

def cross_entropy_loss(logits, target_logits):
    target_probs = F.softmax(target_logits, -1)
    return -torch.sum(target_probs * F.log_softmax(logits, -1), -1).mean()


def partition_wise_forwards_and_backwards(model:llama.LlamaForCausalLM, 
                        inps:torch.FloatTensor,
                        targets:torch.FloatTensor,
                        attention_mask:torch.BoolTensor,
                        loss_fn:Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
                        partition_size:int,
                        device:str,
                        )->None:
    """Partition the model into chunks of size partition_size layers to 
    avoid running out of memory, and perform the forward and backward pass

    currently does not support amp, nor fine tuning the embed tokens

    Args:
        model (llama.LlamaForCausalLM): the model to train
        inps (torch.LongTensor): the input hidden states of shape (seq_len, hidden_size)
        target (torch.FloatTensor): the outputs of the teacher model of shape (seq_len, vocab_size)
        loss_fn (Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]): the loss function to use
        partition_size (int): the number of layers to run at once
        device (str): the device to run the model on

    Returns:
        None: None
    """

    #first pass should be through the 


    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers


    partition_inputs:list[torch.FloatTensor] = [inps]
    partition_outputs:list[torch.FloatTensor] = []

    for i in range(0, len(layers), partition_size):
        print("i",i)
        for j in range(i, min(i+partition_size, len(layers))):
            layers[j].to(device)
        
        tmp = partition_inputs[-1].to(device)
        for j in range(i, min(i+partition_size, len(layers))):
            # print(tmp.shape)
            print("j",j)
            tmp = layers[j](tmp.unsqueeze(0), attention_mask = attention_mask)[0][0]
            # print("post layer",tmp.shape)
        
        partition_outputs.append(tmp.to("cpu"))
        partition_inputs.append(tmp.detach().clone().to("cpu").requires_grad_(True).to("cpu"))
        for j in range(i, min(i+partition_size, len(layers))):
            layers[j].to("cpu")
        del tmp
        
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(int(device.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total") 

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    if obj.device == torch.device(device):
                        print(type(obj), obj.size(), obj.device)
            except:
                pass   



    hidden_states = partition_inputs[-1].to(device)
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
        lm_head_input = model.model.norm(hidden_states)
    else:
        lm_head_input = hidden_states
    model.lm_head = model.lm_head.to(device)

    logits = model.lm_head(lm_head_input)

    loss = loss_fn(logits, targets.to(device))
    loss.backward()

    temp_grad = hidden_states.grad.clone()

    #dump all to cpu
    model.lm_head = model.lm_head.to("cpu")
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to("cpu")
    hidden_states = hidden_states.to("cpu")
    logits = logits.to("cpu")
    targets = targets.to("cpu")
    loss = loss.item()
    torch.cuda.empty_cache()

    #flip
    layers = layers[::-1]
    partition_inputs = partition_inputs[::-1]
    partition_outputs = partition_outputs[::-1]
    i_ = 0
    for i in range(0, len(layers), partition_size):

        #move this partition to the device
        for j in range(i, min(i+partition_size, len(layers))):
            layers[j].to(device)
        tmp_outputs = partition_outputs[i_].to(device)
        tmp_inputs = partition_inputs[i_].to(device)
        #from the partition outputs backpropagate the gradients using the temp_grad
        tmp_outputs.backward(temp_grad)

        temp_grad = tmp_inputs.grad.clone()

        #move back to cpu
        for j in range(i, min(i+partition_size, len(layers))):
            layers[j].to("cpu")
        
        tmp_outputs = tmp_outputs.to("cpu")
        tmp_inputs = tmp_inputs.to("cpu")

        torch.cuda.empty_cache()
        i_ += 1
    

@torch.no_grad()
def get_embedded(train_inps,
                    model:llama.LlamaForCausalLM,
                    device:str,
                    dtype:torch.dtype = torch.float32):
    
    #calculate the embeded hidden inputs
    inps =  torch.zeros((len(train_inps), model.seqlen,
                            model.config.hidden_size), device = device,
                            dtype = dtype)

    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    with torch.no_grad():
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.layers[0] = model.model.layers[0].to(device)
        model.model.layers[0] = Catcher(model.model.layers[0])
        for i in range(len(train_inps)):
            try:
                model(train_inps[i][0].to(device))
            except ValueError:
                pass
        model.model.layers[0] = model.model.layers[0].module

        model.model.layers[0] = model.model.layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        inps = inps.cpu()
        torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    return inps, attention_mask



@torch.enable_grad()
def finetune_end_to_end_partitioned(
    model: llama.LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    train_inps: torch.FloatTensor,
    train_outputs: torch.FloatTensor,
    attention_mask: torch.BoolTensor,
    val_inps: Optional[List[torch.FloatTensor]] = None,
    val_outputs: Optional[List[torch.FloatTensor]] = None,
    discrete_update_fn: Optional[Callable] = None,
    update_every_n_tokens:int = 4096,
    log_wandb:bool = False,
    partition_size:int = 4,
    device:str = "cuda:0",
    use_tqdm:bool = True,
)->Tuple[float, Optional[float]]:
    """finetune the model for one epoch using partitioned training to avoid running out of memory
        
    Args:
        model (llama.LlamaForCausalLM): the model to use
        train_inps (List[torch.LongTensor]): the training inputs, of shape (n_samples, seq_len, hidden_size)
        train_outputs (List[torch.FloatTensor]): the output logits of the teacher model of shape (n_samples, seq_len, vocab_size)   
        val_inps (Optional[List[torch.LongTensor]], optional): the validation inputs, of shape (n_samples_val, seq_len, hidden size). Defaults to None.
        val_outputs (Optional[List[torch.FloatTensor]], optional): the validation outputs, of shape (n_samples_val, seq_len, vocab_size). Defaults to None.
        discrete_update_fn (Optional[Callable], optional): the function to call to update the discrete aspects of the compression/quantization strategy, if none then not called. Defaults to None.
        n_epochs (int, optional): the number of epochs to train for. Defaults to 5.
        update_every_n_tokens (int, optional): accumulate the gradients for this many number of tokens and then update, must be a multiple of the sequence length. Defaults to 4096.
        log_wandb (bool, optional): whether to log to wandb. Defaults to False.
        partition_size (int, optional): the size of the partition to compute on GPU. Defaults to 4.
        device (str, optional): the device to do the training on. Defaults to "cuda:0".
        dtype (torch.dtype, optional): the dtype to use for the model. Defaults to torch.float16.
        lr (float, optional): Adam learning rate. Defaults to 1e-4.
        adam_eps (float, optional): the epsilon for adam, if we are doing fp16 training, as usual, then we typically will need to increase this for numerical stability. Defaults to 1e-4.
        return_best (bool, optional): whether to return the last (return_best = False) or the weights for the best model (return_best = True) best is defined by val loss if val_inps is not None otherise the training loss. Defaults to True.

    Raises:
        NANError: _description_

    Returns:
        llama.LlamaForCausalLM: the trained model
    """

    #move everything to cpu first
    model.to("cpu")
    for i in range(len(train_inps)):
        train_inps[i] = train_inps[i].to("cpu")
        train_outputs[i] = train_outputs[i].to("cpu")

    #check that the update_every_n_tokens is a multiple of the sequence length
    assert model.seqlen % update_every_n_tokens == 0, "update_every_n_tokens must be a multiple of the sequence length"
    

    total_loss = 0
    n_tokens = 0
    print('train_inps', train_inps.shape)
    print("attention_mask", attention_mask.shape)
    for i in tqdm.tqdm(range(len(train_inps)), disable = not use_tqdm):
        n_tokens += train_inps[i].shape[0]
        print("train inps.shape", train_inps[i].shape)
        loss = partition_wise_forwards_and_backwards(model,
                                                train_inps[i],
                                                train_outputs[i],
                                                attention_mask=attention_mask,
                                                loss_fn=cross_entropy_loss,
                                                partition_size=partition_size,
                                                device=device)


        total_loss += loss
        if log_wandb:
            wandb.log({"train_loss": loss})
        if n_tokens >= update_every_n_tokens:
            optimizer.step()
            optimizer.zero_grad()
            if discrete_update_fn is not None:
                discrete_update_fn(model)
            n_tokens = 0

    total_loss = total_loss/(len(train_inps) * train_inps.shape[1])
    if val_inps is not None:
        total_loss_val = 0
        with torch.no_grad():
            model.to(device)
            for i in range(len(val_inps)):
                val_inp = val_inps[i][0].to(device)
                val_out = val_outputs[i][0].to(device)
                output_logits = model(val_inp)[0]
                loss = cross_entropy_loss(output_logits, val_out).item()
                total_loss_val += loss
            model.to("cpu") 
        total_loss_val = total_loss_val/(len(val_inps) * val_inps.shape[1])

        if log_wandb:
            wandb.log({"val_loss": total_loss_val, "train_loss_batch": total_loss})
        return total_loss, total_loss_val

    if log_wandb:
        wandb.log({"train_loss_batch": total_loss})
    
    return total_loss, None
    

@torch.enable_grad()
def finetune_end_to_end(
        model: llama.LlamaForCausalLM,
        optimizer: torch.optim.Optimizer,
        train_tokens: List[torch.LongTensor],
        val_tokens: Optional[List[torch.LongTensor]] = None,
        discrete_update_fn: Optional[Callable] = None,
        update_every_n_tokens:int = 4096,
        log_wandb:bool = False,
        device:str = "cuda:0",
        use_tqdm:bool = True,
    )->Tuple[float, Optional[float]]:
    """finetune the model for one epoch"""
        #move everything to cpu first
    assert model.seqlen % update_every_n_tokens == 0, "update_every_n_tokens must be a multiple of the sequence length"
    

    total_loss = 0
    n_tokens = 0
    for i in tqdm.tqdm(range(len(train_tokens)), disable = not use_tqdm):
        n_tokens += train_inps[i].shape[0]
        print("train inps.shape", train_inps[i].shape)
    
    
    
    
    
        
        
        

            

            
            
        


    




        

        


    




# @torch.enable_grad()
# def finetune_end_to_end(model: llama.LlamaForCausalLM,
#                         train_inps: List[torch.FloatTensor],
#                         train_outputs: List[torch.FloatTensor],
#                         args:argparse.Namespace,
#                         val_inps: Optional[List[torch.FloatTensor]] = None,
#                         val_outputs: Optional[List[torch.FloatTensor]] = None,  
#                         discrete_update_fn: Optional[Callable] = None,
#                         parameters_to_optimize:List[str] = None, 
#                         model_kwargs:dict = {}, early_stop_eps:float = 1e-7,adam_eps:float = 1e-4):
    


#     device = args.device
        
#     #get the parameters
#     if parameters_to_optimize is None:
#         parameters_to_optimize = {name: param for name, param in model.named_parameters() if param.requires_grad}
#         parameters_not_to_optimize = {name: param for name, param in model.named_parameters() if not param.requires_grad}
#         print("the following parameters will not be optimized:", parameters_not_to_optimize.keys())
#         print("number of parameters to not optimize:", sum([param.numel() for param in parameters_not_to_optimize.values()]))
#     else:
#         print("using the provided parameters")
    
#     print("optimizing the following parameters:",parameters_to_optimize.keys())

#     n_total_params = 0
#     for param in parameters_to_optimize.keys():
#         print(param, f"{parameters_to_optimize[param].numel():,}")
#         n_total_params += parameters_to_optimize[param].numel()
#     print("total number of parameters to optimize:", 
#             f"{n_total_params:,}")
    
#     optimizer = torch.optim.Adam(nn.ParameterList(parameters_to_optimize.values()), lr=args.finetune_lr,
#                                  betas = (args.finetune_adam_beta1, args.finetune_adam_beta2),
#                                     eps=adam_eps)   
    
#     # print("initial parameter",list(parameters_to_optimize.keys())[0], parameters_to_optimize[list(parameters_to_optimize.keys())[0]])
#     #set the model to train mode
#     model.train()
    
    
    
#     n_accumulation_steps = args.n_accumulation_steps
    
#     n_batches = len(train_inps)
#     if val_inps is not None:
#         n_batches_val = len(val_inps)
    
#     indexes = torch.randperm(len(train_inps), device=torch.device('cpu'))

    
#     scaler = amp.GradScaler()
#     prev_epoch_loss = np.inf
#     patience = args.finetune_early_stop
#     print("n accumulation steps:", n_accumulation_steps)
#     print("n batches:", n_batches)

#     free, total = torch.cuda.mem_get_info(int(device.split(":")[1]))
#     print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
    

#     for epoch in range(args.finetune_epochs):
#         # print(f"epoch {epoch}")
#         total_loss = 0
#         n = 0
#         for batch_idx in tqdm.tqdm(indexes):
#             batch = train_inps[batch_idx][0].to(device)
#             teacher_out = train_outputs[batch_idx][0].to(device)
#             # with amp.autocast():
#                 #forward pass
#             out = model(batch)[0]
#             loss = cross_entropy_loss(out, teacher_out)
#             if args.log_wandb:
#                 wandb.log({"loss": loss.item()})
#             if not torch.isfinite(loss):
#                 raise NANError("NAN detected in the loss, suggesting increasing the epsilon value")
            
#             loss.backward()
#             # scaler.scale(loss/n_accumulation_steps).backward()
#             #print out the gradient for the first parameter
#             # if i == 4:
#             #     print(parameters_to_optimize[list(parameters_to_optimize.keys())[0]].grad)
#             #     print(out)
#             #     print(batch_outputs)
            
#             total_loss += loss.item()
#             n += 1
            
#             if (n+1) % n_accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 if discrete_update_fn is not None:
#                     discrete_update_fn(model)
       

#         if val_inps is not None:
#             total_loss_val = 0
#             n_val = 0
#             with torch.no_grad():
#                 for val_batch_index in range(len(val_inps)): 
#                     val_batch = val_inps[val_batch_index][0].to(device)
#                     val_teacher_out = val_outputs[val_batch_index][0].to(device)
                    
#                     # with amp.autocast():
#                         #forward pass
#                     out = model(val_batch)[0]
#                     loss = cross_entropy_loss(out, val_teacher_out)
#                     total_loss_val += loss.item()
#                     n_val += 1
#             if args.log_wandb:   
#                 wandb.log({"epoch_loss": total_loss/n, "val_loss": total_loss_val/n_val})
#             print(f"epoch {epoch} loss: {total_loss/n} val loss: {total_loss_val/n_val}")

#         else:    
#             if args.log_wandb:
#                 wandb.log({"epoch_loss": total_loss/n})
#             print(f"epoch {epoch} loss: {total_loss/n}")
#             total_loss_val = total_loss # a hacky way to avoid the if statement
#             n_val = n
#         free, total = torch.cuda.mem_get_info(int(device.split(":")[1]))
#         print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
#         if total_loss_val/n_val > prev_epoch_loss - early_stop_eps:
#             patience -= 1
#             if patience == 0:
#                 print("early stopping")
#                 break
#         #otherwise
#         else:
#             patience = args.finetune_early_stop
#             prev_epoch_loss = total_loss_val/n_val
#             if args.finetune_keep_best:
#                 best_weights = deepcopy(parameters_to_optimize)
        
    
#     if args.finetune_keep_best:
#         # print("best weight 1:",list(best_weights.keys())[0], best_weights[list(best_weights.keys())[0]])
#         model.load_state_dict({name: param for name, param in best_weights.items()}, strict=False)
#     return model

# # def finetune_amp_eps_wrapper(layer: modeling_llama.LlamaDecoderLayer,
# #                                       train_inps: torch.Tensor,
# #                                       train_outputs: torch.Tensor,
# #                                       val_inps: torch.Tensor,
# #                                       val_outputs: torch.Tensor,
# #                                       args:argparse.Namespace,
# #                                       parameters_to_optimize:dict = None,
# #                                       layer_kwargs:dict = None, 
# #                                       early_stop_eps:float = 1e-7,
# #                                       adam_eps_range:tuple = (1e-7, 1e-6, 1e-5, 1e-4)):
    
# #     for eps in adam_eps_range:
# #         try:
# #             return finetune_amp(layer, train_inps, train_outputs, 
# #                                 val_inps, val_outputs,
# #                                 args, parameters_to_optimize, layer_kwargs, early_stop_eps,
# #                                 eps)
# #         except NANError as e:
# #             print(e.message)
# #             print(f"trying the next epsilon value {eps}")
# #             continue
# #     print("all epsilon values failed")
# #     raise NANError("all epsilon values failed")

            
        
            
            

                
                
            
            

    
    
    
    
    
    
    
                                      


    


