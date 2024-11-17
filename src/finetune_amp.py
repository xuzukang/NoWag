# a simpler version of fine tunining to work on one GPU


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import transformers.models.llama.modeling_llama as modeling_llama
import argparse
import tqdm
import numpy as np
from typing import List, Optional
from copy import deepcopy
import wandb

class NANError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        
@torch.enable_grad()
def finetune_amp(layer: modeling_llama.LlamaDecoderLayer,
                                      train_inps: torch.Tensor,
                                      train_outputs: torch.Tensor,
                                      val_inps: torch.Tensor,
                                      val_outputs: torch.Tensor,
                                      args:argparse.Namespace,
                                      parameters_to_optimize:dict = None,
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

@torch.enable_grad()
def finetune_end_to_end(model: modeling_llama.LlamaModel,
                        teacher: modeling_llama.LlamaModel,
                        train_inps: List[torch.FloatTensor],
                        args:argparse.Namespace,
                        val_inps: Optional[List[torch.FloatTensor]] = None,
                        parameters_to_optimize:List[str] = None, 
                        model_kwargs:dict = {}, early_stop_eps:float = 1e-7,adam_eps:float = 1e-4):
    


    device = args.device
        
    #get the parameters
    if parameters_to_optimize is None:
        parameters_to_optimize = {name: param for name, param in model.named_parameters() if param.requires_grad}
        parameters_not_to_optimize = {name: param for name, param in model.named_parameters() if not param.requires_grad}
        print("the following parameters will not be optimized:", parameters_not_to_optimize.keys())
        print("number of parameters to not optimize:", sum([param.numel() for param in parameters_not_to_optimize.values()]))
    else:
        print("using the provided parameters")
    
    print("optimizing the following parameters:",parameters_to_optimize.keys())

    n_total_params = 0
    for param in parameters_to_optimize.keys():
        print(param, f"{parameters_to_optimize[param].numel():,}")
        n_total_params += parameters_to_optimize[param].numel()
    print("total number of parameters to optimize:", 
            f"{n_total_params:,}")
    
    optimizer = torch.optim.Adam(nn.ParameterList(parameters_to_optimize.values()), lr=args.finetune_lr,
                                 betas = (args.finetune_adam_beta1, args.finetune_adam_beta2),
                                    eps=adam_eps)   
    
    # print("initial parameter",list(parameters_to_optimize.keys())[0], parameters_to_optimize[list(parameters_to_optimize.keys())[0]])
    #set the model to train mode
    model.train()
    teacher.eval()
    
    
    
    n_accumulation_steps = 16
    
    n_batches = len(train_inps)
    if val_inps is not None:
        n_batches_val = len(val_inps)
    
    indexes = torch.randperm(len(train_inps), device=torch.device('cpu'))

    
    scaler = amp.GradScaler()
    prev_epoch_loss = np.inf
    patience = args.finetune_early_stop
    print("n accumulation steps:", n_accumulation_steps)
    print("n batches:", n_batches)

    free, total = torch.cuda.mem_get_info(int(device.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
    

    for epoch in range(args.finetune_epochs):
        # print(f"epoch {epoch}")
        total_loss = 0
        n = 0
        for batch_idx in tqdm.tqdm(indexes):
            batch = train_inps[batch_idx][0].to(device)
            # with amp.autocast():
                #forward pass
            out = model(batch)[0]
            # print(out.dtype)
            with torch.no_grad():
                teacher_out = teacher(batch)[0]
                # print(teacher_out.dtype)
                # print(teacher_out)
            loss = cross_entropy_loss(out, teacher_out)
            if args.log_wandb:
                wandb.log({"loss": loss.item()})
            if not torch.isfinite(loss):
                raise NANError("NAN detected in the loss, suggesting increasing the epsilon value")
            
            loss.backward()
            # scaler.scale(loss/n_accumulation_steps).backward()
            #print out the gradient for the first parameter
            # if i == 4:
            #     print(parameters_to_optimize[list(parameters_to_optimize.keys())[0]].grad)
            #     print(out)
            #     print(batch_outputs)
            
            total_loss += loss.item()
            n += 1
            
            if (n+1) % n_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
       

        if val_inps is not None:
            total_loss_val = 0
            n_val = 0
            with torch.no_grad():
                for val_batch in val_inps:  
                    val_batch = val_batch.to(device)
                    
                    # with amp.autocast():
                        #forward pass
                    out, *_ = model(val_batch)
                    teacher_out, *_ = teacher(val_batch)
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
        model.load_state_dict({name: param for name, param in best_weights.items()}, strict=False)
    return model

def finetune_amp_eps_wrapper(layer: modeling_llama.LlamaDecoderLayer,
                                      train_inps: torch.Tensor,
                                      train_outputs: torch.Tensor,
                                      val_inps: torch.Tensor,
                                      val_outputs: torch.Tensor,
                                      args:argparse.Namespace,
                                      parameters_to_optimize:dict = None,
                                      layer_kwargs:dict = None, 
                                      early_stop_eps:float = 1e-7,
                                      adam_eps_range:tuple = (1e-7, 1e-6, 1e-5, 1e-4)):
    
    for eps in adam_eps_range:
        try:
            return finetune_amp(layer, train_inps, train_outputs, 
                                val_inps, val_outputs,
                                args, parameters_to_optimize, layer_kwargs, early_stop_eps,
                                eps)
        except NANError as e:
            print(e.message)
            print(f"trying the next epsilon value {eps}")
            continue
    print("all epsilon values failed")
    raise NANError("all epsilon values failed")

            
        
            
            

                
                
            
            

    
    
    
    
    
    
    
                                      


    


