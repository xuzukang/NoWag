# a simpler version of fine tunining to work on one GPU


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import transformers.models.llama.modeling_llama as modeling_llama
import argparse
import numpy as np
from copy import deepcopy

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
    
    #set the model to train mode
    layer.train()
    
    local_batch_size = args.local_batch_size if args.local_batch_size is not None else args.finetune_batch_size
    #check that the local batch size is a multiple of the number of data points
    assert len(train_inps) % local_batch_size == 0, "the local batch size should be a multiple of the number of data points"
    
    n_accumulation_steps = args.finetune_batch_size//local_batch_size
    #check that the number of accumulation steps is a multiple of the local batch size
    assert args.finetune_batch_size % local_batch_size == 0, "the number of accumulation steps should be a multiple of the local batch size"
    
    n_batches = int(np.ceil(len(train_inps)/local_batch_size))
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
            
            print(f"epoch {epoch} loss: {total_loss/n} val loss: {total_loss_val/n_val}")

        else:    
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
        layer.load_state_dict({name: param for name, param in best_weights.items()}, strict=False)
    return layer

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

            
        
            
            

                
                
            
            

    
    
    
    
    
    
    
                                      


    


