import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import argparse

def align_low_rank(original_weight:torch.Tensor,
                                 reconstruct_fn:nn.Module|callable,
                                 reconstruct_params:dict[str, torch.Tensor],
                                 hessian:torch.Tensor,
                                 regularization_lambda:float,
                                 epochs:int,
                                 lr:float,
                                 parameters_to_exclude:list[str]=[],
                                 verbose:int = -1)->dict[str, torch.Tensor]:
    """aligns the low rank to be close to the original weight on the callibration dataset

    Args:
        original_weight (torch.Tensor): _description_
        reconstruct_fn (nn.Module | callable): _description_
        reconstruct_params (dict[str, torch.Tensor]): _description_
        hessian (torch.Tensor): _description_
        args (argparse.Namespace): _description_
        parameters_to_exclude (list[str], optional): _description_. Defaults to [].
        verbose (int, optional): _description_. Defaults to -1.
    """
    
    
    
    params_to_optimize = {name: param for name, param in reconstruct_params.items() if param.requires_grad and name not in parameters_to_exclude}

    
    optimizer = torch.optim.Adam(nn.ParameterList(params_to_optimize.values())
                                 , lr = lr)

    hessain_to_use = hessian.clone()/hessian.shape[0]
    ids = torch.arange(hessian.shape[0], device=hessian.device, dtype = hessian.dtype)
    hessain_to_use[ids, ids] += regularization_lambda
    

    for epoch in range(epochs):

        optimizer.zero_grad()
        
        reconstructed_weights = reconstruct_fn(**reconstruct_params)
        
        diff = original_weight - reconstructed_weights

        loss = torch.einsum('ij,jk,ik->', diff, hessain_to_use, diff)

        loss.backward()
        optimizer.step()
        
        if verbose > 0 and epoch % verbose == 0:
            print(f"Epoch {epoch} Loss {loss.item()}")
        
        if loss < 0:
            print("breaking because the loss is negative")
        
    return reconstruct_params
