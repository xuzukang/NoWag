import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Tuple, Optional, Union, List
import src.utils.compress_parent as compress_parent
import warnings

def loss(reconstructed_weights, original_weights, 
         hessian):
    diff = original_weights - reconstructed_weights
    loss = torch.einsum('ij,jk,ik->', diff, hessian, diff)
    return loss

class dummy_lr_scheduler:
    def __init__(self):
        pass
    def step(self, x):
        pass
@torch.enable_grad()
def align(compression_module:compress_parent.CompressorParent,
          original_weights:torch.FloatTensor,
          train_hessian:torch.FloatTensor,
          val_hessian:Optional[torch.FloatTensor] = None,
          lr:float = 1e-3,
          lr_multiplier:float = 1, #decay the lr by this factor every time the val loss increases
          n_iters:int = 100,
          val_every:int = 1,
          discrete_update_every:int = 1,
          clip_grad:float = -1,
          verbose:Union[bool, int] = 10,
          low_bound:float = 1e-5,
          patience:int = 10,
          patience_scheduler:int = 2,
          eps:float = 1e-5,
          )->compress_parent.CompressorParent:
    """aligns the compression module to the hessian of the training dataset

    Args:
        compression_module (compress_parent.CompressorParent): the compression module we want to aling
        original_weights (torch.FloatTensor): the original weights of the model
        train_hessian (torch.FloatTensor): the hessian of the training dataset
        val_hessian (Optional[torch.FloatTensor], optional): the hessian of the validation dataset, if None, we don't use it. Defaults to None.
        lr (float, optional): the learning rate for the optimizer. Defaults to 1e-3.
        lr_multiplier (float, optional): multiply the learning rate by this factor every time the validation loss increases. Defaults to 1.
        val_every (int, optional): validate the model every this number of iterations on the validation hessian. Defaults to 1.
        discrete_update_every (int, optional): update the discrete variables every this number of iteration. Defaults to 1.
        clip_grad (float, optional): clip the gradient norm to this value. Defaults to -1 in which case we don't clip the gradient.
        verbose (bool, optional): print every this number of iterations. If False, we don't print anything. If True, we print at the end only. Defaults to False.
        low_bound (float, optional): the lower bound for the error, below which we stop training. Defaults to 1e-5.
        patience (int, optional): the patience for the early stop, if the loss has not improved by eps for this number of iterations, we stop training. Defaults to 10.
        patience_scheduler (int, optional): the patience for the learning rate scheduler. Defaults to 2.
        eps (float, optional): the minimum improvement in the loss to consider it as an improvement. Defaults to 1e-5.
    Returns:
        compress_parent.CompressorParent: the aligned compression module
    """
    
    #initialize the optimizer
    optimizer = torch.optim.Adam(compression_module.parameters(), lr=lr)
    if lr_multiplier < 1:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                patience = patience_scheduler, 
                                                                factor = lr_multiplier,
                                                                eps = eps)
    else:
        lr_scheduler = dummy_lr_scheduler()
                                                              

    #initialize the best loss
    best_loss = float('inf')

    patience_counter = 0
    val_loss = None
    
    for i in range(n_iters):

        reconstructed_weights = compression_module.reconstruct()
        train_loss = loss(reconstructed_weights, original_weights, train_hessian)
        
        if i % val_every == 0 and val_hessian is not None:
            with torch.no_grad():
            
                val_reconstructed_weights = compression_module.reconstruct()
                val_loss = loss(val_reconstructed_weights, original_weights, val_hessian).item()
    

            if val_loss < best_loss - eps and val_loss > low_bound:
                best_loss = val_loss
                best_state_dict = copy.deepcopy(compression_module.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    break
            
            if val_loss < low_bound:
                print("early stopping")
                break

            
        if val_hessian is None:
            if train_loss < best_loss - eps and train_loss > low_bound:
                best_loss = train_loss.item()
                best_state_dict = copy.deepcopy(compression_module.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    break
            if train_loss < low_bound:
                print("early stopping")
                break

        optimizer.zero_grad()
        train_loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(compression_module.parameters(), clip_grad)
        optimizer.step()
        if val_loss is not None:
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step(train_loss)
            
        if i%discrete_update_every == 0 and i != 0:
            compression_module.update_discrete()
        
        if verbose and i % verbose == 0:
            print(f'iter {i}, train loss {train_loss.item()}, val loss {val_loss}')
            
    compression_module.load_state_dict(best_state_dict)
    print("best loss", best_loss)
    return compression_module
        
    
