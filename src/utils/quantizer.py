import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.jit as jit
import os
import sys
from typing import Tuple, Optional, Union, List


def round_to_the_nearest(x, codebook):
    return torch.argmin(torch.abs(x.unsqueeze(0) - codebook.unsqueeze(1)), dim=0)


def normalize(weight, norm_order: list[int] = [0, 1]):
    """normalize the input weight matrix
    norm order dictates the order of the norm to use for normalization
    expected to be returned as norms_0, norms_1
    """
    norms = [None, None]
    weight_use = weight.clone()
    for i in norm_order:
        norm_temp = torch.norm(weight_use, dim=i)
        # print(norm_temp)
        norms[i] = norm_temp
        weight_use = weight_use / norm_temp.unsqueeze(i)

    return norms[0], norms[1], weight_use


@jit.script
def cluster_assignment_step(X_reshaped: torch.Tensor, 
                   centriods: torch.Tensor, 
                   weights: torch.Tensor,
                   n_out: int, 
                   n_in: int,
                   norm_0: Optional[torch.Tensor] = None, 
                   norm_1: Optional[torch.Tensor] = None,
                   subblock_size: int = 1024)->torch.LongTensor:
    """
    vector (length d) weighted k-means algorithm to cluser X which is of shape (n_out, n_in) into len(centriods) clusters
    Assignment step
    
    X_reshaped: torch tensor of shape (n_out*n_in/d, d)
    centriods: torch.tensor of the centriods, shape of (k, d)
    weights: torch.tensor of shape (n_in/d, d)
    n_out: int, number of output neurons
    n_in: int, number of input neurons
        norm_0: torch.tensor of shape (n_in) if None, then normalization is not considered
    norm_1: torch.tensor of shape (n_out) if None, then normalization is not considered
    subblock_size: int, for simplicty we will asume that this number is a multiple of n_in
    
    returns: torch.tensor of the assignments, shape of (n_out*n_in/d)
    """

    n_combined, d = X_reshaped.shape
    
    weights_use = torch.tile(weights, (subblock_size*d // n_in, 1)) #shape of (subblock_size, d)
    if norm_0 is not None:
        #de normalize the weights
        # print(weights_use.shape, norm_0.shape)
        weights_use = weights_use * (norm_0.reshape(-1, d).tile((subblock_size*d // n_in,1)))**2
    
    with torch.no_grad():
        assignments = torch.zeros(n_combined, dtype=torch.int64, device=X_reshaped.device)

        for i in range(0, n_combined, subblock_size):
            X_block = X_reshaped[i : i + subblock_size]
            errors = (X_block.unsqueeze(-1) - centriods.T.unsqueeze(0)) ** 2
            # shape of (subblock size, d, k)
            
            if norm_1 is not None:
                weights_rescaled = weights_use.clone()
                for j in range(subblock_size*d//n_in):
                    # print("weights_rescaled[j*n_in//d:(j+1)*n_in//d].numel()",weights_rescaled[j*n_in//d:(j+1)*n_in//d].numel())
                    weights_rescaled[j*n_in//d:(j+1)*n_in//d] *= norm_1[i//n_in + j]**2
                errors = errors * weights_rescaled.unsqueeze(-1)
            else:
                errors = errors * weights_use.unsqueeze(-1)

            # sum by the d
            errors = errors.sum(1)
            # shape of (n, k)
            # print(errors[0,10,:])
            assignments_block = errors.argmin(-1)
            # print(assignments_block[0,10])
            assignments[i : i + subblock_size] = assignments_block
    return assignments


@jit.script
def cluster_update_step(
    X_reshaped: torch.FloatTensor, 
    assignments: torch.LongTensor, 
    weights: torch.FloatTensor,
    n_out: int, 
    n_in: int,
    k: int,
    norm_0: Optional[torch.Tensor] = None, 
    norm_1: Optional[torch.Tensor] = None
)->torch.FloatTensor:
    """
    vector (length d) weighted k-means algorithm to cluser X which is of shape (n_out, n_in) into len(centriods) clusters
    
    cluster update step
    
    input:
        X_reshaped: torch tensor of shape (n_out*n_in/d, d)
        centriods: torch.tensor of the centriods, shape of (k, d)
        weights: torch.tensor of shape (n_in/d, d)
        n_out: int, number of output neurons
        n_in: int, number of input neurons
        k: int, number of clusters
        norm_0: torch.tensor of shape (n_in) if None, then normalization is not considered
        norm_1: torch.tensor of shape (n_out) if None, then normalization is not considered
        subblock_size: int, for simplicty we will asume that this number is a multiple of n_in
    
    returns: torch.tensor of the new centriods, shape of (k, d)
    
    """
    n, d = weights.shape

    # compute the new centriods
    centriods = torch.zeros((k, d), dtype=weights.dtype, device=weights.device)
    if norm_0 is not None:
        weights = weights * norm_0.reshape(-1, d)**2
    # shape of (k,d)
    for i in range(k):
        idxs = torch.where(assignments == i)[0]
        assignment_X = X_reshaped[idxs]  # shape of (n_i,d)
        assignments_weights = weights[idxs % weights.shape[0]]  # shape of (n_i,d)
        if norm_1 is not None:
            assignments_weights *= (norm_1[idxs // n_in]**2).unsqueeze(-1)    
            
        # print(assignments_weights.shape, assignment_X.shape)
        centriods[i] = torch.sum(assignments_weights * assignment_X, dim=0) / torch.sum(
            assignments_weights, dim=0
        )

    return centriods
