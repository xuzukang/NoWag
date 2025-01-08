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

class Normalizer(nn.Module):
    pass 

class Normalizer(nn.Module):

    def __init__(self, 
                 norms: List[Union[torch.Tensor|None]],
                 zeros: List[Union[torch.Tensor|None]],
                 norm_order: List[int] = [0, 1],
                 ):
        super(Normalizer, self).__init__()
        self.norm_order = norm_order

        self.norms = nn.ParameterList([nn.Parameter(norm) for norm in norms])
        self.zeros = nn.ParameterList([nn.Parameter(zero) for zero in zeros])

        

    def denormalize(self, normalized_weight:torch.FloatTensor)->torch.FloatTensor:
        """denormalize the input weight matrix"""

        #we have to do this in reverse
        denormalized_weight = normalized_weight.clone()
        for i in reversed(self.norm_order):
            if self.norms[i] is not None and self.norms[i].numel() > 0:
                denormalized_weight = denormalized_weight * self.norms[i].unsqueeze(i)
            if self.zeros[i] is not None and self.zeros[i].numel() > 0:
                # print(self.zeros[i].shape)
                denormalized_weight = denormalized_weight + self.zeros[i].unsqueeze(i)
            
        return denormalized_weight

    def denormalize_codebook(self, normalized_codebook:torch.FloatTensor, subblock:list[list[int]])->torch.FloatTensor:
        """denormalize the input codebook

        Args:
            normalized_codebook (torch.FloatTensor): tensor of shape (1, d, n_codes)
            subblock (list[list[int]]): list of the sublock dimensions of shape 
            [[i_start, i_end], [j_start, j_end], ...]

        Returns:
            torch.FloatTensor: _description_
        """
        denormalized_subblock = normalized_codebook.clone()
        # print(normalized_codebook.shape)
        # print("subblock", subblock)
        subblock = subblock[::-1]
        for i in reversed(self.norm_order):
            # print("i", i)
            idx_start, idx_end = subblock[i]
            # print("idx_start, idx_end", idx_start, idx_end)
            if self.norms[i] is not None and self.norms[i].numel() > 0:
                denormalized_subblock = denormalized_subblock * self.norms[i][idx_start:idx_end].unsqueeze(i).unsqueeze(-1)
            if self.zeros[i] is not None and self.zeros[i].numel() > 0:
                denormalized_subblock = denormalized_subblock + self.zeros[i][idx_start:idx_end].unsqueeze(i).unsqueeze(-1)
            
        return denormalized_subblock
    
    def normalize(self, weight:torch.FloatTensor)->torch.FloatTensor:
        """normalize the input weight matrix"""
        normalized_weight = weight.clone()
        for i in self.norm_order:
            if self.zeros[i] is not None and self.zeros[i].numel() > 0:
                normalized_weight = normalized_weight - self.zeros[i].unsqueeze(i)
            if self.norms[i] is not None and self.norms[i].numel() > 0:
                normalized_weight = normalized_weight / self.norms[i].unsqueeze(i)
        
        return normalized_weight
    
    @staticmethod
    def normalize_init(weight:torch.FloatTensor, 
                  norm_order:list[int],
                  zero:list[bool],
                  eps:float = 1e-5)->Tuple[Normalizer, torch.FloatTensor]:
        
        norms = [None] * len(weight.shape)
        zeros = [None] * len(weight.shape)

        for dim in norm_order:
            if zero[dim]:
                zeros[dim] = torch.mean(weight, dim=dim)
                weight = weight - zeros[dim].unsqueeze(dim)
            norms[dim] = torch.norm(weight, dim=dim) + eps
            weight = weight / norms[dim].unsqueeze(dim)
            assert torch.all(torch.isfinite(weight))
            assert torch.all(torch.isfinite(norms[dim]))
        
        return Normalizer(norms, zeros, norm_order), weight
    
    def get_n_bits(self):
        n_bits = 0
        for norm in self.norms:
            if norm is not None:
                n_bits += norm.numel() * 16
        for zero in self.zeros:
            if zero is not None:
                n_bits += zero.numel() * 16
        return n_bits


        



# def normalize(weight, norm_order: list[int] = [0, 1]):
#     """normalize the input weight matrix
#     norm order dictates the order of the norm to use for normalization
#     expected to be returned as norms_0, norms_1
#     """
#     norms = [None, None]
#     # print(torch.max(torch.mean(weight, dim=0)/torch.mean(torch.abs(weight),dim=0)))
#     # print(torch.max(torch.mean(weight, dim=1)/torch.mean(torch.abs(weight),dim=1)))
#     weight_use = weight.clone()
#     for i in norm_order:
#         norm_temp = torch.norm(weight_use, dim=i)
#         # print(norm_temp)
#         norms[i] = norm_temp
#         weight_use = weight_use / norm_temp.unsqueeze(i)

#     return norms[0], norms[1], weight_use

# def denormalize(normalized_weights, norms
class QuickStopException(Exception):
    pass


def find_optimal_subblock_size(X_reshaped: torch.Tensor, 
                   centriods: torch.Tensor, 
                   weights: torch.Tensor,
                   n_out: int, 
                   n_in: int,
                   norm_0: torch.Tensor = torch.empty(0), 
                   norm_1: torch.Tensor = torch.empty(0)):

    d = centriods.shape[1]

    subblock_base_size = n_in//d

    subblock_multiple_range = [1,n_out]

    while subblock_multiple_range[1] - subblock_multiple_range[0] > 1:
        print("subblock_multiple_range", subblock_multiple_range)
        subblock_size = (subblock_multiple_range[0] + subblock_multiple_range[1]) // 2
        try:
            cluster_assignment_step(X_reshaped, centriods, weights, n_out, n_in, norm_0, norm_1, subblock_size*subblock_base_size, True)
            subblock_multiple_range[0] = subblock_size
        except RuntimeError:
            subblock_multiple_range[1] = subblock_size
        torch.cuda.empty_cache()
    print("subblock_multiple_range", subblock_multiple_range)

    free, total = torch.cuda.mem_get_info(int(str(X_reshaped.device).split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
    return subblock_multiple_range[0]*subblock_base_size

@jit.script
def cluster_assignment_step(X_reshaped: torch.Tensor, 
                   centriods: torch.Tensor, 
                   weights: torch.Tensor,
                   n_out: int, 
                   n_in: int,
                   norm_0: torch.Tensor = torch.empty(0), 
                   norm_1: torch.Tensor = torch.empty(0),
                   subblock_size: int = 1024,
                   quick_stop:bool = False)->torch.LongTensor:
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
    # print("subblock_size", subblock_size, "d", d, "n_in", n_in)
    # print(subblock_size*d // n_in)
    weights_use = torch.tile(weights, (subblock_size*d // n_in, 1)) #shape of (subblock_size, d)
    # print("norm_0", norm_0, "norm_1", norm_1)
    # print(norm_0)
    if norm_0.numel() > 0:
        #de normalize the weights
        # print(weights_use.shape, norm_0.shape)
        weights_use = weights_use * (norm_0.reshape(-1, d).tile((subblock_size*d // n_in,1)))**2
    
    ignore_norm_1 = norm_1.numel() == 0
    with torch.no_grad():
        assignments = torch.zeros(n_combined, dtype=torch.int64, device=X_reshaped.device)

        for i in range(0, n_combined, subblock_size):
            X_block = X_reshaped[i : i + subblock_size]
            errors = (X_block.unsqueeze(-1) - centriods.T.unsqueeze(0)) ** 2
            # shape of (subblock size, d, k)
            if not ignore_norm_1:
                weights_rescaled = weights_use.clone()
                for j in range(subblock_size*d//n_in):
                    # print("weights_rescaled[j*n_in//d:(j+1)*n_in//d].numel()",weights_rescaled[j*n_in//d:(j+1)*n_in//d].numel())
                    weights_rescaled[j*n_in//d:(j+1)*n_in//d] *= norm_1[i//n_in + j]**2
                errors = errors * weights_rescaled.unsqueeze(-1)
            else:
                # print(errors.shape, weights_use.shape)
                errors = errors * weights_use.unsqueeze(-1)

            # sum by the d
            errors = errors.sum(1)
            # shape of (n, k)
            # print(errors[0,10,:])
            assignments_block = errors.argmin(-1)
            # print(assignments_block[0,10])
            assignments[i : i + subblock_size] = assignments_block
            # if quick_stop:
            #     return assignments
    return assignments


@jit.script
def cluster_update_step(
    X_reshaped: torch.FloatTensor, 
    assignments: torch.LongTensor, 
    weights: torch.FloatTensor,
    n_out: int, 
    n_in: int,
    k: int,
    norm_0: torch.Tensor = torch.empty(0), 
    norm_1: torch.Tensor = torch.empty(0),
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
    with torch.no_grad():
        n, d = weights.shape

        # compute the new centriods
        centriods = torch.zeros((k, d), dtype=weights.dtype, device=weights.device)
        if norm_0.numel() > 0:
            weights = weights * norm_0.reshape(-1, d)**2
        # shape of (k,d)
        ignore_norm_1 = norm_1.numel() == 0 
        for i in range(k):
            idxs = torch.where(assignments == i)[0]
            assignment_X = X_reshaped[idxs]  # shape of (n_i,d)
            assignments_weights = weights[idxs % weights.shape[0]]  # shape of (n_i,d)
            if not ignore_norm_1:
                assignments_weights *= (norm_1[idxs // n_in]**2).unsqueeze(-1)    
                
            # print(assignments_weights.shape, assignment_X.shape)
            centriods[i] = torch.sum(assignments_weights * assignment_X, dim=0) / torch.sum(
                assignments_weights, dim=0
            )

        return centriods
