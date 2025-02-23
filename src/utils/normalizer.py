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
                 shape:Tuple[int,int] = (0,0)
                 ):
        super(Normalizer, self).__init__()
        self.norm_order = norm_order

        self.norms = nn.ParameterList([nn.Parameter(norm) for norm in norms])
        self.zeros = nn.ParameterList([nn.Parameter(zero) for zero in zeros])

        self.original_shape = shape
        
    def denormalize(self, normalized_weight:torch.FloatTensor, debias = True)->torch.FloatTensor:

        reversed_norm_order = [1,0]
        # print("reversed_norm_order", reversed_norm_order)
        # print("norm_order", self.norm_order)
        # for norm in self.norms:
        #     # print(norm.shape)
        for i in reversed(self.norm_order):
            # print("i", i)
            # print("reversed_norm_order[i]", reversed_norm_order[i])
            if self.norms[i] is not None and self.norms[i].numel() > 0:
                # print(self.norms[i].shape)
                # print(normalized_weight.shape[reversed_norm_order[i]])
                # print(self.norms[i][:normalized_weight.shape[reversed_norm_order[i]]].unsqueeze(i).shape)
                normalized_weight = normalized_weight * self.norms[i].unsqueeze(i)
            if self.zeros[i] is not None and self.zeros[i].numel() > 0 and debias:
                normalized_weight = normalized_weight + self.zeros[i].unsqueeze(i)
            
        return normalized_weight
    
    def denormalize_otf_in(self, input_activation:torch.FloatTensor)->torch.FloatTensor:
        if 0 in self.norm_order:
            idx = self.norm_order.index(0)
            if self.norms[idx] is not None and self.norms[idx].numel() > 0:
                input_activation = input_activation * self.norms[idx]
        return  input_activation
    
    def denormalize_otf_out(self, output_activation:torch.FloatTensor)->torch.FloatTensor:
        if 1 in self.norm_order:
            idx = self.norm_order.index(1)
            if self.norms[idx] is not None and self.norms[idx].numel() > 0:
                output_activation = output_activation * self.norms[idx]
            if self.zeros[idx] is not None and self.zeros[idx].numel() > 0:
                raise ValueError("not implemented")
            
        return output_activation
    
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
                  norm_order:list[int] = [0, 1],
                  zero:list[bool]= [True, True],
                  eps:float = 1e-5,
                  norm_rescale:float = True,
                  powers:float = 1,
                  p:float = 2
                  )->Tuple[Normalizer, torch.FloatTensor]:
        
        norms = [None] * len(weight.shape)
        zeros = [None] * len(weight.shape)
        
        n_out, n_in = weight.shape
        # print(zero)
        # print("norm_order", norm_order)
        for dim in norm_order:
            # print("dim",dim)
            if zero[dim]:
                zeros[dim] = torch.mean(weight, dim=dim)
                if norm_rescale:
                    weight = weight - zeros[dim].unsqueeze(dim)
                # weight = weight - zeros[dim].unsqueeze(dim)
            norms[dim] = torch.norm(weight, dim=dim, p = p
                                    )**powers + eps
            if norm_rescale:
                weight = weight / norms[dim].unsqueeze(dim)
            assert torch.all(torch.isfinite(weight))
            assert torch.all(torch.isfinite(norms[dim]))
            
        normalizer = Normalizer(norms, zeros, norm_order,(n_out,n_in))
        
        if not norm_rescale:
            weight = normalizer.normalize(weight)
        
        return normalizer, weight
    
    @staticmethod
    def blank_recreate(weight,
                       norm_order:List[int] = [0, 1], 
                        zero:List[bool] = [False, False]):
        
        norms = [None] * len(weight.shape)
        zeros = [None] * len(weight.shape)

        n_out, n_in = weight.shape

        for dim in norm_order:
            shape = [s for i,s in enumerate(weight.shape) if i != dim]
            if zero[dim]:
                zeros[dim] = torch.zeros(shape).to(weight.device)
            norms[dim] = torch.ones(shape).to(weight.device)
        
        return Normalizer(norms, zeros, norm_order, (n_out, n_in))
    
    
    def get_n_bits(self):
        n_bits = 0
        for norm in self.norms:
            if norm is not None:
                n_bits += norm.numel() * 16
        for zero in self.zeros:
            if zero is not None:
                n_bits += zero.numel() * 16
        return n_bits


        
