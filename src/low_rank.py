import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional,List

from utils.compress_parent import CompressParent

class LowRank(CompressParent):
    def __init__(self, A:torch.FloatTensor,
                    B:torch.FloatTensor,
                    norm_0:Optional[torch.FloatTensor]=None,
                    norm_1:Optional[torch.FloatTensor]=None,
    ):
        """Low Rank Decomposition

        Args:
            A (torch.FloatTensor): A of shape (n_out, low_rank)
            B (torch.FloatTensor): B of shape (low_rank, n_in)
            norms_1 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 0th dimension, of shape n_in. Defaults to None.
            norms_0 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 1st dimension, of shape n_out. Defaults to None.
        """
        super(LowRank, self).__init__({"norm_0":norm_0, "norm_1":norm_1})
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

        self.fast_forward = False #a shortcut forward 

    def forward(self, x:torch.FloatTensor):
        """x is of shape (d,n_in)
        """
        if self.fast_forward:
            if self.norm_0 is not None:
                x = x * self.norm_0.unsqueeze(0)
            
            x = F.linear(F.linear(x, self.B), self.A)

            if self.norm_1 is not None: 
                x = x * self.norm_1.unsqueeze(1)
            return x
        
        else:
            weight = self.reconstruct()
            return F.linear(x, weight)
    
    def reconstruct(self):
        w = self.A @ self.B
        if self.norm_0 is not None:
            w = w * self.norm_0.unsqueeze(0)
        if self.norm_1 is not None:
            w = w * self.norm_1.unsqueeze(1)
        return w


    @staticmethod
    def low_rank_decomposition(weight:torch.FloatTensor, low_rank:int,
                               norm_order:List[int]=[0,1]):
        """Low Rank Decomposition of a weight matrix

        Args:
            weight (torch.FloatTensor): The weight matrix of shape (n_out, n_in)
            low_rank (int): The low rank of the decomposition

        Returns:
            torch.FloatTensor: A of shape (n_out, low_rank)
            torch.FloatTensor: B of shape (low_rank, n_in)
        """
        u, s, v = torch.svd(weight)
        A = u[:, :low_rank] * s[:low_rank]
        B = v[:, :low_rank].T
        return A, B
    

    