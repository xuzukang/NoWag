import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import src.quantizer as quantizer
import numpy as np
import os
from typing import Tuple, Optional, Union, List
from src.quantizers.quantizer_parent import QuantizerParent
import src.utils.compress_parent as compress_parent
import src.alignment.hessian_general_align as hessian_general_align

class LinearQuantized(compress_parent.CompressorParent):
    """A class to pass a linear layer in and get a quantized version of it
        Also servers as our parent class for the quantized linear layers 
    """
    def __init__(self, weight:torch.FloatTensor,
                 bias:Optional[torch.FloatTensor] = None,
                 add_bias:bool = False):
        """quantized linear layer

        Args:
            weight (torch.FloatTensor): the original weight matrix of shape (out_features, in_features)
            bias (Optional[torch.FloatTensor], optional): the original bias vector of shape (out_features) or None.
            add_bias (bool, optional): should we add a bias to the layer or not. Defaults to False.
        """
        
        super(LinearQuantized, self).__init__()
        self.out_features, self.in_features = weight.shape
        self.original_weight = weight.clone()
        self.original_parameters = self.in_features * self.out_features

        if bias is not None:
            self.bias = nn.Parameter(bias.clone(), requires_grad = True)
            self.original_parameters += self.out_features

        else:
            if add_bias:
                self.original_bias = nn.Parameter(torch.zeros(self.out_features), requires_grad = True)
            else:
                self.original_bias = None

        self.quantized = False
        self.quantizer:QuantizerParent = None
        self.log_hessian = False

    def enable_hessian_logging(self):
        """enable hessian logging
        """
        self.log_hessian = True
        self.hessian = torch.zeros(self.original_parameters, self.original_parameters, device=self.original_weight.device,
                                dtype = torch.float32)
        self.n_samples = 0

    def dump_hessian(self)->torch.FloatTensor:
        """gives the hessian calculated and stops logging the inputs for the hessian

        Returns:
            torch.FloatTensor: the hessian
        """
        hessian = self.hessian
        self.log_hessian = False
        del self.hessian
        del self.n_samples
        return hessian
    
    def log_to_hessian_(self, x:torch.FloatTensor):
        """log to the hessian

        Args:
            x (torch.FloatTensor): x is of shape (..., in_features)
        """
        x_flattened = x.reshape(-1, self.in_features)
        n_new_samples = x_flattened.shape[0]
        #multiply the hessian by:
        self.hessian *= self.n_samples/(self.n_samples + n_new_samples)
        #outer product of the flattened x
        #first scale x_flattened
        self.n_samples += n_new_samples
        x_flattened = x_flattened * math.sqrt(2/(self.n_samples * self.in_features))
        self.hessian += x_flattened.T @ x_flattened

    def forward(self, x:torch.FloatTensor):
        """forward pass of the linear layer"""
        if self.log_hessian:
            self.log_to_hessian_(x)
        return F.linear(x, self.reconstruct(), self.original_bias)

        
    def quantize(self, quantizer_class:QuantizerParent, **kwargs):
        """quantize the weights"""
        self.quantizer = quantizer_class.quantize(self.original_weight, self.hessian, **kwargs)
        self.quantized = True

    def reconstruct(self)->torch.FloatTensor:
        """reconstructs the weigth matrix from the quantized version"""
        if self.quantized:
            return self.quantizer()
        else:
            return self.original_weight
        
    def align(self, 
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
          eps:float = 1e-5
    ):
        """aligns the compression module to the hessian of the training dataset

        Args:
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

        """

        hessian_general_align.align(compression_module = self,
                                    original_weights = self.original_weight,
                                    train_hessian = self.hessian,
                                    val_hessian = val_hessian,
                                    lr = lr,
                                    lr_multiplier = lr_multiplier,
                                    n_iters = n_iters,
                                    val_every = val_every,
                                    discrete_update_every = discrete_update_every,
                                    clip_grad = clip_grad,
                                    verbose = verbose,
                                    low_bound = low_bound,
                                    patience = patience,
                                    patience_scheduler = patience_scheduler,
                                    eps = eps)
        
    def update_discrete(self):
        """updates the discrete values of the quantizer"""
        # def recursive_discrete_update(module):
        #     for children in module.children():
        #         if hasattr(children, 'update_discrete'):
        #             if isinstance(getattr(children, 'update_discrete'),callable):
        #                 children.update_discrete()
        #         else:
        #             recursive_discrete_update(children) 
                
        # recursive_discrete_update(self)
        if self.quantized:
            self.quantizer.update_discrete()

    def clean(self):
        if self.quantized:
            self.quantizer.clean()
        if self.log_hessian:
            del self.hessian
            del self.n_samples
            self.log_hessian = False

    def get_n_bits(self):
        if self.quantized:
            return self.quantizer.get_n_bits()
        else:
            return self.original_parameters * 16
    
    def get_n_original_parameters(self):
        return self.original_parameters
    
    



class LowRankSparse(LinearQuantized):
    def __init__(self, weight:torch.FloatTensor,
                 bias:Optional[torch.FloatTensor] = None,
                 add_bias:bool = False):
        """A low rank sparse linear layer

        Args:
            weight (torch.FloatTensor): the original weight matrix of shape (out_features, in_features)
            bias (Optional[torch.FloatTensor], optional): the original bias vector of shape (out_features) or None.
            add_bias (bool, optional): should we add a bias to the layer or not. Defaults to False.
        """
        super(LowRankSparse, self).__init__(weight, bias, add_bias)
        self.low_ranked = False
        self.debug = False
        
    def low_rank(self, rank:int, sparse_frac:float):
        """low rank the weight matrix

        Args:
            rank (int): the rank of the low rank approximation
            sparse_frac (float): the fraction of the weight matrix to be sparse
        """
        self.rank = rank
        self.sparse_frac = sparse_frac
        self.low_ranked = True
        
        norm_0 = torch.norm(self.original_weight, p=2, dim=0)
        norm_1 = torch.norm(self.original_weight, p=2, dim=1)
        
        threshold = torch.kthvalue(torch.cat([norm_0, norm_1]), int(self.sparse_frac * (self.in_features + self.out_features))).values
        
        mask_0 = norm_0 < threshold
        mask_1 = norm_1 < threshold
        self.n_non_sparse_0 = mask_0.sum().item()
        self.n_non_sparse_1 = mask_1.sum().item()
        
        # self.register_buffer("mask", mask_0.unsqueeze(0) & mask_1.unsqueeze(1))
        self.register_buffer("mask_0", mask_0)
        self.register_buffer("mask_1", mask_1)
        
        self.sparse_values_1 = nn.Parameter(self.original_weight[~mask_1], requires_grad = True) #shape of (n_non_sparse_1, in_features)
        self.sparse_values_0 = nn.Parameter(
                                            (self.original_weight[mask_1.unsqueeze(1) & ~mask_0.unsqueeze(0)]
                                                    ).reshape(self.n_non_sparse_0,
                                                              self.in_features - self.n_non_sparse_1))
                                #shape of (n_non_sparse_0, in_features - n_non_sparse_0)
        
        
        non_sparse_weight = self.original_weight[mask_1][:, mask_0]
        
        u,s,v = torch.svd(non_sparse_weight)
        
        A = u[:,:rank] @ torch.sqrt(torch.diag(s[:rank]))
        B = (v[:,:rank] @ torch.sqrt(torch.diag(s[:rank])))
        
        self.A = LinearQuantized(A,None, False)
        self.B = LinearQuantized(B,None, False)
        
    def reconstruct(self):
        
        if self.low_ranked:
            
            non_sparse_weight = self.A.reconstruct() @ self.B.reconstruct().T
            mask = self.mask_0.unsqueeze(0) & self.mask_1.unsqueeze(1)
            return_weight = torch.empty(self.out_features, self.in_features, 
                                        device = non_sparse_weight.device, dtype = non_sparse_weight.dtype)
            return_weight[~mask] = self.sparse_weights
            return_weight[mask] = non_sparse_weight.flatten()
            return return_weight
        else:
            return self.original_weight
        
    def quantize(self, quantizer_class, **kwargs):
        """quantize the weights"""
        if self.low_ranked:
            self.A.quantize(quantizer_class, **kwargs)
            self.B.quantize(quantizer_class, **kwargs)
        else:
            raise ValueError("The module is not low ranked")
        
    def update_discrete(self):
        if self.low_ranked:
            self.A.update_discrete()
            self.B.update_discrete()
            
    def forward(self, x:torch.FloatTensor):
        if self.low_ranked:
            x = x.reshape(-1, self.in_features)
            y = torch.zeros(x.shape[0], self.out_features, device = x.device, dtype = x.dtype)
            y[:,~self.mask_1] += F.linear(x, self.sparse_values_1)
            y[:,self.mask_1] += self.B(self.A(x[:,self.mask_0])) + F.linear(x[:,~self.mask_0], self.sparse_values_0)
            
        else:
            return super(LowRankSparse, self).forward(x)    
        
    
    def enable_hessian_logging(self):
        """enable hessian logging
        """
        self.log_hessian = True
        self.hessian = torch.zeros(self.original_parameters, self.original_parameters, device=self.original_weight.device,
                                dtype = torch.float32)
        self.n_samples = 0
        if self.low_ranked:
            self.A.enable_hessian_logging()
            self.B.enable_hessian_logging()

    def dump_hessian(self)->List[torch.FloatTensor]:
        """gives the hessian calculated and stops logging the inputs for the hessian

        Returns:
            torch.FloatTensor: the hessian
        """
        hessians = [self.hessian]
        self.log_hessian = False
        del self.hessian
        del self.n_samples
        if self.low_ranked:
            hessians.append(self.A.dump_hessian())
            hessians.append(self.B.dump_hessian())
        return hessians
    
    def clean(self):
        if self.low_ranked:
            self.A.clean()
            self.B.clean()
        super(LowRankSparse, self).clean()
        
    def get_n_bits(self):
        return self.A.get_n_bits() + self.B.get_n_bits() + self.sparse_weights.numel() * 16 + 16 * (self.out_features + self.in_features - self.n_non_sparse_0 - self.n_non_sparse_1)
    
    

        



