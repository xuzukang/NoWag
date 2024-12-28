import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
from typing import Tuple, Optional, Union, List
from src.linear_compress import LinearQuantized
from src.tensor_compress import LinearTensorized
from src.utils.compress_parent import CompressorParent


class JointCompressor(LinearQuantized):
    
    def __init__(
        self,
        weight: torch.FloatTensor,
        bias: Optional[torch.FloatTensor] = None,
        add_bias: bool = False,
    ):
        """quantized linear layer

        Args:
            weight (torch.FloatTensor): the original weight matrix of shape (out_features, in_features)
            bias (Optional[torch.FloatTensor], optional): the original bias vector of shape (out_features) or None.
            add_bias (bool, optional): should we add a bias to the layer or not. Defaults to False.
        """

        super(JointCompressor, self).__init__(weight, bias, add_bias)
    
                
        self.compressed = False
        
        
    def initalize_2_compressors(self, quantization_compression_algorithm, 
                 quantization_kwargs, tensor_compression_algorithm, tensor_compression_kwargs):
        """compresses the weight matrix"""
        W_remaining = self.original_weight.clone()
        self.initalize_quantization_(W_remaining, quantization_compression_algorithm, quantization_kwargs)
        with torch.no_grad():
            W_remaining = self.original_weight.clone() - self.quantization_compressor.reconstruct()
        self.initalize_tensorization_(W_remaining, tensor_compression_algorithm, tensor_compression_kwargs)
        self.compressed = True

        

    def initalize_quantization_(self, W_remaining, quantization_compression_algorithm,
                                 quantization_kwargs = None):
        if quantization_kwargs is None:
            assert hasattr(self, "quantization_compressor_kwargs"), "quantization_kwargs is not provided"
            quantization_kwargs = self.quantization_compressor_kwargs
            
        self.quantization_compressor:LinearQuantized = quantization_compression_algorithm(W_remaining)
        self.quantization_compressor.hessian = self.hessian
        self.quantization_compressor.quantize(**quantization_kwargs)
        self.quantization_compressor_kwargs = quantization_kwargs

    def initalize_tensorization_(self, W_remaining, tensor_compression_algorithm,
                                 tensor_compression_kwargs = None):
        if tensor_compression_kwargs is None:
            assert hasattr(self, "tensor_compressor_kwargs"), "tensor_compression_kwargs is not provided"
            tensor_compression_kwargs = self.tensor_compressor_kwargs

        self.tensor_compressor:LinearTensorized = tensor_compression_algorithm(W_remaining)
        self.tensor_compressor.hessian = self.hessian
        self.tensor_compressor.tensor_decompose(**tensor_compression_kwargs)
        self.tensor_compressor_kwargs = tensor_compression_kwargs

        # self.frozen_tensor_compressor:LinearTensorized = tensor_compression_algorithm(W_remaining)
        # self.frozen_tensor_compressor.hessian = self.hessian
        # self.frozen_tensor_compressor.tensor_decompose(**tensor_compression_kwargs)
        # self.frozen_tensor_compressor.load_state_dict(self.tensor_compressor.state_dict())
        
        # for param in self.frozen_tensor_compressor.parameters():
        #     param.requires_grad = False


        # best_loss = float("inf")
        # prev_tensor_compress_state_dict = None
        # for iter in range(n_iters):
        #     self.quantization_compressor:LinearQuantized = quantization_compression_algorithm(W_remaining)
        #     self.quantization_compressor.hessian = self.hessian
        #     self.quantization_compressor.quantize(**quantization_kwargs)
        #     self.quantization_compressor.set_additional_attributes_as_trainable()
        #     self.quantization_compressor.align(**quantization_align_kwargs)
        #     with torch.no_grad():
        #         W_remaining = self.original_weight.clone() - self.quantization_compressor.reconstruct()
        #     torch.save(W_remaining,"test/W_remaining.pt")
            
        #     self.tensor_compressor:LinearTensorized = tensor_compression_algorithm(W_remaining)
        #     self.tensor_compressor.hessian = self.hessian
        #     self.tensor_compressor.tensor_decompose(**tensor_compression_kwargs)
        #     self.tensor_compressor.set_additional_attributes_as_trainable()
        #     if prev_tensor_compress_state_dict is not None:
        #         self.tensor_compressor.load_state_dict(prev_tensor_compress_state_dict)
        #     loss = self.tensor_compressor.align(**tensor_compression_align_kwargs)
        #     prev_tensor_compress_state_dict = self.tensor_compressor.state_dict()
        #     with torch.no_grad():
        #         W_remaining = self.original_weight.clone() - self.tensor_compressor.reconstruct()
        #     if loss < best_loss:
        #         best_loss = loss    
            
        # self.compressed = True
        # return best_loss
    
    def alternating_align(self, n_iters:int, 
                          quantization_align_kwargs:dict, 
                          tensor_compression_align_kwargs:dict,
                          reinitialize_quantization:bool = False,
                          reinitialize_tensorization:bool = False,
                          set_additional_attributes_as_trainable:bool = True,
                          **kwargs):
        """alternate between freezing the quantization and tensorization and aligning the other"""

        best_loss = float("inf")
        prev_tensor_compress_state_dict = None

        W_remaining = self.original_weight.clone()

        if set_additional_attributes_as_trainable:
            self.quantization_compressor.set_additional_attributes_as_trainable()
            self.tensor_compressor.set_additional_attributes_as_trainable()


        for iter in range(n_iters):
            print("-"*10, "iter", iter, "-"*10)
            if reinitialize_quantization:
                self.initalize_quantization_(W_remaining, self.quantization_compressor.__class__,)
                if set_additional_attributes_as_trainable:
                    self.quantization_compressor.set_additional_attributes_as_trainable()
                self.quantization_compressor.hessian = self.hessian

            self.quantization_compressor.align(**quantization_align_kwargs)
            with torch.no_grad():
                W_remaining = self.original_weight - self.quantization_compressor.reconstruct()
            
            if reinitialize_tensorization:
                self.initalize_tensorization_(W_remaining, self.tensor_compressor.__class__)
            
                if set_additional_attributes_as_trainable:
                    self.tensor_compressor.set_additional_attributes_as_trainable()
                
                self.tensor_compressor.hessian = self.hessian

            loss = self.tensor_compressor.align(**tensor_compression_align_kwargs)
            with torch.no_grad():
                W_remaining = self.original_weight - self.tensor_compressor.reconstruct()
            if loss < best_loss:
                best_loss = loss
                best_quantization_state_dict = self.quantization_compressor.state_dict()
                best_tensor_compress_state_dict = self.tensor_compressor.state_dict()

        self.quantization_compressor.load_state_dict(best_quantization_state_dict)
        self.tensor_compressor.load_state_dict(best_tensor_compress_state_dict)
        # self.compressed = True
        return best_loss
    
    def warmup_tensorization(self, 
                        lr: Union[float, dict[str, float]] = 1e-3,
                        lr_multiplier: float = 1,  # decay the lr by this factor every time the val loss increases
                        n_iters: int = 100,
                        clip_grad: float = -1,
                        verbose: Union[bool, int] = 10,
                        low_bound: float = 1e-5,
                        patience: int = 10,
                        patience_scheduler: int = 2,
                        eps: float = 1e-5,
                        **kwargs,
                        ):  
        """warmup the tensorization"""
        self.tensor_compressor.set_additional_attributes_as_trainable()
        self.tensor_compressor.align(
            lr_multiplier=lr_multiplier,
            lr_warmup = lr,
            n_iters = 0,
            n_iters_warmup_task = n_iters,
            clip_grad = clip_grad,
            verbose = verbose,
            low_bound = low_bound,
            patience = patience,
            patience_scheduler = patience_scheduler,
            eps = eps,
            **kwargs,
        )



    def joint_align(
        self,
        val_hessian: Optional[torch.FloatTensor] = None,
        lr: Union[float, dict[str, float]] = 1e-3,
        lr_multiplier: float = 1,  # decay the lr by this factor every time the val loss increases
        n_iters: int = 100,
        val_every: int = 1,
        discrete_update_every: int = 1,
        clip_grad: float = -1,
        verbose: Union[bool, int] = 10,
        low_bound: float = 1e-5,
        patience: int = 10,
        patience_scheduler: int = 2,
        eps: float = 1e-5,
        **kwargs,
    ):
        
        return self.align(
            val_hessian=val_hessian,
            lr=lr,
            lr_multiplier=lr_multiplier,
            n_iters=n_iters,
            val_every=val_every,
            discrete_update_every=discrete_update_every,
            clip_grad=clip_grad,
            verbose=verbose,
            low_bound=low_bound,
            patience=patience,
            patience_scheduler=patience_scheduler,
            eps=eps,
            **kwargs,
        )
        

        

    def update_discrete(self):
        """updates the discrete parameters"""
        
        if self.compressed:
            print("here")

            #update the reference weights
            with torch.no_grad():
                reference_weights =  (self.original_weight - self.tensor_compressor.reconstruct())
                self.quantization_compressor.quantizer.reference_weight = reference_weights
                self.quantization_compressor.quantizer.update_discrete()
    

    
    def reconstruct(self):
        """reconstructs the original weight matrix"""
        if not self.compressed:
            return self.original_weight
        else:
            return self.quantization_compressor.reconstruct() + self.tensor_compressor.reconstruct() #- self.frozen_tensor_compressor.reconstruct()
        
    def forward(self, x: torch.FloatTensor):
        """forward pass of the layer"""
        if not self.compressed:
            return F.linear(x, self.original_weight, self.bias)
        else:
            return self.quantization_compressor(x) + self.tensor_compressor(x)
        
    
    def blank_recreate(self, quantization_compression_algorithm,
                        quantization_kwargs,
                        tensor_compression_algorithm,
                        tensor_compression_kwargs,
                        **kwargs):
        """initializes a blank quantizer with the same shape to be filled by
        a state_dict"""
        self.quantization_compressor = quantization_compression_algorithm(self.original_weight)
        self.quantization_compressor.blank_recreate(**quantization_kwargs)
        
        self.tensor_compressor = tensor_compression_algorithm(self.original_weight)
        self.tensor_compressor.blank_recreate(**tensor_compression_kwargs)
        self.compressed = True
        
    
    def get_n_bits(self):
        return self.quantization_compressor.get_n_bits() + self.tensor_compressor.get_n_bits()
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        quantizer_compressor_state_dict = {k.split("quantization_compressor.")[1]:v for k,v in state_dict.items() if "quantization_compressor" in k}
        tensor_compressor_state_dict = {k.split("tensor_compressor.")[1]:v for k,v in state_dict.items() if "tensor_compressor" in k}
        self.quantization_compressor.load_state_dict(quantizer_compressor_state_dict, strict = strict, assign = assign)
        self.tensor_compressor.load_state_dict(tensor_compressor_state_dict, strict = strict, assign = assign)
