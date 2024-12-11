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
        
        
    def compress(self, quantization_compression_algorithm, 
                 quantization_kwargs,
                 quantization_align_kwargs,
                 tensor_compression_algorithm,
                    tensor_compression_kwargs,
                    tensor_compression_align_kwargs,
                    n_iters: int = 1):
        """compresses the weight matrix"""
        W_remaining = self.original_weight.clone()
        
        best_loss = float("inf")
        for iter in range(n_iters):
            self.quantization_compressor:LinearQuantized = quantization_compression_algorithm(W_remaining)
            self.quantization_compressor.hessian = self.hessian
            self.quantization_compressor.quantize(**quantization_kwargs)
            self.quantization_compressor.set_additional_attributes_as_trainable()
            self.quantization_compressor.align(**quantization_align_kwargs)
            with torch.no_grad():
                W_remaining = self.original_weight.clone() - self.quantization_compressor.reconstruct()
            
            self.tensor_compressor:LinearTensorized = tensor_compression_algorithm(W_remaining)
            self.tensor_compressor.hessian = self.hessian
            self.tensor_compressor.tensor_decompose(**tensor_compression_kwargs)
            self.tensor_compressor.set_additional_attributes_as_trainable()
            print(tensor_compression_align_kwargs)
            loss = self.tensor_compressor.align(**tensor_compression_align_kwargs)
            with torch.no_grad():
                W_remaining = self.original_weight.clone() - self.tensor_compressor.reconstruct()
            if loss < best_loss:
                best_loss = loss    
            
        self.compressed = True
        return best_loss
    
    def reconstruct(self):
        """reconstructs the original weight matrix"""
        if not self.compressed:
            return self.original_weight
        else:
            return self.quantization_compressor.reconstruct() + self.tensor_compressor.reconstruct()
        
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
        
    
    def get_n_bits(self):
        return self.quantization_compressor.get_n_bits() + self.tensor_compressor.get_n_bits()
