#quantizer parent class
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizerParent(nn.Module):

    def __init__(self, codes, codebook, reconstructed_shape):
        super(QuantizerParent, self).__init__()
        self.register_buffer('codes', codes)
        self.codebook = nn.Parameter(codebook)
        self.reconstructed_shape = reconstructed_shape 

    
    def forward(self):
        """forward should dequantize the codes and return the reconstructed tensor"""
        raise NotImplementedError
    
    def reconstruct(self):
        """alias for forward"""
        return self()
    
    @staticmethod
    def quantize(**kwargs):
        raise NotImplementedError
    
    def get_n_bits(self):
        raise NotImplementedError
    
    def get_n_original_parameters(self):
        return torch.prod(torch.tensor(self.codes.shape)).item()
