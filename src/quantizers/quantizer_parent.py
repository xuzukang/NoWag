# quantizer parent class
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.utils.compress_parent as compress_parent
from typing import Union, Tuple, Optional, List


class QuantizerParent(nn.Module):
    def __init__(
        self,
        codes: torch.LongTensor,
        codebook: torch.FloatTensor,
        reconstructed_shape: Union[Tuple[int, int], torch.Size],
        reference_weight: Optional[torch.FloatTensor],
        additional_attributes: Optional[dict] = None,
    ):
        super(QuantizerParent, self).__init__()
        self.register_buffer("codes", codes)
        self.codebook = nn.Parameter(codebook)
        self.reconstructed_shape = reconstructed_shape
        # print("reference_weight", reference_weight)
        self.register_buffer("reference_weight", reference_weight)
        self.n_out, self.n_in = self.reconstructed_shape

    def forward(self):
        """forward should dequantize the codes and return the reconstructed tensor"""
        raise NotImplementedError

    def reconstruct(self):
        """alias for forward"""
        return self()

    @staticmethod
    def quantize(weight: torch.FloatTensor, hessian: torch.FloatTensor, **kwargs):
        raise NotImplementedError

    def get_n_bits(self):
        n_bits = 0
        # sum the bits of the codebook
        n_bits += self.codebook.numel() * 16
        # sum the bits of the codes
        n_bits += (
            self.codes.numel() * torch.log2(torch.tensor(self.codebook.shape[0])).item()
        )
        return n_bits

    def get_n_original_parameters(self):
        return torch.prod(torch.tensor(self.reconstructed_shape)).item()

    def clean(self):
        # print("here")
        if hasattr(self, "reference_weight"):
            delattr(self, "reference_weight")

    @staticmethod
    def blank_recreate(**kwargs):
        """initializes a blank quantizer with the same shape to be filled by
        a state_dict"""
        raise NotImplementedError
