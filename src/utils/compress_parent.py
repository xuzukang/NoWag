# parent class for all compression algorithms
import torch
import torch.nn as nn
from typing import List, Dict


class CompressorParent(nn.Module):
    """Parent class of all compression algorithms"""

    def __init__(self, additional_parameters: Dict[str, torch.FloatTensor] = {}):
        super(CompressorParent, self).__init__()
        self.additional_active_attribute_names = []
        for name, param in additional_parameters.items():
            if param is not None:
                setattr(self, name, nn.Parameter(param, requires_grad=False))
                self.additional_active_attribute_names.append(name)
            else:
                setattr(self, name, None)
    def compress(self, **kwargs):
        raise NotImplementedError
    def reconstruct(self):
        raise NotImplementedError

    def add_additional_attributes(
        self, additional_parameters: Dict[str, torch.FloatTensor]
    ):
        """add additional attributes"""
        for name, param in additional_parameters.items():
            if param is not None:
                setattr(self, name, nn.Parameter(param, requires_grad=False))
                self.additional_active_attribute_names.append(name)
            else:
                setattr(self, name, None)

    def set_additional_attributes_as_trainable(self):
        """set additional attributes as trainable"""
        for name in self.additional_active_attribute_names:
            getattr(self, name).requires_grad = True

    def set_additional_attributes_as_non_trainable(self):
        """set additional attributes as non trainable"""
        for name in self.additional_active_attribute_names:
            getattr(self, name).requires_grad = False

    def get_additional_attributes(self) -> Dict[str, torch.FloatTensor]:
        """get the additional attributes"""
        additional_parameters = {}
        for name in self.additional_active_attribute_names:
            additional_parameters[name] = getattr(self, name)
        return additional_parameters

    def update_discrete(self):
        """if our compression algorithm has discrete values, we need to update them"""
        pass

    def clean(self):
        """clean the additional attributes"""
        pass

    def get_n_bits(self):
        """get the number of bits"""
        raise NotImplementedError

    def get_n_original_parameters(self):
        """get the number of original parameters"""
        raise NotImplementedError
        # return torch.prod(torch.tensor(self.reconstructed_shape)).item()
