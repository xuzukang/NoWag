#parent class for all compression algorithms
import torch
import torch.nn as nn


class CompressorParent(nn.Module):
    """Parent class of all compression algorithms"""
    def __init__(self):
        super(CompressorParent, self).__init__()

    def reconstruct(self):
        raise NotImplementedError
    
    def update_discrete(self):
        """if our compression algorithm has discrete values, we need to update them"""
        pass 
    
    