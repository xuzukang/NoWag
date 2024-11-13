#parent class for all compression algorithms
import torch
import torch.nn as nn


class CompressorParent(nn.Module):

    def __init__(self):
        super(CompressorParent, self).__init__()

    def reconstruct(self):
        raise NotImplementedError
    
    def update_discrete(self):
        raise NotImplementedError