import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLayer(nn.Module):

    def __init__(self, linear_layer:nn.Linear, 
                 k:int = 1):
        
        super(LoraLayer, self).__init__()

        self.frozen_weight = nn.Parameter(linear_layer.weight.clone().detach(), requires_grad = False)
        if linear_layer.bias is not None:
            self.frozen_bias = nn.Parameter(linear_layer.bias.clone().detach(), requires_grad = False)
        else:
            self.frozen_bias = None

        n_out,n_in = self.frozen_weight.shape

        self.A = nn.Parameter(torch.randn(n_in, k, device = self.frozen_weight.device), requires_grad = True)
        self.B = nn.Parameter(torch.zeros(k, n_out, device = self.frozen_weight.device), requires_grad = True)

    
    def forward(self, x:torch.Tensor):

        return F.linear(x, self.frozen_weight, self.frozen_bias) + F.linear(F.linear(x, self.A), self.B)
    
    def reconstruct(self):
        
        return self.frozen_weight + (self.A @ self.B).T
    
    
    
