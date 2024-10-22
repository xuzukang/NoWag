import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import src.quantizer as quantizer
#prune the Feedforward network
#normally for llama and mistral the MLP is defined as:
# w2(sigma(w1x) (.) w3 x)
# where w1, w2, w3 are the weights of the layers
# sigma is the activation function
# (.) is the elementwise product
# so when we prune, we want to prune a whole channel of w1, w2, w3

def select_channel(H_2:torch.Tensor, W_2:torch.Tensor,
                   keep_top:float = 0.125, keep_bottom:float = 0.0,
                   debug:bool = False, damping:float = 1e-6
                   ) -> torch.Tensor:
    
    """
    Select the channel to keep in the weights
    
    H_2: torch.Tensor the Hessian of the calibration dataset (N_in, N_in) 
    W_2: torch.Tensor the weights of the layer (N_out, N_in)
    keep_top: float the percentage of the top channels to keep, default 0.125 = 1/8
    keep_bottom: float the percentage of the bottom channels to keep, default 0
    debug: bool print debug information, default False
    damping: float the damping factor to add to the Hessian, default 1e-6
    
    out:
    mask: torch.Tensor the mask of the channels to keep of shape (N_in)
    """

    #compute the inverse of the Hessian
    H_chol = torch.cholesky(H_2 + damping * torch.mean(torch.diag(H_2))* torch.eye(H_2.shape[0], device = H_2.device))
    H_inv = torch.cholesky_inverse(H_chol)

    #get the diagonal of the Hessian
    diag_H = torch.diag(H_inv) # (N_in)

    #computer the importance of the channels
    channel_importance = torch.sum(W_2**2, dim = 1) / diag_H # (N_out)

    #sort the channels by importance
    _, sorted_indices = torch.sort(channel_importance, descending = True)
    #shape (N_out)

    #compute the number of channels to keep
    n_to_keep_top = int(keep_top * W_2.shape[1])
    n_to_keep_bottom = int(keep_bottom * W_2.shape[1])

    mask = torch.zeros(W_2.shape[1], device = W_2.device, dtype = torch.bool)
    mask[sorted_indices[:n_to_keep_top]] = True
    mask[sorted_indices[-n_to_keep_bottom:]] = True

    return mask


class pruned_feed_forward(nn.Module):
    def __init__(self, w1:torch.Tensor, w2:torch.Tensor, w3:torch.Tensor, activation = nn.SiLU):
        super().__init__()
        self.w1 = w1.clone().requires_grad_(False)
        self.w2 = w2.clone().requires_grad_(False)
        self.w3 = w3.clone().requires_grad_(False)
        self.activation = activation()
        self.n_original_params = self.w1.numel() + self.w2.numel() + self.w3.numel()

        self.quantized = False
        self.pruned = False
        self.add_batch_ = False

    def forward(self, x:torch.Tensor):
        if self.quantized:
            return F.linear((
                self.activation(F.linear(x, self.w1()), bias = self.b1)
                                * F.linear(x, self.w3(), bias = self.b3)
            ), self.w2(), bias = self.b2)
        elif self.pruned:
            
            hidden = self.activation(F.linear(x, self.w1, self.b1)) * F.linear(x, self.w3, self.b3)
            self.add_batch(x, hidden)
            return F.linear(hidden, self.w2, bias = self.b2)

        else:
            
            hidden = self.activation(F.linear(x, self.w1)) * F.linear(x, self.w3)
            self.add_batch(x, hidden)
            return F.linear(hidden, self.w2)
            
            
    
    def add_batch(self, inp, hidden):
        if not self.add_batch_:
            return
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            hidden = hidden.unsqueeze(0)
        tmp = inp.shape[0]
 
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            hidden = hidden.reshape((-1, hidden.shape[-1]))
        inp = inp.t()
        hidden = hidden.t()

        if not hasattr(self, 'H_in'):
            self.H_in = torch.zeros((self.w1.in_features, self.w1.in_features), device = self.w1.device,
                                    dtype = self.w1.dtype)
            self.H_hidden = torch.zeros((self.w2.in_features, self.w2.in_features), device = self.w2.device,
                                    dtype = self.w2.dtype)
            self.nsamples = 0


        self.H_in *= self.nsamples / (self.nsamples + tmp)
        self.H_hidden *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        hidden = math.sqrt(2 / self.nsamples) * hidden.float()
        self.H_in += inp.matmul(inp.t())
        self.H_hidden += hidden.matmul(hidden.t())

    def prune(self, keep_top:float = 0.125, keep_bottom:float = 0.0,damping:float = 1e-6,
                add_bias:bool = False): 
        mask = select_channel(self.H_hidden, self.w, keep_top = keep_top, keep_bottom = keep_bottom,
                                damping = damping)
        self.w1 = nn.Parameter(self.w1[mask], requires_grad = True)
        self.w3 = nn.Parameter(self.w3[mask], requires_grad = True)
        self.w2 = nn.Parameter(self.w2[:, mask], requires_grad = True)

        if add_bias:
            self.b1 = nn.Parameter(torch.zeros_like(self.w1[0]), requires_grad = True)
            self.b3 = nn.Parameter(torch.zeros_like(self.w3[0]), requires_grad = True)
            self.b2 = nn.Parameter(torch.zeros_like(self.w2[0]), requires_grad = True)  
        else:
            self.b1 = None
            self.b3 = None
            self.b2 = None

        self.pruned = True
        del self.H_in
        del self.H_hidden
        del self.nsamples
        self.add_batch_ = False

    def quantize(self, **kwargs):
        
        self.w1 = quantizer.Quantize(self.w1.detach(), self.H_in, **kwargs)
        self.w2 = quantizer.Quantize(self.w2.detach(), self.H_hidden, **kwargs)
        self.w3 = quantizer.Quantize(self.w3.detach(), self.H_in, **kwargs)
        
        del self.H_in
        del self.H_hidden
        self.add_batch_ = False
        
    def turn_on_batch_add(self):
        self.add_batch_ = True
        
    def get_n_bits(self):
        
        sum_bits = 0
        
        if self.quantized:
            sum_bits += self.w1.get_n_bits() + self.w2.get_n_bits() + self.w3.get_n_bits()
        
        elif self.pruned:
            sum_bits += 16 * self.w1.numel() + 16 * self.w2.numel() + 16 * self.w3.numel()
            
        if hasattr(self, 'b1'):
            sum_bits += 16 * self.b1.numel() + 16 * self.b2.numel() + 16 * self.b3.numel()
            
        
        return sum_bits, self.n_original_params

        
        




