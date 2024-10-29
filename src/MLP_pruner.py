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
#raise
def hessian_importances(H_2:torch.Tensor, W_2:torch.Tensor,
                        damping:float = 1e-6,
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
    assert W_2.shape[1] == H_2.shape[0], "The number of input channels of the weights and the Hessian must be the same"
    assert H_2.shape[1] == H_2.shape[0], "The Hessian must be square"
    #compute the inverse of the Hessian
    # torch.save({"H_2":H_2}, "H_2.pt")
    mean = torch.mean(torch.diag(H_2))
    indexs = torch.arange(H_2.shape[0], device = H_2.device)
    d = damping
    while True:
        H_ = H_2.clone()
        print(d * mean)
        H_[indexs, indexs] += d * mean
        H_chol = torch.linalg.cholesky(H_.float())
        if torch.any(torch.isnan(H_chol)):
            d *= 2
            print(f"Damping factor too small, increasing to {d}")
            torch.save({"H_2":H_2, "d":d}, "error.pt")
            raise ValueError("Damping factor too small")
        else:
            break
    H_inv = torch.cholesky_inverse(H_chol)
    #get the diagonal of the Hessian
    diag_H = torch.diag(H_inv) # (N_in)
    assert torch.all(torch.isfinite(diag_H)), f"Hessian is not invertible, check the damping factor, {diag_H}"

    #computer the importance of the channels
    channel_importance = torch.sum(W_2**2, dim = 0) / diag_H # (N_in)
    return channel_importance

def create_mask(channel_importance, n_in:int, keep_top:float = 0.125, keep_bottom:float = 0.0, d:int = 1):

    #sort the channels by importance
    _, sorted_indices = torch.sort(channel_importance, descending = True)
    #shape (N_in)

    #compute the number of channels to keep
    n_to_keep_top = int(keep_top * n_in)
    n_to_keep_bottom = int(keep_bottom * n_in)

    #we want the sum of n_to_keep_top and n_to_keep_bottom to be divisible by d
    n_to_keep_top += int(d - (n_to_keep_top + n_to_keep_bottom) % d)
    print(f"keeping {n_to_keep_top} top channels and {n_to_keep_bottom} bottom channels")

    mask = torch.zeros(n_in, device = channel_importance.device, dtype = torch.bool)
    mask[sorted_indices[:n_to_keep_top]] = True
    if n_to_keep_bottom > 0:
        mask[sorted_indices[-n_to_keep_bottom:]] = True

    return mask

def gradient_importances(grad:torch.Tensor, W_2:torch.Tensor):
    
    #grad is the same shape as W_2
    importances = torch.mean(torch.abs(grad*W_2), dim = 0)
    return importances
    


class pruned_feed_forward(nn.Module):
    def __init__(self, w1:torch.Tensor, w2:torch.Tensor, w3:torch.Tensor, activation = nn.SiLU,initial_grad:bool = False):
        super().__init__()
        self.initial_grad = initial_grad  
        if initial_grad:
            self.w1 = nn.Parameter(w1.clone().requires_grad_())
            self.w2 = nn.Parameter(w2.clone().requires_grad_())
            self.w3 = nn.Parameter(w3.clone().requires_grad_())
        else:
            self.w1 = w1.clone().requires_grad_(False)
            self.w2 = w2.clone().requires_grad_(False)
            self.w3 = w3.clone().requires_grad_(False)
        self.activation = activation()
        self.n_original_params = self.w1.numel() + self.w2.numel() + self.w3.numel()
        self.inp_size = self.w1.shape[1]
        self.hidden_size = self.w1.shape[0]
        self.out_size = self.w2.shape[0]
        self.quantized = False
        self.pruned = False
        self.add_batch_ = False

        self.b1 = None
        self.b3 = None
        self.b2 = None

    def forward(self, x:torch.Tensor):
        if self.quantized:
            return F.linear((
                self.activation(F.linear(x, self.w1(), bias = self.b1))
                                * F.linear(x, self.w3(), bias = self.b3)
            ), self.w2(), bias = self.b2)
        elif self.pruned:
            
            hidden = self.activation(F.linear(x, self.w1, self.b1)) * F.linear(x, self.w3, self.b3)
            self.add_batch(x, hidden)
            return F.linear(hidden, self.w2, bias = self.b2)

        else:
            
            hidden = self.activation(F.linear(x, self.w1)) * F.linear(x, self.w3)
            # self.add_batch(x, hidden)
            self.add_importances(x, hidden)
            return F.linear(hidden, self.w2)
            
            
    
    def add_batch(self, inp:torch.Tensor, hidden:torch.Tensor):
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
            self.H_in = torch.zeros((self.inp_size, self.inp_size), device = self.w1.device,
                                    dtype = torch.float32)
            self.H_hidden = torch.zeros((self.hidden_size,self.hidden_size), device = self.w2.device,
                                    dtype = torch.float32)
            self.nsamples = 0


        self.H_in *= self.nsamples / (self.nsamples + tmp)
        self.H_hidden *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        hidden = math.sqrt(2 / self.nsamples) * hidden
        tmp = torch.clip(hidden.matmul(hidden.t()),-1e5,1e5)
        
        #assert that tmp is positive definite
        # cholenksy = torch.linalg.cholesky(tmp + 1e-6 * torch.eye(tmp.shape[0], device = tmp.device))
        # if not torch.all(torch.isfinite(cholenksy)):
        #     torch.save({"tmp":tmp, "hidden":hidden, "inp":inp, "H_hidden":self.H_hidden}, "error.pt")
        #     raise ValueError("The matrix is not positive definite")
        self.H_in += torch.clip(inp.matmul(inp.t()), -1e5, 1e5)
        self.H_hidden += tmp

    def add_importances(self, inps, hidden):

        if not self.add_batch_:
            return
        
        if not hasattr(self, 'importances'):
            self.importances = torch.zeros(self.hidden_size, device = self.w2.device, dtype = torch.float32)
            self.nsamples = 0
            

        hidden = hidden.reshape(-1, hidden.shape[-1])

        self.importances *= self.nsamples / (self.nsamples + hidden.shape[0])

        hidden *= math.sqrt(2 / hidden.shape[0])
        self.importances += torch.einsum("ij,kj->j", hidden, self.w2)**2
        self.nsamples += hidden.shape[0]



    def prune(self, keep_top:float = 0.125, keep_bottom:float = 0.0,damping:float = 1e-6,
                add_bias:bool = False, d:int = 1, random_mask:bool = False):
        if random_mask:
            print("using a random mask")
            mask = torch.zeros(self.w2.shape[1], device = self.w2.device, dtype = torch.bool)
            mask[:int((keep_top + keep_bottom) * self.w2.shape[1])] = True
            mask = mask[torch.randperm(mask.shape[0])]
        else:
            if hasattr(self, 'importances'):
                print("using importances")
                importances = self.importances.clone()
                del self.importances
            elif hasattr(self, 'H_in'):
                importances = hessian_importances(torch.clip(self.H_hidden,-1e5,1e5)
                                    , self.w2, damping = damping)
            #otherwise prune based on gradient
            else:
                print("pruning based on gradient")
                importances = gradient_importances(self.w2.grad, self.w2)
            mask = create_mask(importances, self.w2.shape[1], keep_top = keep_top, keep_bottom = keep_bottom, d = d)
            
        
        # print(f"of the {self.w2.shape[1]} channels, {mask.sum()} are kept and {mask.shape[0] - mask.sum()} are pruned")
        # print("old shapes: ", self.w1.shape, self.w2.shape, self.w3.shape)
        self.w1 = nn.Parameter(self.w1[mask], requires_grad = True)
        self.w3 = nn.Parameter(self.w3[mask], requires_grad = True)
        self.w2 = nn.Parameter(self.w2[:, mask], requires_grad = True)
        self.hidden_size = mask.sum().item()
        # print("new shapes: ", self.w1.shape, self.w2.shape, self.w3.shape)

        if add_bias:
            self.b1 = nn.Parameter(torch.zeros_like(self.w1[:,0]), requires_grad = True)
            self.b3 = nn.Parameter(torch.zeros_like(self.w3[:,0]), requires_grad = True)
            self.b2 = nn.Parameter(torch.zeros_like(self.w2[:,0]), requires_grad = True)
            # print("self.b1 shape: ", self.b1.shape, "self.b2 shape: ", self.b2.shape, "self.b3 shape: ", self.b3.shape)  

        self.pruned = True
        if hasattr(self, 'H_in'):
            del self.H_in
            del self.H_hidden
            del self.nsamples
        self.add_batch_ = False

    def quantize(self, **kwargs):
        w1_values = self.w1.detach().clone().float()
        w2_values = self.w2.detach().clone().float()
        w3_values = self.w3.detach().clone().float()

        if self.w1.grad is not None:
            w1_grad = self.w1.grad.detach().clone().float()
            w2_grad = self.w2.grad.detach().clone().float() 
            w3_grad = self.w3.grad.detach().clone().float()

        del self.w1
        del self.w2
        del self.w3

        if hasattr(self, 'H_in'):
            self.w1 = quantizer.Quantize(w1_values, self.H_in, **kwargs)
            self.w2 = quantizer.Quantize(w2_values, self.H_hidden, **kwargs)
            self.w3 = quantizer.Quantize(w3_values, self.H_in, **kwargs)
            
            del self.H_in
            del self.H_hidden
        else:
            raise ValueError("Hessian must be computed before quantizing")

        if kwargs.get('add_bias', False):
            if not hasattr(self, 'b1'):
                print("adding bias because bias was not added during pruning")
                self.b1 = nn.Parameter(torch.zeros_like(self.w1[:,0]), requires_grad = True)
                self.b3 = nn.Parameter(torch.zeros_like(self.w3[:,0]), requires_grad = True)
                self.b2 = nn.Parameter(torch.zeros_like(self.w2[:,0]), requires_grad = True)

            
        self.add_batch_ = False
        self.quantized = True
        
    def turn_on_batch_add(self):
        self.add_batch_ = True
        
    def get_n_bits(self):
        
        sum_bits = 0
        # print("w1 size: ", self.w1.shape, "w2 size: ", self.w2.shape, "w3 size: ", self.w3.shape)
        if self.quantized:
            sum_bits += self.w1.get_n_bits() + self.w2.get_n_bits() + self.w3.get_n_bits()
        
        elif self.pruned:
            sum_bits += 16 * self.w1.numel() + 16 * self.w2.numel() + 16 * self.w3.numel()
        else:
            sum_bits += 16 * self.w1.numel() + 16 * self.w2.numel() + 16 * self.w3.numel()
        if self.b1 is not None:
            sum_bits += 16 * self.b1.numel() + 16 * self.b2.numel() + 16 * self.b3.numel()
            
        
        return sum_bits, self.n_original_params

        
        




