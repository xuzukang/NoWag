import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import src.quantizer as quantizer
import numpy as np
import os
import src.utils.alignment.low_rank_align as low_rank_align


def create_mask_helper(data:list,frac_top,d  =1):
    """
    data: list of torch.tensor of shape (n)
    percent_top: float, the percentage of the top values to keep
    """
    mask = torch.ones(data[0].shape, dtype = torch.bool, device = data[0].device)

    datas_sorted = [torch.sort(data[i], descending = True)[0] for i in range(len(data))]
    i = 0
    big_i = 0
    # print(mask.sum(), (100-percent_top)/100 * mask.numel())
    # raise ValueError("The mask is not sparse enough")
    while (mask.sum() > (1-frac_top) * mask.numel()) or (mask.sum()%d != 0):
        # print(mask.sum())
        mask &= data[i%len(data)] < datas_sorted[i][big_i]
        i += 1
        i = i % len(data)
        if i == 0:
            big_i += 1
        # if i==0:
        #     raise ValueError("The mask is not sparse enough")
    # threshold = torch.quantile(data, 1-percent_top/100)
    return mask


def create_mask(weights:torch.Tensor,
                H:torch.Tensor,
                sparse_frac:float = 0,
                sparse_criterion:list[str] = ["weight","hessian"],
                dim:int = 0,
                d:int = 1):
    
    tmp = []

    for criterion in sparse_criterion:
        if criterion == "weight":
            tmp.append(torch.norm(weights, dim = dim))
        elif criterion == "hessian":
            if H is None:
                print("Hessian is None, skipping")
                continue
            tmp.append(torch.norm(H, dim = dim))
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
    assert len(tmp) > 0, "tmp is empty"
    print(tmp)
    print("sparse_frac = ", sparse_frac)
    mask = create_mask_helper(tmp, sparse_frac, d)
    return mask

def construct_weights(module,weights,mask, row_norms, column_norms,
                      module_kwargs = {}):

    weights_reconstructed = torch.zeros_like(weights)
    weights_reconstructed[mask] = (module(**module_kwargs) * row_norms.unsqueeze(0) * column_norms.unsqueeze(1)
                                   ).flatten()
    weights_reconstructed[~mask] = weights[~mask]
    return weights_reconstructed


def reconstruct_fn(A:torch.Tensor,
                            B:torch.Tensor,
                            sparse_1:torch.Tensor,
                            sparse_2:torch.Tensor,
                            weights_norms_rowwise:torch.Tensor,
                            row_mask:torch.Tensor,
                            column_mask:torch.Tensor,
                            n_in:int,
                            n_out:int):
    weights_reconstructed = torch.zeros((n_out, n_in), device = A.device, dtype = A.dtype)
    # print(torch.sum(row_mask), torch.sum(column_mask))
    # print((A @ B) * weights_norms_rowwise.unsqueeze(0))
    weights_reconstructed[row_mask.unsqueeze(1)*column_mask.unsqueeze(0)] = ((A @ B) * weights_norms_rowwise.unsqueeze(0)).flatten()
    # print("reconstructed weights = ", weights_reconstructed)
    weights_reconstructed[row_mask.unsqueeze(1) * (~column_mask).unsqueeze(0)] = sparse_1.flatten() 
    weights_reconstructed[~row_mask] = sparse_2
    # print("reconstructed weights = ", weights_reconstructed)
    return weights_reconstructed
        
        

class compressLinear(nn.Module):
    
    def __init__(self, W:torch.Tensor):
        super().__init__()
        
        #assert that W is a square matrix 
        # assert W.shape[0] == W.shape[1]
        self.W = W.clone().requires_grad_(False)
        self.n_original_params = W.numel()
        self.n_out, self.n_in = W.shape
        
        self.low_ranked = False
        self.quantized = False
        self.add_batch_ = False
        self.use_precomputed = False
        
    def forward(self, x:torch.Tensor):
        if self.low_ranked:
            return self.A(self.B(x))
        elif self.quantized:
            self.compute_quantized()
            return F.linear(x, self.W)
        else:
            self.add_batch(x)
            return F.linear(x, self.W)
        
    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        
        if self.quantized:
            if not self.low_rank:
                self.use_precomputed = True
                self.register_buffer("precomputed", self.W())
        

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True
        
        if self.quantized and not self.low_rank:
            self.use_precomputed = False
            del self.precomputed  

    def compute_quantized(self):
        if not self.quantized:
            raise ValueError("The weights are not quantized")
        if self.use_precomputed:
            return self.precomputed
        else:
            return self.W()
        
    def add_batch(self, inp:torch.Tensor):
        if not self.add_batch_:
            return
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
 
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        if not hasattr(self, 'H_in'):
            in_features = inp.shape[0]
            self.H = torch.zeros((in_features, in_features), device = inp.device,
                                    dtype = torch.float32)
            self.nsamples = 0


        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.to(torch.float32)
        self.H += torch.clip(inp.matmul(inp.t())/self.H.shape[0], -1e5, 1e5)
        
            
    
    def low_rank(self, 
                low_rank:int = 196,
                regulatization_lambda:float = 1e-3,
                damping:float = 1e-3,
                align_epochs:int = 100,
                lr:float = 1e-3,
                lr_multiplier:float = 1,
                d:int = 1):
        
        low_rank_use = int(np.ceil(low_rank/d)*d)
        print("using low rank = ", low_rank_use)
        
        original_dtype = self.W.dtype
        W = self.W.clone().to(torch.float32)
        if hasattr(self, 'H'):
            H = self.H.clone().to(torch.float32)
        else:
            raise ValueError("The Hessian is not available")

        
        U, S, V = torch.svd(W)
        # print("S.shape = ", S.shape, "U.shape = ", U.shape, "V.shape = ", V.shape)
        # print("low_rank = ", low_rank)
        A = (U[:, :low_rank_use] @ torch.sqrt(torch.diag(S[:low_rank_use])))
        B = (torch.sqrt(torch.diag(S[:low_rank_use])) @ V[:, :low_rank_use].T)

        # @torch.enable_grad()
        
        #if the alignment epochs is greater than 0, align the low rank
        if align_epochs > 0:
            ids = torch.arange(W.shape[0], device = W.device)
            #first ensure that the diagonal of the hessian is positive
            H[ids, ids] = torch.clip(torch.diag(H), 1e-5)
            H[ids, ids] += damping * torch.mean(torch.diag(H))
            
            A, B = low_rank_align.align_simple(W,A,B,lr,H,
                                               regularization_lambda=regulatization_lambda,
                                               epochs = align_epochs,
                                                lr_multiplier=lr_multiplier, 
                                                verbose = align_epochs//10)
            
            

        # self.A = nn.Parameter(A.to(original_dtype))
        # self.B = nn.Parameter(B.to(original_dtype))

        self.A = compressLinear(A.to(original_dtype))
        self.B = compressLinear(B.to(original_dtype))
        # self.A.turn_on_batch_add()
        # self.B.turn_on_batch_add()

        del self.W
        del W
        del H
        if hasattr(self, 'H'):

            del self.H
        
        self.low_ranked = True
        self.add_batch_ = False
            
    def quantize(self, **kwargs):

        if self.low_ranked:
            kwargs_use_A = kwargs.copy()
            kwargs_use_A["debug_path"] = kwargs["debug_path"] + "_A"
            self.A.quantize(**kwargs_use_A)
            kwargs_use_B = kwargs.copy()
            kwargs_use_B["debug_path"] = kwargs["debug_path"] + "_B"
            self.B.quantize(**kwargs_use_B)

        else:
            W = self.W.clone()
            H = self.H.clone()
            if kwargs["debug"]:
                os.makedirs(os.path.dirname(kwargs["debug_path"]), exist_ok = True)
                torch.save({"W":W, "H":H}, kwargs["debug_path"] + ".pth")            
            del self.W
            self.W = quantizer.Quantize.quantize(W, H, **kwargs)

            del W
            del H
            self.quantized = True
            self.add_batch_ = False

            del self.H

    
        self.quantized = True
            
        self.add_batch_ = False
        
    def turn_on_batch_add(self):
        self.add_batch_ = True
        
    def turn_off_batch_add(self):
        self.add_batch_ = False
        

    def clear_batches(self):
        if hasattr(self, 'H'):

            del self.H
            del self.nsamples
            
    def __str__(self):  
        if self.low_ranked:
            return f"Low Rank Compressed Linear Layer with \n{self.A} and \n{self.B}"
        elif self.quantized:
            return f"Quantized Linear Layer with {self.n_in} inputs and {self.n_out} outputs"
        else:
            return f"Linear Layer with {self.n_in} inputs and {self.n_out} outputs"
        


        
    def get_n_bits(self):
        
        # sum_bits = 8*(torch.sum(self.row_mask)+torch.sum(self.column_mask)) + 16 * self.weights_norms_rowwise.numel() + 16 * self.sparse_weights1.numel() + 16 * self.sparse_weights2.numel()
        # print("A size: ", self.A.shape, "B size: ", self.B.shape)
        # print("row mask size: ", self.row_mask.shape, "column mask size: ", self.column_mask.shape)
        # print("weights_norms_rowwise size: ", self.weights_norms_rowwise.shape)
        # print("sparse_weights1 size: ", self.sparse_weights1.shape, "sparse_weights2 size: ", self.sparse_weights2.shape)
        # print("="*10)


        if self.low_ranked:
            A_bits, *_ = self.A.get_n_bits()
            B_bits, *_ = self.B.get_n_bits()
            sum_bits = A_bits + B_bits

        elif self.quantized:
            sum_bits = self.W.get_n_bits()
        
        else:
            sum_bits = 16 * self.W.numel()
        
        return sum_bits, self.n_original_params
        
    