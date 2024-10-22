import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import src.quantizer as quantizer


def create_mask_helper(data:list,percent_top):
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
    while mask.sum() > (100-percent_top)/100 * mask.numel():
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
                sparse_percent:float = 0,
                sparse_criterion:list[str] = ["weight","hessian"],
                dim:int = 0):
    
    tmp = []

    for criterion in sparse_criterion:
        if criterion == "weight":
            tmp.append(torch.norm(weights, dim = dim))
        elif criterion == "hessian":
            tmp.append(torch.norm(H, dim = dim))
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        
    mask = create_mask_helper(tmp, sparse_percent)
    return mask

def construct_weights(module,weights,mask, row_norms, column_norms,
                      module_kwargs = {}):

    weights_reconstructed = torch.zeros_like(weights)
    weights_reconstructed[mask] = (module(**module_kwargs) * row_norms.unsqueeze(0) * column_norms.unsqueeze(1)
                                   ).flatten()
    weights_reconstructed[~mask] = weights[~mask]
    return weights_reconstructed



class Low_Rank_linear(nn.Module):
    
    def __init__(self, W:torch.Tensor):
        super().__init__()
        
        #assert that W is a square matrix 
        assert W.shape[0] == W.shape[1]
        self.W = W.clone().requires_grad_(False)
        self.n_original_params = W.numel()
        self.low_ranked = False
        self.quantized = False
        self.add_batch_ = False
        
    def forward(self, x:torch.Tensor):
        if self.quantized:
            y = torch.zeros_like(x)
            hidden = F.linear(x[..., self.column_mask] * self.weights_norms_rowwise.unsqueeze(0), self.B())
            self.add_batch(x, hidden)
            y[...,self.row_mask] = torch.nn.functional.linear(hidden, self.A())
            y[...,self.row_mask] += torch.nn.functional.linear(x[..., ~self.column_mask], self.sparse_weights1)
            y[...,~self.row_mask] = torch.nn.functional.linear(x, self.sparse_weights2)
            return y 
        elif self.low_ranked:
            
            y = torch.zeros_like(x)
            hidden = F.linear(x[..., self.column_mask] * self.weights_norms_rowwise.unsqueeze(0), self.B)
            self.add_batch(x, hidden)
            y[...,self.row_mask] = torch.nn.functional.linear(hidden, self.A)
            y[...,self.row_mask] += torch.nn.functional.linear(x[..., ~self.column_mask], self.sparse_weights1)
            y[...,~self.row_mask] = torch.nn.functional.linear(x, self.sparse_weights2)
            return y    
            # hidden = F.linear(x, self.A)
            # self.add_batch(x, hidden)
            # return F.linear(hidden, self.B) * self.weights_norms_rowwise.unsqueeze(0)
        else:
            self.add_batch(x)
            return F.linear(x, self.W)
        
    def add_batch(self, inp, hidden = None):
        if not self.add_batch_:
            return
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            if hidden is not None:
                hidden = hidden.unsqueeze(0)
        tmp = inp.shape[0]
 
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            if hidden is not None:
                hidden = hidden.reshape((-1, hidden.shape[-1]))
        inp = inp.t()
        hidden = hidden.t()

        if not hasattr(self, 'H_in'):
            in_features = inp.shape[0]
            self.H_in = torch.zeros((in_features, in_features), device = self.inp.device,
                                    dtype = self.inp.dtype)
            if hidden is not None:
                hidden_in_features = hidden.shape[0]
                self.H_hidden = torch.zeros((hidden_in_features, hidden_in_features), device = self.hidden.device,
                                        dtype = self.hidden.dtype)
            self.nsamples = 0


        self.H_in *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H_in += inp.matmul(inp.t())
        
        if hidden is not None:
            self.H_hidden *= self.nsamples / (self.nsamples + tmp)
            hidden = math.sqrt(2 / self.nsamples) * hidden.float()
            self.H_hidden += hidden.matmul(hidden.t())
            
    
    def low_rank(self, 
                low_rank:int = 196,
                sparse_rowise:float = 0,
                sparse_rowsie_criterion:list[str] = ["weight"],
                sparse_colwise:float = 0,
                sparse_colwise_criterion:list[str] = ["weight","hessian"]):
        
    
        print("using low rank = ", low_rank)
        
        original_dtype = self.W.dtype
        W = self.W.clone().float()
        H = self.H_in.clone().float()
        
        if sparse_colwise > 0:
            self.register_buffer("column_mask", create_mask(W, H, sparse_colwise, sparse_colwise_criterion, dim = 0))
        else:
             self.register_buffer("column_mask", torch.ones(self.W.shape[1]
                                    , dtype = torch.bool, device = self.W.device))
        
        if sparse_rowise > 0:
            self.register_buffer("row_mask", create_mask(W, H, sparse_rowise, sparse_rowsie_criterion, dim = 1))
        else:
            self.register_buffer("row_mask", torch.ones(self.W.shape[0]
                                , dtype = torch.bool, device = self.W.device))

        print("row mask = ", self.row_mask.sum().item(), "column mask = ", self.column_mask.sum().item())
        # mask = self.row_mask.unsqueeze(1) & self.column_mask.unsqueeze(0)

        print("self.row_mask.shape = ", self.row_mask.shape, "self.column_mask.shape = ", self.column_mask.shape)
        weights_adjusted = W[self.row_mask,:][:, self.column_mask]
        weights_norms_rowwise = torch.norm(weights_adjusted, dim = 0)
        weights_norms_rowwise[torch.isclose(weights_norms_rowwise, torch.zeros_like(weights_norms_rowwise))] = 1
        weights_normalized = weights_adjusted / self.weights_norms_rowwise.unsqueeze(0)
        
        
        S, U, V = torch.svd(weights_normalized)
        self.A = nn.Parameter((U[:, :low_rank] @ torch.sqrt(torch.diag(S[:low_rank]))).to(original_dtype))
        self.B = nn.Parameter((torch.sqrt(torch.diag(S[:low_rank])) @ V[:, :low_rank].T).to(original_dtype))
        self.weights_norms_rowwise = nn.Parameter((weights_norms_rowwise).to(original_dtype))
        self.sparse_weights1 = nn.Parameter((self.W[~self.row_mask,:][:, self.column_mask].clone()).to(original_dtype))
        self.sparse_weights2 = nn.Parameter(self.W[~self.row_mask].to(original_dtype))
        del self.W
        del W
        del H
        
        self.low_ranked = True
            
    def quantize(self, **kwargs):
        self.A = quantizer.Quantize(self.A.detach().float(), self.H_hidden.float(), **kwargs).to(self.A.device)
        self.B = quantizer.Quantize(self.B.detach().float(), self.H_in.float(), **kwargs).to(self.B.device)
        self.quantized = True
        del self.H_in
        del self.H_hidden
        del self.nsamples
        self.add_batch_ = False
        
    def turn_on_batch_add(self):
        self.add_batch_ = True
        
    def get_n_bits(self):
        
        sum_bits = 8*(torch.sum(self.row_mask)+torch.sum(self.column_mask)) + 16 * self.weights_norms_rowwise.numel()
        
        if self.quantized:
            sum_bits += self.A.get_n_bits() + self.B.get_n_bits()
        
        elif self.low_ranked:
            sum_bits += 16 * self.A.numel() + 16 * self.B.numel()
        
        return sum_bits, self.n_original_params
        
    