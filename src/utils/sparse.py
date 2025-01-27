import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#a class of sparse stuff

class SparseParent(nn.Module):
    
    def __init__(self, n_out:int, n_in:int, device:torch.device):
        super(SparseParent, self).__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.device = device
    
    def reconstruct(self)->torch.Tensor:
        raise NotImplementedError

    def forward(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

    def get_n_bits(self):
        raise NotImplementedError
    
class Dim0_StructuredSparse(SparseParent):
    sparse_mask:torch.BoolTensor
    
    def __init__(self, n_out:int, n_in:int, frac_sparse:float, device:torch.device):
        
        super(Dim0_StructuredSparse, self).__init__(n_out, n_in, device)
        
        sparse_mask = torch.zeros(n_in, dtype=torch.bool, device=device)
        # print("n_in", n_in, "n_out", n_out)
        self.frac_sparse = frac_sparse
        self.n_sparse = int(frac_sparse*n_in)
        
        self.register_buffer('sparse_mask', sparse_mask)
        self.sparse_values = nn.Parameter(torch.zeros((self.n_out, self.n_sparse), device=device))
        
    def update_sparse_norm(self, remaining_error:torch.FloatTensor # remaining error of shape (n_out, n_in)
                           ):
        
        #calculate the norm of the remaining error
        sparse_norm = torch.norm(remaining_error, dim=0)
        
        #sort the norm
        idxs_sorted = torch.argsort(sparse_norm, descending=True)
        
        new_mask = torch.zeros_like(self.sparse_mask)
        
        new_mask[idxs_sorted[:self.n_sparse]] = True
        
        self.register_buffer('sparse_mask', new_mask)
        
        self.sparse_values.data = -remaining_error[:, new_mask]
        # print("sparse_values", self.sparse_values)
        
    def update_sparse_hessian_importance(self, remaining_error:torch.FloatTensor, # remaining error of shape (n_out, n_in)
                                         importance:torch.FloatTensor # importance of shape (n_in) diagonal of hessian
                    ):
        
        
        #sort the importance
        idxs_sorted = torch.argsort(importance, descending=True)
        new_mask = torch.zeros_like(self.sparse_mask)
        
        new_mask[idxs_sorted[:self.n_sparse]] = True
        
        self.register_buffer('sparse_mask', new_mask)
        self.sparse_values.data = -remaining_error[:,new_mask]
        
        # -remaining_error[:, idxs_sorted[:self.n_sparse]] #shape (n_out, n_sparse)

    def reconstruct(self):
        # print("sparse_idxs", torch.where(self.sparse_mask))
        reconstructed_large = torch.zeros(self.n_out, self.n_in, device=self.sparse_values.device,
                                          dtype=self.sparse_values.dtype)
        reconstructed_large[:,self.sparse_mask] = self.sparse_values
        return reconstructed_large
    
    def forward(self, x:torch.FloatTensor, #shape (...,n_in)
                y:Optional[torch.FloatTensor] = None
               ):
        return x[...,self.sparse_mask] @ self.sparse_values.T
    
    def get_n_bits(self):
        return self.sparse_values.numel()*16 + self.sparse_mask.numel()
    
    def update_fixed_mask(self, remaining_error:torch.FloatTensor):
        
        self.sparse_values.data = -remaining_error[:, self.sparse_mask]


class Dim1_StructuredSparse(SparseParent):
    sparse_mask:torch.BoolTensor
    def __init__(self, n_out:int, n_in:int, frac_sparse:float, device:torch.device):
        
        super(Dim1_StructuredSparse, self).__init__(n_out, n_in, device)
        
        sparse_mask = torch.zeros(n_out, dtype=torch.bool, device=device)
        # print("n_in", n_in, "n_out", n_out)
        self.frac_sparse = frac_sparse
        self.n_sparse = int(frac_sparse*n_out)
        
        self.register_buffer('sparse_mask', sparse_mask)
        self.sparse_values = nn.Parameter(torch.zeros((self.n_sparse, self.n_in), device=device))
        
    def update_sparse_norm(self, remaining_error:torch.FloatTensor # remaining error of shape (n_out, n_in)
                           ):
        
        #calculate the norm of the remaining error
        sparse_norm = torch.norm(remaining_error, dim=1)
        
        #sort the norm
        idxs_sorted = torch.argsort(sparse_norm, descending=True)
        
        new_mask = torch.zeros_like(self.sparse_mask)
        
        new_mask[idxs_sorted[:self.n_sparse]] = True
        
        self.register_buffer('sparse_mask', new_mask)
        
        self.sparse_values.data = -remaining_error[new_mask, :]
        print("sparse_values", self.sparse_values)

    def forward(self, x:torch.FloatTensor, #shape (...,n_in)
                y:Optional[torch.FloatTensor] = None
               ):
        
        if y is None:
            y = torch.empty(x.shape[:-1] + (self.n_out,), device = x.device, dtype = x.dtype)
        
        y[...,self.sparse_mask] = x @ self.sparse_values.T
        return y

    def reconstruct(self):
        # print("sparse_idxs", torch.where(self.sparse_mask))
        reconstructed_large = torch.zeros(self.n_out, self.n_in, device=self.sparse_values.device,
                                          dtype=self.sparse_values.dtype)
        reconstructed_large[self.sparse_mask,:] = self.sparse_values
        return reconstructed_large
    
    
    def get_n_bits(self):
        return self.sparse_values.numel()*16 + self.sparse_mask.numel()
    
    def update_fixed_mask(self, remaining_error:torch.FloatTensor):
        
        self.sparse_values.data = -remaining_error[self.sparse_mask]
    
                                         
        
    
class UnstructuredSparse(nn.Module):
    #completely unstructured sparse and a parent class
    sparse_mask:torch.BoolTensor
    compressed:bool
    def __init__(self,
                 n_out:int,
                 n_in:int,
                 frac_sparse:float,
                 device:torch.device):
        
        super(UnstructuredSparse, self).__init__()
        
        #create a placeholder sparse mask
        sparse_mask = torch.zeros(n_out, n_in, dtype=torch.bool, device=device)
        
        self.n_out = n_out
        self.n_in = n_in
        self.frac_sparse = frac_sparse
        self.n_sparse = int(frac_sparse*n_out*n_in)
        
        
        self.register_buffer('sparse_mask', sparse_mask)
        self.sparse_values = nn.Parameter(torch.zeros(self.n_sparse, device=device))
        # self.compressed = False #convert to csr tensor for 
        
    @staticmethod
    def generate_mask(importances:torch.FloatTensor,
                      n_sparse:int):
        idxs_sorted = torch.argsort(importances.flatten(), descending=True)
        idx0, idx1 = idxs_sorted//importances.shape[1], idxs_sorted%importances.shape[1]
        
        new_mask = torch.zeros_like(importances, dtype=torch.bool)
        new_mask[idx0[:n_sparse], idx1[:n_sparse]] = True
        return new_mask
    
        
    def update_wanda_like(self, remaining_error:torch.FloatTensor,
                          hessian_diag:torch.FloatTensor, 
                          ):
        
        #calculate the importance
        importances = remaining_error**2 * hessian_diag.unsqueeze(0)
        
        new_mask = self.generate_mask(importances, self.n_sparse)
        # #sort the importances
        # idxs_sorted = torch.argsort(importances.flatten(), descending=True)
        # idx0, idx1 = idxs_sorted//self.n_in, idxs_sorted%self.n_in
        
        # new_mask = torch.zeros_like(self.sparse_mask)
        # new_mask[idx0[:self.n_sparse], idx1[:self.n_sparse]] = True
        
        self.register_buffer('sparse_mask', new_mask)
        
        self.sparse_values.data = -remaining_error[new_mask]
        
    def update_fixed_mask(self, remaining_error:torch.FloatTensor):
        # print("remaining_error", remaining_error)
        self.sparse_values.data = -remaining_error[self.sparse_mask]
        
        
        

    def reconstruct(self):
        if hasattr(self, 'cached_reconstruction'):
            return self.cached_reconstruction
        out = torch.zeros((self.n_out, self.n_in), device=self.sparse_values.device,
                         dtype=self.sparse_values.dtype)
        
        out[self.sparse_mask] = self.sparse_values
        return out
    
    def cache_reconstruct(self):
        self.register_buffer('cached_reconstruction', self.reconstruct())
    
    def delete_cache_reconstruct(self):
        delattr(self, 'cached_reconstruction')
    
    def forward(self, x):
        return F.linear(x, self.reconstruct())
        
    def get_n_bits(self):
        nse = torch.sum(self.sparse_mask)
        nrows = self.n_out
        # n_indicies_bits = torch.ceil(torch.log2(max(nse, nrows)))
        nbits = ((nrows * 16 + (16 + 16)*nse)).item()
        return nbits

