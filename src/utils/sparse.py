import torch 
import torch.nn as nn

#a class of sparse stuff

# class SparseParent(nn.Module):
    
#     def reconstruct(self)->torch.Tensor:
#         raise NotImplementedError

#     def forward(self, x:torch.Tensor)->torch.Tensor:
#         raise NotImplementedError
    
#     def update(self, importance: torch.FloatTensor):
#         """updates which elements are sparse,
#         importance is of the same shape as the overall weight matrix and
#         contains the importance of each element"""
#         raise NotImplementedError

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
        

    def update(self, hessian:torch.FloatTensor, 
               quantized_weight:torch.FloatTensor, #shape (n_out, n_in)
               original_weight:torch.FloatTensor, #shape (n_out, n_in)
               block_size:int = 128,
               random_shuffle:bool = False):
        
        quantized_weight_use = quantized_weight.clone()
        
        if n_parallel == -1:
            n_parallel = self.n_out

        #we can do each column together, but we need to reorder the columns
        if random_shuffle:
            idxs = torch.randperm(self.n_in,device = self.sparse_values.device)
        else:
            idxs = torch.arange(self.n_in, device = self.sparse_values.device)
            
        #create a hessian with a zeroed out diagonal
        hessian_zero_diag = hessian.clone() #shape of (n_in, n_in)
        hessian_zero_diag[torch.eye(self.n_out, dtype=torch.bool, device = hessian.device)] = 0
        hessian_diag = hessian.diag() #shape of (n_in)
        
        n_remaining = self.n_sparse
        n_done = 0
        
        new_mask = torch.zeros_like(self.sparse_mask)
        
        for i in range(0, self.n_in, block_size):
            idxs_block = idxs[i:i+block_size]
            hessian_zero_block = hessian_zero_diag[idxs_block,:] #shape of (block_size, n_in)
            hessian_diag_block = hessian_diag[idxs_block] #shape of (block_size)
            
            first_order_term = 2 * torch.einsum('ij,kj->ik', quantized_weight_use - original_weight, hessian_zero_block) #shape (n_out, block_size)
            
            element_wise_block_error = quantized_weight_use - original_weight #shape (n_out, n_in)
            before_sparse_error = element_wise_block_error[:, idxs_block] ** 2 * hessian_diag_block.unsqueeze(0) + element_wise_block_error[:, idxs_block] * element_wise_block_error  #shape (n_out, block_size)
            
            best_post_sparse_error = -first_order_term**2/hessian_diag_block.unsqueeze(0) #shape (n_out, block_size)
            sparse_optimal_values = - first_order_term/hessian_diag_block.unsqueeze(0) - element_wise_block_error[:, idxs_block]
            reduction = before_sparse_error - best_post_sparse_error 
            
            if i + block_size >= self.n_in:
                n_sparse_block = n_remaining
            else:
                n_sparse_block = int(self.frac_sparse * block_size * self.n_out)
                
            
            _, flattened_idxs = torch.topk(reduction.view(-1), n_sparse_block)
            
            sparse_optimal_values = sparse_optimal_values.view(-1)[flattened_idxs]
            self.sparse_values.data[n_done:n_done+len(sparse_optimal_values)] = sparse_optimal_values
            row_idxs = flattened_idxs // block_size
            col_idxs = idxs_block[flattened_idxs % block_size]
            
            new_mask[row_idxs, col_idxs] = True
            
            quantized_weight_use[row_idxs, col_idxs] = quantized_weight_use[row_idxs, col_idxs] + sparse_optimal_values
                #pick the best int(n_parallel)
        
        self.register_buffer('sparse_mask', new_mask)

    def reconstruct(self, quantized_weight:torch.FloatTensor):
        reconstructed = quantized_weight.clone()
        reconstructed[self.sparse_mask] = self.sparse_values
        return reconstructed
        

