import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import tqdm
import torch.jit as jit

@jit.script
def cluster_e_step(X:torch.Tensor,centriods:torch.Tensor,
                   weights:torch.Tensor,
                     subblock_size:int = 1024):
    
    """
    X: torch tensor of the weights, rearanged into a shape of (n,m/d,d)
    centriods: torch.tensor of the centriods, shape of (d,k)
    weights: torch.tensor of shape (m/d,d,d) or m/n,d)
    """

    n, m_d, d = X.shape
    assignments = torch.zeros((n,m_d), dtype = torch.int64, device = X.device)
    
    for i in range(n):
        errors =  (X[i].unsqueeze(-1) - centriods.unsqueeze(0)) #shape of (m/d,d,k)
        
        if len(weights.shape) == 3:
            errors = torch.einsum("mik,mij,mjk->mk", errors, weights, errors)
        else:
            errors = torch.einsum("mik,mi,mik->mk", errors, weights, errors)
        
        assignments[i] = errors.argmin(-1)
    return assignments


@jit.script
def cluster_m_step(X:torch.Tensor, assignments:torch.Tensor, k:int, weights:torch.Tensor):
    """
    X: torch tensor of the weights, rearanged into a shape of (n,m/d,d)
    assignments: torch.tensor of the assignments, shape of (n,m/d)
    k: int, number of clusters
    weights: torch.tensor of shape (m/d,d,d) or (m/n,d)
    """
    n, m_d, d = X.shape

    #compute the new centriods
    centriods = torch.zeros((d,k), dtype = weights.dtype, device = weights.device)
    #shape of (k,d)
    
    # assignment_counts = torch.unique(assignments, return_counts = True)[1]
    if len(weights.shape) == 3:
        for i in range(k):
            mask =  assignments == i
            H_sum = torch.zeros_like(weights[0])
            centriod = torch.zeros_like(centriods[:,i])
            n_samples = 0
            for j in range(n):
                mask_j = mask[j]
                H_sum *= n_samples/(n_samples + torch.sum(mask_j))
                centriod *= n_samples/(n_samples + torch.sum(mask_j))
                
                n_samples += torch.sum(mask_j)
                masked_weights = weights[mask_j] #shape of (n_i,d,d)
                masked_X = X[j][mask_j] #shape of (n_i,d)
                H_sum += torch.sum(masked_weights, dim = 0)/n_samples
                centriod += torch.einsum("mij,mj->i", masked_weights, masked_X)/n_samples
            
            centriods[:,i] = torch.linalg.pinv(H_sum, hermitian = True) @ centriod
    else:
        for i in range(k):
            mask =  assignments == i
            H_sum = torch.zeros_like(weights[0])
            centriod = torch.zeros_like(centriods[:,i])
            n_samples = 0
            for j in range(n):
                mask_j = mask[j]
                H_sum *= n_samples/(n_samples + torch.sum(mask_j))
                centriod *= n_samples/(n_samples + torch.sum(mask_j))
                n_samples += torch.sum(mask_j)
                masked_weights = weights[mask_j] #shape of (n_i,d)
                masked_X = X[j][mask_j] #shape of (n_i,d)
                H_sum += torch.sum(masked_weights, dim = 0)/n_samples #shape of (d)
                centriod += torch.einsum("mi,mi->i", masked_weights, masked_X)/n_samples
            
            centriods[:,i] = centriods/H_sum

    return centriods



def cluster(X, k, weights, n_iter = 100,
            centriods = None,
            disable_tqdm = False,
            device = 'cuda'):
    """
    weights: torch tensor of the weights, rearanged into a shape of (n, m/d, d)
    k: int, number of clusters
    weights: torch.tensor of shape (m/d,d,d) or (m/n,d)
    n_iter: int, number of iterations
    """
    n, d = weights.shape

    #randomly select k centriods
    if centriods is None:
        n_1 = torch.from_numpy(np.random.choice(n, k, replace = False)).to(device)
        # print("n_1", n_1)
        # print("max", torch.max(n_1), "min", torch.min(n_1))
        # print(X.shape)
        centriods = X[n_1, :].T
        # print(centriods)
    #shape of (k, d)
    for i in tqdm.tqdm(range(n_iter), disable = disable_tqdm, miniters= n_iter//10):
        # print("X.shape = ", X.shape, "centriods.shape = ", centriods.shape, "weights.shape = ", weights.shape)
        assignments = cluster_e_step(X, centriods, weights)
        # print(assignments)
        # print(assignments.shape)
        centriods = cluster_m_step(X, assignments, k, weights)
        if i > 0:
            if torch.all(assignments == assignments_old):
                # print("breaking at iteration", i)
                break
            # print("n_change:", torch.sum(assignments != assignments_old))
        assignments_old = assignments.clone()
    return assignments, centriods


class Quantize(nn.Module):
    
    def __init__(self, weights:torch.Tensor, hessian:torch.Tensor,
                    d:int = 4,
                    n_centriods:int = 256,
                    n_iter:int = 100,
                    normalize_rowise:bool = False,
                    normalize_columnwise:bool = False,
                    diagonal_only:bool = False):
        """quantize the weights using k-means clustering

        Args:
            weights (torch.Tensor): the weights to quantize of shape (n_out, n_in)
            hessian (torch.Tensor): the hessian of the calibration dataset of shape (n_in, n_in)
            d (int, optional): _description_. Defaults to 4.
            n_centriods (int, optional): _description_. Defaults to 256.
            n_iter (int, optional): _description_. Defaults to 100.
            normalize_rowise (bool, optional): _description_. Defaults to False.
            normalize_columnwise (bool, optional): _description_. Defaults to False.
            diagonal_only (bool, optional): whether to use the block diagonal or only the diagonal. Defaults to False,
                in which case the weights are reshaped to (n_out, n_in/d, d)
        """
        super().__init__()
        
        self.n_centriods = n_centriods
        self.normalize_rowise = normalize_rowise
        self.normalize_columnwise = normalize_columnwise
        self.d = d
        self.n_in, self.n_out = weights.shape
        
        assert weights.shape[1] % d == 0, "The number of input channels must be divisible by d"
        weights_normalized = weights.clone()
        
        if normalize_rowise:
            rowwise_norms = torch.norm(weights_normalized, dim = 1)
            rowwise_norms[torch.isclose(rowwise_norms, torch.zeros_like(rowwise_norms))] = 1
            weights_normalized = weights_normalized/rowwise_norms.unsqueeze(0)
            
            self.rowwise_norms = nn.Parameter(rowwise_norms)
            
        if normalize_columnwise:
            columnwise_norms = torch.norm(weights_normalized, dim = 0)
            columnwise_norms[torch.isclose(columnwise_norms, torch.zeros_like(columnwise_norms))] = 1
            weights_normalized = weights_normalized/columnwise_norms.unsqueeze(1)
            
            self.columnwise_norms = nn.Parameter(columnwise_norms)
            
        if diagonal_only:
            clustering_weights = torch.diag(hessian)
        else:
            clustering_weights = torch.zeros((weights.shape[1]//d, d, d), dtype = weights.dtype, device = weights.device)
            for i in range(weights.shape[1]//d):
                clustering_weights[i] = hessian[d*i:d*(i+1), d*i:d*(i+1)]
                
        
        assignments, centriods = cluster(weights_normalized, n_centriods, clustering_weights, n_iter = n_iter)   
        
        self.centriods = nn.Parameter(centriods.T) #shape of (n_centriods, d)
        self.register_buffer("assignments", assignments) #shape of (n_out, n_in/d)
        
    def forward(self):
        
        #recover the weights
        weights_reconstructed = self.centriods[self.assignments] #shape of (n_out, n_in/d, d)
        weights_reconstructed = weights_reconstructed.view(self.n_out, self.n_in)
        
        if self.normalize_rowise:
            weights_reconstructed = weights_reconstructed * self.rowwise_norms.unsqueeze(0)
        if self.normalize_columnwise:
            weights_reconstructed = weights_reconstructed * self.columnwise_norms.unsqueeze(1)
        
        return weights_reconstructed
    
    def get_n_bits(self):
        encoding_bits = np.ceil(np.log2(self.n_centriods)) * self.assignments.numel()
        codebook_bits = self.centriods.numel() * 16
        norm_bits = 0
        if self.normalize_rowise:
            norm_bits += self.rowwise_norms.numel() * 16
        if self.normalize_columnwise:
            norm_bits += self.columnwise_norms.numel() * 16
            
        return encoding_bits + codebook_bits + norm_bits

        
        
            
            
        
        