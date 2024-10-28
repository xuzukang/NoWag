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
    X: torch tensor of the weights, rearanged into a shape of (n, d)
    centriods: torch.tensor of the centriods, shape of (k, d)
    weights: torch.tensor of shape (n,d)
    """

    n = X.shape[0]
    assignments = torch.zeros(n, dtype = torch.int64, device = X.device)
    
    for i in range(0, n, subblock_size):
        X_block = X[i:i+subblock_size]
        weights_block = weights[i:i+subblock_size]
        errors = (X_block.unsqueeze(-1) - centriods.T.unsqueeze(0))**2
        #shape of (n, d, k)

        #multiply by the diagonal
        errors = errors * weights_block.unsqueeze(-1)

        #sum by the d
        errors = errors.sum(1)
        #shape of (n, k)
        # print(errors[0,10,:])
        assignments_block = errors.argmin(-1)
        # print(assignments_block[0,10])
        assignments[i:i+subblock_size] = assignments_block
    # errors = (X.unsqueeze(-1) - centriods.T.unsqueeze(0))**2
    # #shape of (n, d, k)

    # #multiply by the diagonal
    # errors = errors * weights.unsqueeze(-1)

    # #sum by the d
    # errors = errors.sum(1)
    # # print("errors[0,10,:] = ", errors[0,10,:])
    # #shape of (n, k)
    # # print(errors[0,10,:])
    # assignments = errors.argmin(-1)
    # # print("assignments[0,10] = ", assignments[0,10])
    # # print("="*10)
    # #shape of (n)
    return assignments

@jit.script
def cluster_m_step(X:torch.Tensor, assignments:torch.Tensor, k:int, weights:torch.Tensor):
    """
    X: torch tensor of the weights, rearanged into a shape of (n, d)
    assignments: torch.tensor of the assignments, shape of (n)
    k: int, number of clusters
    weights: torch.tensor of shape (n, d)
    """
    n, d = weights.shape

    #compute the new centriods
    centriods = torch.zeros((k,d), dtype = weights.dtype, device = weights.device)
    #shape of (k,d)
    for i in range(k):
        assignment_X = X[assignments == i] #shape of (n_i,d)
        assignments_weights = weights[assignments == i] #shape of (n_i,d)

        centriods[i] = torch.sum(assignments_weights * assignment_X, dim = 0) / torch.sum(assignments_weights, dim = 0)

    return centriods



def cluster(X, k, weights, n_iter = 100,
            centriods = None,
            disable_tqdm = False,
            device = 'cuda'):
    """
    weights: torch tensor of the weights, rearanged into a shape of (n, d)
    k: int, number of clusters
    weights: torch.tensor of shape (n, d)
    n_iter: int, number of iterations
    """
    n, d = weights.shape

    #randomly select k centriods
    if centriods is None:
        n_1 = torch.from_numpy(np.random.choice(n, k, replace = False)).to(device)
        # print("n_1", n_1)
        # print("max", torch.max(n_1), "min", torch.min(n_1))
        # print(X.shape)
        centriods = X[n_1, :]
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
    
    def __init__(self, weights:torch.Tensor, hessian:torch.Tensor = None,
                    d:int = 4,
                    n_centriods:int = 256,
                    n_iter:int = 100,
                    normalize_rowwise:bool = False,
                    normalize_columnwise:bool = False,
                    diagonal_only:bool = False,
                    importances:torch.Tensor = None,
                    damping:float = 1e-3):
        """quantize the weights using k-means clustering

        Args:
            weights (torch.Tensor): the weights to quantize of shape (n_out, n_in)
            hessian (torch.Tensor): the hessian of the calibration dataset of shape (n_in, n_in)
            d (int, optional): _description_. Defaults to 4.
            n_centriods (int, optional): _description_. Defaults to 256.
            n_iter (int, optional): _description_. Defaults to 100.
            normalize_rowwise (bool, optional): _description_. Defaults to False.
            normalize_columnwise (bool, optional): _description_. Defaults to False.
            diagonal_only (bool, optional): whether to use the block diagonal or only the diagonal. Defaults to False,
                in which case the weights are reshaped to (n_out, n_in/d, d)
        """
        super().__init__()
        
        self.n_centriods = n_centriods
        self.normalize_rowwise = normalize_rowwise
        self.normalize_columnwise = normalize_columnwise
        self.d = d
        self.n_out, self.n_in = weights.shape
        print("weights.shape = ", weights.shape)
        assert weights.shape[1] % d == 0, "The number of input channels must be divisible by d"
        weights_normalized = weights.clone()
        denoramalize_matrix = torch.ones_like(weights)
        if normalize_rowwise:
            rowwise_norms = torch.norm(weights_normalized, dim = 1)
            rowwise_norms[torch.isclose(rowwise_norms, torch.zeros_like(rowwise_norms))] = 1
            weights_normalized = weights_normalized/rowwise_norms.unsqueeze(0)
            
            self.rowwise_norms = nn.Parameter(rowwise_norms)
            denoramalize_matrix *= rowwise_norms.unsqueeze(0)
            
        if normalize_columnwise:
            columnwise_norms = torch.norm(weights_normalized, dim = 0)
            columnwise_norms[torch.isclose(columnwise_norms, torch.zeros_like(columnwise_norms))] = 1
            weights_normalized = weights_normalized/columnwise_norms.unsqueeze(1)
            
            self.columnwise_norms = nn.Parameter(columnwise_norms)
            denoramalize_matrix *= columnwise_norms.unsqueeze(1)
            
    
                
        # subvector_assignments = torch.arange(weights_normalized.shape[1]).reshape((-1, subvector_dim))

        # weights_reshaped = weights_normalized[:,subvector_assignments] 
        if hessian is not None:
            H_diag = torch.diag(hessian)
            H_diag += damping * torch.mean(H_diag)
            H_diag = H_diag.reshape(-1,d)
            mappings, codebooks = cluster(weights_normalized.reshape((-1, d)), n_centriods, 
                                        (H_diag.unsqueeze(0).expand(weights.shape[0], -1, -1) * denoramalize_matrix.reshape(denoramalize_matrix.shape[0], -1, d)
                                                ).reshape(-1, d), 
                                        n_iter = n_iter,
                                        device = weights.device,
                                        disable_tqdm=True)
        elif importances is not None:
            mappings, codebooks = cluster(weights_normalized.reshape((-1, d)), n_centriods, 
                                        importances.reshape(-1, d), 
                                        n_iter = n_iter,
                                        device = weights.device,
                                        disable_tqdm=True)
        else:
            raise ValueError("Either the hessian or the importances must be provided")
        print("quantized")
        self.centriods = nn.Parameter(codebooks) #shape of (n_centriods, d)
        self.register_buffer("assignments", mappings) #shape of (n_out, n_in/d)
        
    def forward(self):
        
        #recover the weights
        weights_reconstructed = self.centriods[self.assignments] #shape of (n_out, n_in/d, d)
        weights_reconstructed = weights_reconstructed.view(self.n_out, self.n_in)
        
        if self.normalize_rowwise:
            weights_reconstructed = weights_reconstructed * self.rowwise_norms.unsqueeze(0)
        if self.normalize_columnwise:
            weights_reconstructed = weights_reconstructed * self.columnwise_norms.unsqueeze(1)
        
        return weights_reconstructed
    
    def get_n_bits(self):
        encoding_bits = int(np.ceil(np.log2(self.n_centriods))) * self.assignments.numel()
        codebook_bits = self.centriods.numel() * 16
        norm_bits = 0
        if self.normalize_rowwise:
            norm_bits += self.rowwise_norms.numel() * 16
        if self.normalize_columnwise:
            norm_bits += self.columnwise_norms.numel() * 16
            
        return encoding_bits + codebook_bits + norm_bits

        
        
            
            
        
        