import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import tqdm
import torch.jit as jit
import os 
import sys 
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
import src.utils.alignment.quantize_align as quantize_align

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
    with torch.no_grad():
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




class Quantize(nn.Module):
    
    def __init__(self, centriods:torch.Tensor,
                 assignments:torch.Tensor,
                 rowwise_norms:torch.Tensor = None,
                    columnwise_norms:torch.Tensor = None,
                    n_out:int = None,
                    n_in:int = None):

        super().__init__()
        self.centriods = nn.Parameter(centriods)
        self.n_centriods, self.d = centriods.shape
        self.register_buffer("assignments", assignments)
        self.rowwise_norms = nn.Parameter(rowwise_norms) if rowwise_norms is not None else None
        self.normalize_rowwise = rowwise_norms is not None
        self.columnwise_norms = nn.Parameter(columnwise_norms) if columnwise_norms is not None else None
        self.normalize_columnwise = columnwise_norms is not None
        self.n_out = n_out
        self.n_in = n_in
        
    # def disable_grad(self):
    #     self.centriods.requires_grad = False
    #     if self.rowwise_norms is not None:
    #         self.rowwise_norms.requires_grad = False
    #     if self.columnwise_norms is not None:
    #         self.columnwise_norms.requires_grad = False
                 
    @staticmethod
    def quantize(
                 weights:torch.Tensor, hessian:torch.Tensor,
                    d:int = 4,
                    n_centriods:int = 256,
                    n_iter:int = 100,
                    normalize_rowwise:bool = False,
                    normalize_columnwise:bool = False,
                    align_regularization:float = 1e-3,
                    align_lr:float = 1e-3,
                    align_lr_multiplier:float = 0.9,
                    align_early_stop_eps:float = 1e-3,
                    align_early_stop_patience:int = 20,
                    align_clip_grad:float = 0,
                    n_sub_steps:int = 1,
                    damping:float = 1e-3,
                    seed:int = 0,
                    debug:bool = False,
                    debug_path:str = ""):
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
        
        n_centriods = n_centriods
        n_out, n_in = weights.shape
        # print("weights.shape = ", weights.shape)
        # torch.save({"weights":weights, "hessian":hessian}, "test/weights_hessian.pt")
        assert weights.shape[1] % d == 0, "The number of input channels must be divisible by d"
        weights_normalized = weights.clone()
        denoramalize_matrix = torch.ones_like(weights)
        if normalize_rowwise:
            rowwise_norms = torch.norm(weights_normalized, dim = 0)
            rowwise_norms[torch.isclose(rowwise_norms, torch.zeros_like(rowwise_norms))] = 1
            weights_normalized = weights_normalized/rowwise_norms.unsqueeze(0)
            denoramalize_matrix *= rowwise_norms.unsqueeze(0)
        else:
            rowwise_norms = None
            
        if normalize_columnwise:
            columnwise_norms = torch.norm(weights_normalized, dim = 1)
            columnwise_norms[torch.isclose(columnwise_norms, torch.zeros_like(columnwise_norms))] = 1
            weights_normalized = weights_normalized/columnwise_norms.unsqueeze(1)
            denoramalize_matrix *= columnwise_norms.unsqueeze(1)
        else:
            columnwise_norms = None
            
        if seed is not None:
            np.random.seed(seed)
                
        # subvector_assignments = torch.arange(weights_normalized.shape[1]).reshape((-1, subvector_dim))

        # weights_reshaped = weights_normalized[:,subvector_assignments] 

        weights_subvectors = weights_normalized.reshape((-1, d))

        #add a damping to the hessian
        idxs = torch.arange(hessian.shape[0], device=hessian.device)
        hessian[idxs, idxs] = torch.clip(hessian[idxs, idxs], 1e-5)
        if damping > 0:
            hessian[idxs, idxs] += damping * torch.mean(hessian[idxs, idxs])
        H_diag = torch.diag(hessian)
        H_diag = H_diag.reshape(-1,d)
        importances = (H_diag.unsqueeze(0).expand(weights.shape[0], -1, -1) * denoramalize_matrix.reshape(denoramalize_matrix.shape[0], -1, d)
                                                ).reshape(-1, d)
        
        n_subvectors = weights_subvectors.shape[0]


        n_1 = torch.from_numpy(np.random.choice(n_subvectors, n_centriods, replace = False)).to(weights.device)
        # print("n_1", n_1)
        # print("max", torch.max(n_1), "min", torch.min(n_1))
        # print(X.shape)
        centriods = weights_subvectors[n_1, :]
        
        # print(weights.dtype, centriods.dtype, importances.dtype)
        # print(hessian.dtype,hessian.shape)
        def reconstruction_fn(centriods:torch.Tensor,
                                  assignments:torch.Tensor,
                                  denormalize_matrix:torch.Tensor):
                
                weights_reconstructed = centriods[assignments].reshape(weights_normalized.shape)
                weights_reconstructed *= denormalize_matrix
                return weights_reconstructed
        
        #first cluster
        for i in range(n_iter):
            assignments = cluster_e_step(
                weights_subvectors, centriods, importances)
            # print(assignments)
            # print(assignments.shape)
            centriods = cluster_m_step(weights_subvectors, assignments, n_centriods, importances)
            if i > 0:
                if torch.all(assignments == assignments_old):
                    # print("breaking at iteration", i)
                    break
                # print("n_change:", torch.sum(assignments != assignments_old))
            assignments_old = assignments.clone()
        
        best_centriods = centriods
        best_assignments = assignments
        
        #align again n_iter times with a new optimizer
        print("realigning")
        centriods = best_centriods.requires_grad_(True)
        centriods_optimizer = torch.optim.Adam([centriods], lr = align_lr)
        centriods_scheduler = torch.optim.lr_scheduler.StepLR(centriods_optimizer, step_size = 1, gamma = align_lr_multiplier)
        hessian_use = hessian.clone()/hessian.shape[0]
        ids = torch.arange(hessian.shape[0], device=hessian.device)
        hessian_use[ids, ids] += align_regularization
        # print(hessian_use.dtype)
        prev_loss = np.inf
        remaining_patience = align_early_stop_patience
            
        for i in range(n_iter):
            loss,centriods = quantize_align.align_cluster_one_step(
                weights, centriods,
                reconstruction_fn,
                {"assignments":assignments, "denormalize_matrix":denoramalize_matrix},
                hessian_use,
                centriods_optimizer,
                align_clip_grad
            )
            if i % (n_iter//10) == 0:
                print(i,loss)
            # print(i,loss,remaining_patience)
            if i == 0:
                print("initial loss", loss)
            if loss < align_early_stop_eps:
                break
            if loss > prev_loss - align_early_stop_eps:
                remaining_patience -= 1
                if remaining_patience == 0:
                    break
                # align_lr *= align_lr_multiplier
                # print("reducing lr to", align_lr)
                centriods_scheduler.step()
                n_sub_steps *= 0.5
                n_sub_steps = max(1, int(n_sub_steps))
            else:
                best_centriods = centriods.clone()
                best_assignments = assignments.clone()
                remaining_patience = align_early_stop_patience
                prev_loss = loss
        print("final loss", prev_loss)
            
            
        print("quantized")
        raise ValueError("done")
        return Quantize(best_centriods.detach().clone(), best_assignments, rowwise_norms, columnwise_norms, n_out, n_in)
        
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

        
        
            
            
        
if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)
    
    data = torch.load("test/weights_hessian.pt")
    weights = data["weights"]
    hessian = data["hessian"]
    
    import time
    t = time.time()
    quantized = Quantize.quantize(weights, hessian, 
                                  d = 4, 
                                  n_centriods = 256, 
                                  n_iter = 100, 
                                  normalize_rowwise = True, 
                                  normalize_columnwise = True, 
                                  align_regularization = 0, 
                                  align_lr = 1e-4, 
                                  align_lr_multiplier = 0.9, 
                                  align_early_stop_eps = 1e-3, 
                                  align_early_stop_patience = 100, 
                                  align_clip_grad = 1e-1, 
                                  damping = 0)
    
    print("time", time.time() - t)