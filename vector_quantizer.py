import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import packbits
import tqdm
from quant import *
import low_rank_and_vector as lora_quantizer


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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
    



    

class VectorQuantizerTemp:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev,
                                dtype=W.dtype)
        self.nsamples = 0

    def set_n_samples(self, nsamples):
        self.nsamples = 0
        
    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fastquant(
        self, subvector_dim:int = 16,
        k_magnitude_codebook:int = 256,
        k_cosine_codebook:int = 256,
        keep_top:float = 0.01,
        keep_top_criterion:list = ['magnitude', 'hessian'],
        lr:float = 10,
        lr_multiple:float = 0.9,
        n_iters:int = 100,
        clamp_gradients:float = 1e-1,
        n_iters_cluster:int = 1

    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            raise ValueError("not supported")   
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        H = H.float()
        H = torch.clip(H, -1e6, 1e6)
        del self.H
        
        torch.save({'weights': W, 'H': H}, 'test/original_weights_test.pt')


        # print("average bits:", (np.log2(k_magnitude_codebook) + np.log2(k_cosine_codebook))/d)



        row_assignments = torch.arange(W.shape[1]).reshape(-1, subvector_dim)
# row_assignments = torch.randperm(weights.shape[1]).reshape(-1, d)


        # try:
        weights_reshaped = W[:,row_assignments] 
        #shape of (n, m/d, d)

        H_diag = torch.diag(H)[row_assignments].unsqueeze(0).expand(weights_reshaped.shape[0], -1, -1)
        # H_diag = torch.ones_like(weights_reshaped)
        #shape of (n, m/d, d)

        weights_norms = torch.norm(weights_reshaped, dim = -1)
        # print(H[:,row_assignments].shape)
        H_norms = torch.norm(H[:,row_assignments], dim = (0,-1))
        masks = []
        if 'hessian' in keep_top_criterion:
            print(f"using {keep_top/len(keep_top_criterion)} of the hessian")
            mask_H = H_norms < torch.quantile(H_norms, 1-keep_top/len(keep_top_criterion))
            mask_H = mask_H.unsqueeze(0).expand(W.shape[0], -1)
            masks.append(mask_H)
        if 'magnitude' in keep_top_criterion:
            print(f"using {keep_top/len(keep_top_criterion)} of the magnitude")
            mask_norm = weights_norms < torch.quantile(weights_norms, 1-keep_top/len(keep_top_criterion))
            masks.append(mask_norm)

        if len(masks) == 0 or keep_top == 0.0:
            print("using all the weights")
            mask = torch.ones_like(weights_norms).bool()
        else:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m

        # print("mask", torch.sum(mask), "mask total", mask.numel())
    
        # raise ValueError
        #mask of the top 1% of the weights
        # print(mask.shape)
        weights_norms_masked = weights_norms[mask]
        # print(weights_norms_masked.shape)

        if k_magnitude_codebook == 0:
            n_bits = W.numel() * np.ceil(np.log2(k_cosine_codebook))/subvector_dim + torch.sum(~mask).item()*(32 + 16*subvector_dim)
            weights_use = weights_reshaped[mask,:]
            H_diag_use = H_diag[mask,:]

            best_error = float('inf')

            for i in range(n_iters_cluster):
                mappings, codebooks = cluster(weights_use, k_cosine_codebook, H_diag_use, n_iter = n_iters, disable_tqdm=True,
                                                device = self.dev)
                
                weights_reconstructued_flat =  torch.zeros_like(weights_reshaped)

                weights_reconstructued_flat[~mask,:] = weights_reshaped[~mask]

                weights_reconstructued_flat[mask,:] = codebooks[mappings,:]


                weights_reconstructued = torch.empty_like(W)

                weights_reconstructued[:,row_assignments] = weights_reconstructued_flat.reshape(weights_reconstructued.shape[0], -1, subvector_dim)
                # print(weights_reconstructued)

                diff = W - weights_reconstructued
                average_error = torch.sum(torch.abs(diff)**1)/torch.sum(torch.abs(W)**1)

                H_error = torch.einsum('ik,kl,il->', diff, H/H.shape[0], diff).item()

                if H_error < best_error:
                    best_error = H_error
                    best_codebooks = codebooks.clone()
                    best_mappings = mappings.clone()
            
            # print("mappings", mappings)
            # print("codebooks", codebooks)

            
            prev_loss = float('inf')
            codebooks_use = best_codebooks.clone().requires_grad_(True)
            for i in range(n_iters):
                weights_reconstructued_flat =  torch.zeros_like(weights_reshaped)

                weights_reconstructued_flat[~mask,:] = weights_reshaped[~mask]

                weights_reconstructued_flat[mask,:] = codebooks_use[best_mappings,:]


                weights_reconstructued = torch.empty_like(W)

                weights_reconstructued[:,row_assignments] = weights_reconstructued_flat.reshape(weights_reconstructued.shape[0], -1, subvector_dim)
                # print(weights_reconstructued)

                diff = W - weights_reconstructued
                average_error = torch.sum(torch.abs(diff)**1)/torch.sum(torch.abs(W)**1)

                H_error = torch.einsum('ik,kl,il->', diff, H/H.shape[0], diff)
                # print(f"average error {average_error}, H error {H_error}")
                H_error.backward()


                if H_error > prev_loss:
                    lr = lr * lr_multiple
                    print("reducing lr to ", lr)
                prev_loss = H_error.item()

                if i < n_iters - 1:
                    with torch.no_grad():

                        codebooks_use.grad = torch.clamp(codebooks_use.grad, -clamp_gradients, clamp_gradients)
                        codebooks_use -= lr * codebooks_use.grad
                        codebooks_use.grad.zero_()
        # H_diag_use = torch.clip(H_diag_use, 0,100)
            # raise ValueError

        else:

            raise ValueError("not supported")

        tock = time.time()
        print(round(tock-tick),"H_error", H_error, "average_error", average_error)
        if isinstance(self.layer, transformers.Conv1D):
            weights_reconstructued = weights_reconstructued.t()
        self.layer.weight.data = weights_reconstructued.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        return n_bits, weights_reconstructued.numel()
        
    def structured_sparse_quantize(self,
            subvector_dim:int = 16,
            k_codebook:int = 256,
            keep_top_rowise = 0.7,
            keep_top_colwise = 1,
            lr:float = 10,
            lr_multiple:float = 0.9,
            n_iters:int = 100,
            clamp_gradients:float = 1e-1):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            raise ValueError("not supported")   
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        H = H.float()
        H = torch.clip(H, -1e6, 1e6)
        del self.H
        
        torch.save({'weights': W, 'H': H}, 'test/original_weights_test.pt')
        

        row_mask = self.create_mask(torch.norm(W, dim = 1), keep_top_rowise)
        column_mask = self.create_mask(torch.norm(W, dim = 0), keep_top_colwise/2) & self.create_mask(torch.norm(H, dim = 0), keep_top_colwise/2)

        row_mask = self.mask_round(row_mask, subvector_dim)
        column_mask = self.mask_round(column_mask, subvector_dim)

        mask = row_mask.unsqueeze(1) & column_mask.unsqueeze(0)


        encoding_bits = (np.ceil(np.log2(k_codebook)))/subvector_dim * torch.sum(mask).item()

        sparse_bits = 16 * torch.sum(~mask).item()

        total_bits = encoding_bits + sparse_bits

        weights_masked = W[row_mask,:][:,column_mask]

        subvector_assignments = torch.arange(weights_masked.shape[1]).reshape((-1, subvector_dim))

        weights_reshaped = weights_masked[:,subvector_assignments] 
        #shape of (n, m/d, d)

        H_diag = H[column_mask,column_mask][subvector_assignments]
        #shape of (n, m/d, d)

        mappings, codebooks = cluster(weights_reshaped.reshape(-1,subvector_dim), k_codebook, H_diag.unsqueeze(0).expand(weights_reshaped.shape[0], -1, -1).reshape(-1, subvector_dim), n_iter = n_iters,
                                      device = self.dev, disable_tqdm=True)

        prev_loss = float('inf')
        codebooks_use = codebooks.clone().requires_grad_(True)
        for i in range(n_iters):
            weights_reconstructed = torch.empty_like(weights_masked)
            weights_reconstructed[:,subvector_assignments] = codebooks_use[mappings,:].reshape(weights_reconstructed.shape[0], -1, subvector_dim)




            weights_quantized = torch.empty_like(W)

            weights_quantized[mask] = weights_reconstructed.flatten()
            weights_quantized[~mask] = W[~mask]

            diff = W - weights_quantized
            average_error = torch.sum(torch.abs(diff)**1)/torch.sum(torch.abs(W)**1)

            H_error = torch.einsum('ik,kl,il->', diff, H/H.shape[0], diff)
            # print(f"average error {average_error}, H error {H_error}")
            H_error.backward()


            if H_error > prev_loss:
                lr = lr * lr_multiple
                print("reducing lr to ", lr)
            prev_loss = H_error.item()

            if i < n_iters - 1:
                with torch.no_grad():

                    codebooks_use.grad = torch.clamp(codebooks_use.grad, -clamp_gradients, clamp_gradients)
                    codebooks_use -= lr * codebooks_use.grad
                    codebooks_use.grad.zero_()

        tock = time.time()
        print(round(tock-tick,3),"s","H_error", H_error, "average_error", average_error)
        if isinstance(self.layer, transformers.Conv1D):
            weights_quantized = weights_quantized.t()
        self.layer.weight.data = weights_quantized.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # raise ValueError("stop")
        return total_bits, weights_quantized.numel()
    
    def normalized_clustering(self,
            subvector_dim:int = 16,
            k_codebook:int = 256,
            keep_top_rowise = 0.7,
            keep_top_colwise = 1,
            lr:float = 10,
            lr_multiple:float = 0.9,
            n_iters_cluster:int = 100,
            n_iters:int = 1000,
            clamp_gradients:float = 1e-1,
            eps:float = 1e-4,
            patience:int = 50):
    
        
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            raise ValueError("not supported")   
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        H = H.float()
        H = torch.clip(H, -1e6, 1e6)
        del self.H
        
        # torch.save({'weights': W, 'H': H}, 'test/original_weights_test.pt')
        
        if keep_top_rowise == 0.0:
            print("using all the rows")
            row_mask = torch.ones(W.shape[0], device = W.device).bool()
        else:
            row_mask = self.create_mask(torch.norm(W, dim = 1), keep_top_rowise)
            row_mask = self.mask_round(row_mask, subvector_dim)


        if keep_top_colwise == 0.0:
            print("using all the columns")
            column_mask = torch.ones(W.shape[1], device = W.device).bool()
        else:
            column_mask = self.create_mask(torch.norm(W, dim = 0), keep_top_colwise/2) & self.create_mask(torch.norm(H, dim = 0), keep_top_colwise/2)
            column_mask = self.mask_round(column_mask, subvector_dim)

        mask = row_mask.unsqueeze(1) & column_mask.unsqueeze(0)


        weights_norms_rowwise = torch.norm(W, dim = 0)
        weights_norms_rowwise[torch.abs(weights_norms_rowwise) < 1e-6] = 1
        weights_normalized = W / weights_norms_rowwise.unsqueeze(0)
        weights_norms_columnwise = torch.norm(weights_normalized, dim = 1)
        weights_norms_columnwise[torch.abs(weights_norms_columnwise) < 1e-6] = 1
        weights_normalized = weights_normalized / weights_norms_columnwise.unsqueeze(1)

        denormalize_matrix = weights_norms_rowwise.unsqueeze(0) * weights_norms_columnwise.unsqueeze(1)



        encoding_bits = (np.ceil(np.log2(k_codebook)))/subvector_dim * W.numel()

        sparse_bits = 16 * torch.sum(~mask).item()
        codebook_bits = 16*k_codebook*subvector_dim

        normalize_bits = 16*(weights_norms_rowwise.numel() + weights_norms_columnwise.numel())

        total_bits = encoding_bits + sparse_bits + codebook_bits + normalize_bits

        
        subvector_assignments = torch.arange(weights_normalized.shape[1]).reshape((-1, subvector_dim))

        weights_reshaped = weights_normalized[:,subvector_assignments] 
        #shape of (n, m/d, d)

        H_diag = H[column_mask,column_mask][subvector_assignments]
        #shape of (n, m/d, d)


        mappings, codebooks = cluster(weights_reshaped.reshape(-1,subvector_dim), k_codebook, 
                                      (H_diag.unsqueeze(0).expand(weights_reshaped.shape[0], -1, -1) * denormalize_matrix[:,subvector_assignments]).reshape(-1, subvector_dim), 
                                      n_iter = n_iters_cluster,
                                      device = self.dev, disable_tqdm=True)

        # prev_loss = float('inf')
        # codebooks_use = codebooks.clone().requires_grad_(True)

        # inital_patience = patience
        # for i in range(n_iters):

            

        #     weights_quantized = torch.empty_like(W)
        #     weights_quantized[:,subvector_assignments] = codebooks_use[mappings,:].reshape(weights_quantized.shape[0], -1, subvector_dim)
        #     weights_quantized *= denormalize_matrix
        #     weights_quantized[~mask] = W[~mask]

        #     diff = W - weights_quantized
        #     average_error = torch.sum(torch.abs(diff)**1)/torch.sum(torch.abs(W)**1)

        #     H_error = torch.einsum('ik,kl,il->', diff, H/H.shape[0], diff)
        #     # print(H_error.item(), average_error.item())
        #     # print(f"average error {average_error}, H error {H_error}")
        #     H_error.backward()


        #     if H_error > prev_loss:
        #         lr = lr * lr_multiple
        #         print("reducing lr to ", lr)
        #     prev_loss = H_error.item()

        #     if prev_loss - H_error < eps:
        #         patience -= 1
        #         if patience == 0:
        #             print("stopped after", i, "iterations")
        #             break
        #     else:
        #         patience = inital_patience
            
        #     if H_error < eps:
        #         print("stopped after", i, "iterations")
        #         break
    
        #     if i < n_iters - 1:
        #         with torch.no_grad():

        #             codebooks_use.grad = torch.clamp(codebooks_use.grad, -clamp_gradients, clamp_gradients)
        #             codebooks_use -= lr * codebooks_use.grad
        #             codebooks_use.grad.zero_()

        tock = time.time()
        print("finished in:",round(tock-tick,3),)
        # raise ValueError("stop")
        # if isinstance(self.layer, transformers.Conv1D):
        #     weights_quantized = weights_quantized.t()
        # self.layer.weight.data = weights_quantized.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # raise ValueError("stop")
        return (codebooks, mappings, mask, weights_norms_rowwise, weights_norms_columnwise, W[~mask], W.shape, subvector_assignments), total_bits, W.numel()
    
    def low_rank(self,low_rank_frac:float = 1/16,
                    n_bits:int = 6,
                    sparse_rowise:float = 0,
                    sparse_rowsie_criterion:list[str] = ["weight"],
                    sparse_colwise:float = 0.5,
                    sparse_colwise_criterion:list[str] = ["weight","hessian"],
                    lr:float = 1e-2,
                    lr_multiplier:float = 0.9,
                    grad_clip = 1e-1,
                    eps:float = 1e-3,
                    n_iters = 1000,
                    patience = 100
    ):
        
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            raise ValueError("not supported")   
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        H = H.float()
        H = torch.clip(H, -1e6, 1e6)
        H += torch.eye(H.shape[0], device = H.device) * 1e-5
        del self.H
        
        torch.save({'weights': W, 'H': H}, 'test/original_weights_test.pt')
        # raise ValueError("stop")
        weights_reconstructed, total_bits = lora_quantizer.low_rank_and_quantize(W, H, low_rank_frac = low_rank_frac,
                                                                        n_bits = n_bits,
                                                                        sparse_rowise = sparse_rowise,
                                                                        sparse_rowsie_criterion = sparse_rowsie_criterion,
                                                                        sparse_colwise = sparse_colwise,
                                                                        sparse_colwise_criterion = sparse_colwise_criterion,
                                                                        lr = lr,
                                                                        lr_multiplier = lr_multiplier,
                                                                        grad_clip = grad_clip,
                                                                        eps = eps,
                                                                        n_iters = n_iters,
                                                                        patience = patience,
                                                                        device = self.dev)
        tock = time.time()
        print("total time taken",round(tock-tick,3),"s")

        if isinstance(self.layer, transformers.Conv1D):
            weights_reconstructed = weights_reconstructed.t()
        # self.layer.weight.data = weights_reconstructed.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # raise ValueError("stop")
        return weights_reconstructed.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype), total_bits, weights_reconstructed.numel()
        

    
    

    @staticmethod
    def create_mask(data,percent_top):
        """
        data: torch.tensor of shape (n)
        percent_top: float, the percentage of the top values to keep
        """

        threshold = torch.quantile(data, 1-percent_top/100)
        return data < threshold
    
    @staticmethod
    def mask_round(mask, d, round_up = True):

        indexs = torch.arange(mask.shape[0], device = mask.device)
        if round_up:
            indexs = indexs[~mask]
        else:
            indexs = indexs[mask]

        indexs = indexs[torch.randperm(indexs.shape[0])]
        i = 0
        while mask.sum() % d != 0:
            
            mask[indexs[i]] = ~mask[indexs[i]]
            i += 1

        return mask

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
