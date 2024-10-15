import torch
import torch.nn as nn
import numpy as np 


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


class low_rank_module(nn.Module):

    def __init__(self,X,low_rank):
        super(low_rank_module, self).__init__()
        U, S, V = torch.svd(X)

        self.A = nn.Parameter(U[:, :low_rank] @ torch.sqrt(torch.diag(S[:low_rank])))
        self.B = nn.Parameter(torch.sqrt(torch.diag(S[:low_rank])) @ V[:, :low_rank].T)

    def forward(self):
        return self.A @ self.B
    
class lora_quantizer_module(nn.Module):
    def __init__(self, A, B, n_bits):
        super(lora_quantizer_module, self).__init__()

        self.A_codebook = nn.Parameter(torch.linspace(A.min(), A.max(), 2**n_bits).to(A.device))
        self.B_codebook = nn.Parameter(torch.linspace(B.min(), B.max(), 2**n_bits).to(B.device))

    def forward(self, A_assignments, B_assignments):
        return self.A_codebook[A_assignments] @ self.B_codebook[B_assignments]
    
    def recompute_assignments(self, A, B):
        
        with torch.no_grad():
            A_assignments = torch.argmin(torch.abs(self.A_codebook.reshape([1]*len(A.shape) + list(self.A_codebook.shape)) - A.unsqueeze(-1)), dim = -1)
            B_assignments = torch.argmin(torch.abs(self.B_codebook.reshape([1]*len(B.shape) + list(self.B_codebook.shape)) - B.unsqueeze(-1)), dim = -1)
            return A_assignments, B_assignments


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
                
def low_rank_and_quantize(weights:torch.Tensor,
                          H:torch.Tensor,
                            low_rank_frac:int = 1/16,
                            n_bits:int = 8,
                            sparse_rowise:float = 0,
                            sparse_rowsie_criterion:list[str] = ["weight"],
                            sparse_colwise:float = 0,
                            sparse_colwise_criterion:list[str] = ["weight","hessian"],
                            lr:float = 1e-2,
                            lr_multiplier:float = 0.9,
                            grad_clip = 1e-1,
                            eps:float = 1e-3,
                            n_iters = 1000,
                            patience = 100,
                            device = None):
    
    low_rank = int((weights.shape[0] + weights.shape[1])/2 * low_rank_frac)
    print("using low rank = ", low_rank)
    if sparse_colwise > 0:
        column_mask = create_mask(weights, H, sparse_colwise, sparse_colwise_criterion, dim = 0)
    else:
        column_mask = torch.ones(weights.shape[1]
                                 , dtype = torch.bool, device = weights.device)
    
    if sparse_rowise > 0:
        row_mask = create_mask(weights, H, sparse_rowise, sparse_rowsie_criterion, dim = 1)
    else:
        row_mask = torch.ones(weights.shape[0]
                              , dtype = torch.bool, device = weights.device)

    print("row mask = ", row_mask.sum().item(), "column mask = ", column_mask.sum().item())
    mask = row_mask.unsqueeze(1) & column_mask.unsqueeze(0)

    print("row_mask.shape = ", row_mask.shape, "column_mask.shape = ", column_mask.shape)
    weights_adjusted = weights[row_mask,:][:, column_mask]
    weights_norms_rowwise = torch.norm(weights_adjusted, dim = 0)
    weights_norms_rowwise[torch.isclose(weights_norms_rowwise, torch.zeros_like(weights_norms_rowwise))] = 1
    weights_normalized = weights_adjusted / weights_norms_rowwise.unsqueeze(0)
    weights_norms_columnwise = torch.norm(weights_normalized, dim = 1)
    weights_norms_columnwise[torch.isclose(weights_norms_columnwise, torch.zeros_like(weights_norms_columnwise))] = 1
    weights_normalized = weights_normalized / weights_norms_columnwise.unsqueeze(1)

    lora_module = low_rank_module(weights_normalized, low_rank).to(device)

    lora_optimizer = torch.optim.Adam(lora_module.parameters(), lr = lr)
    lora_optimizer_scheduler = torch.optim.lr_scheduler.StepLR(lora_optimizer, step_size = 1, gamma = lr_multiplier)

    used_patience = 0
    prev_H_error = 1e10

    for i in range(n_iters):
        lora_optimizer.zero_grad()
        weights_reconstructed = construct_weights(lora_module, weights, mask, weights_norms_rowwise, weights_norms_columnwise)   
        diff = weights - weights_reconstructed

        H_error = torch.einsum('ik,kl,il->', diff, H/H.shape[0], diff)
        

        if i % 50 == 0:
            print("i",i,"H_error = ", H_error.item())
        H_error.backward()

        if prev_H_error - H_error < eps:
            lora_optimizer_scheduler.step()

            used_patience += 1
            if used_patience >= patience:
                break
        
        else:
            used_patience = 0
            prev_H_error = H_error.item()

        nn.utils.clip_grad_norm_(lora_module.parameters(), grad_clip)
        lora_optimizer.step()
        
    print("LoRA finished")
    print(f"Converged in {i} iterations final H_error = {H_error.item()}")
    if n_bits < 16:
        A = lora_module.A.detach().requires_grad_(False)
        B = lora_module.B.detach().requires_grad_(False)
        
        quantizer = lora_quantizer_module(A, B, n_bits).to(device)
        quantizer_optimizer = torch.optim.Adam(quantizer.parameters(), lr = lr)
        quantizer_optimizer_scheduler = torch.optim.lr_scheduler.StepLR(quantizer_optimizer, step_size = 1, gamma = lr_multiplier)

        A_assignments, B_assignments = quantizer.recompute_assignments(A,B)

        used_patience = 0
        prev_H_error = 1e10
        for j in range(n_iters):
            quantizer_optimizer.zero_grad()
            weights_reconstructed = construct_weights(quantizer, weights, mask, weights_norms_rowwise, weights_norms_columnwise,
                                                    {"A_assignments":A_assignments, "B_assignments":B_assignments})
            diff = weights - weights_reconstructed

            H_error = torch.einsum('ik,kl,il->', diff, H/H.shape[0], diff)
            if j % 50 == 0:
                print("j",j,"H_error = ", H_error.item())
            H_error.backward()

            if prev_H_error - H_error < eps:
                quantizer_optimizer_scheduler.step()

                used_patience += 1
                if used_patience >= patience:
                    break
            
            else:
                used_patience = 0
                prev_H_error = H_error.item()

            nn.utils.clip_grad_norm_(quantizer.parameters(), grad_clip)
            quantizer_optimizer.step()

            A_assignments, B_assignments = quantizer.recompute_assignments(A,B)

        print("Quantizer finished")
        print(f"Converged in {j} iterations final H_error = {H_error.item()}")

        total_bits = n_bits * (A_assignments.numel() + B_assignments.numel()) +\
                        16 * (torch.sum(~mask).item() + 2*2**n_bits)
        
    
        with torch.no_grad():
            weights_reconstructed = construct_weights(quantizer, weights, mask, weights_norms_rowwise, weights_norms_columnwise,
                                                  {"A_assignments":A_assignments, "B_assignments":B_assignments})
        
    else:
        print("skipping quantization")
        total_bits = 16 * ( torch.sum(~mask).item() + lora_module.A.numel() + lora_module.B.numel())
        weights_reconstructed = construct_weights(lora_module, weights, mask, weights_norms_rowwise, weights_norms_columnwise)
        
    return weights_reconstructed, total_bits

