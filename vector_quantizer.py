import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import packbits
import tqdm
from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def cluster_e_step(X,centriods,
                   weights):
    
    """
    X: torch tensor of the weights, rearanged into a shape of (n, d)
    centriods: torch.tensor of the centriods, shape of (k, d)
    weights: torch.tensor of shape (n,d)
    """

    # k = centriods.shape[0]

    errors = (X.unsqueeze(-1) - centriods.T.unsqueeze(0))**2
    #shape of (n, d, k)

    #multiply by the diagonal
    errors = errors * weights.unsqueeze(-1)

    #sum by the d
    errors = errors.sum(1)
    # print("errors[0,10,:] = ", errors[0,10,:])
    #shape of (n, k)
    # print(errors[0,10,:])
    assignments = errors.argmin(-1)
    # print("assignments[0,10] = ", assignments[0,10])
    # print("="*10)
    #shape of (n)
    return assignments


def cluster_m_step(X, assignments, k, weights):
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
    

    


class VectorQuantizerLayer(nn.Module):
    def __init__(self, 
                 original_layer: nn.Linear,
                 n_quantize : int = 128,
                 nsamples : int = 128):
        

        super(VectorQuantizerLayer, self).__init__()

        self.original_weights = original_layer.weight.data
        # print("original weights", self.original_weights.shape,"dtype", self.original_weights.dtype, "device", self.original_weights.device)
        self.b = original_layer.bias
        # self.H = torch.zeros((self.original_weights.shape[1], self.original_weights.shape[1]), 
        #                      device=self.original_weights.device).float() #will cast to half later
        
        self.Input = []
        self.Output = []
        self.n_quantize = n_quantize
        self.nsamples = nsamples

                 

    def add_batch(self, inp):
        inp = inp.reshape((-1, inp.shape[-1])).float()
        inp = inp.t()
        # print(inp.shape)
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        assert torch.all(torch.isfinite(self.H)), f"H is not finite, {self.H}, {self.H[~torch.isfinite(self.H)]}"

    def quantize(self):
        # assert torch.all(torch.isfinite(self.H)), f"H is not finite, {self.H}, {self.H[~torch.isfinite(self.H)]}"

        #save the original weights, and H
        torch.save({'weights': self.original_weights, 'bias': self.b,
                    'Input': self.Input, 'Output': self.Output,
                    }, 'test/original_weights2.pt')
        raise ValueError("stop")
        quantized_vectors,assignments = vector_quantize(self.original_weights
                                                                  , (self.H/self.H.shape[0]).to(self.original_weights.dtype), self.n_quantize)
        b = self.b
        # print("self.bias", self.b)
        # self.quantized_vectors,self.assignments = vector_quantize(self.original_weights.float()
        #                                                           , self.H.float(), self.n_quantize)

        # #some more debug code
        # for i in range(self.original_weights.shape[1]):
        #     print(self.original_weights[i], "\n", quantized_vectors[:,assignments[i]])
        #     print()
        #     break

        # self.quantized_vectors = self.quantized_vectors.to(self.original_weights.dtype)

        #add both as parameters to save but no gradients
        self.register_buffer('quantized_vectors', quantized_vectors)
        self.register_buffer('assignments', assignments)
        self.register_buffer('bias', b)
        # print("self.bias", self.bias)
        # print("quantized, to self.quantized_vectors and self.assignments")


        #delete the original weights and H
        del self.original_weights
        del self.H
        # del self.b

    def forward(self, x):
        #calculate the quantized weights
        #normally what we would do is 
        #Y = XW^T + b
        #X is of shape (batch_size, n)

        #W is of shape (n,k)
        #if we have not quantized the weights
        if not hasattr(self, 'quantized_vectors'):
            # self.add_batch(x)
            y = F.linear(x, self.original_weights, self.b)
            self.Input.append(x)    
            self.Output.append(y)
            raise ValueError("stop")
            return y
        
        # print("original shape", x.shape)
        original_shape = x.shape
        quantized_multiplications = torch.einsum('ij,jk->ik', x.reshape(-1, self.quantized_vectors.shape[0]), 
                                                 self.quantized_vectors[self.assignments]) #shape (batch_size, k)
        if self.bias is not None:
            quantized_output = quantized_multiplications[:,self.assignments] + self.bias.unsqueeze(0) #shape (batch_size, n)
        else:
            # print(quantized_multiplications.shape)
            quantized_output = quantized_multiplications[:,self.assignments]

            # print(quantized_output.shape)
        

        #debug code
        # quantized_output = quantized_output.reshape(original_shape)
        # expected_output = F.linear(x, self.original_weights, self.b)
        
        # for i in range(x.shape[1]):
        #     print(quantized_output[0,i], "\n", expected_output[0,i])
        #     print()

        #     raise ValueError("stop")


        return quantized_output.reshape(original_shape)

    def extra_repr(self):
        return f"quantized to {self.n_quantize} vectors"
    
    def __repr__(self):
        return f"VectorQuantizerLayer(nn.Linear({self.original_weights.shape[1]}, {self.original_weights.shape[0]}, quantized to {self.n_quantize} vectors))"
    


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
        lr:float = 10,
        lr_multiple:float = 0.9,
        n_iters:int = 100,
        clamp_gradients:float = 1e-1,

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

        print(keep_top)
        mask_H = H_norms < torch.quantile(H_norms, 1-keep_top/2)
        mask_H = mask_H.unsqueeze(0).expand(W.shape[0], -1)


        mask_norm = weights_norms < torch.quantile(weights_norms, 1-keep_top/2)

        mask = mask_norm & mask_H
        # print("mask", torch.sum(mask), "mask total", mask.numel())
    
        # raise ValueError
        #mask of the top 1% of the weights

        weights_norms_masked = weights_norms[mask]
        print(weights_norms_masked.shape)
        weights_use = weights_reshaped[mask,:]/weights_norms_masked.unsqueeze(-1)
        H_diag_use = H_diag[mask,:] * weights_norms_masked.unsqueeze(-1)**2
        # H_diag_use = torch.clip(H_diag_use, 0,100)


        #first we get the magnitude codebook
        magnitude_assignments, magnitude_codebook = cluster(torch.log(weights_norms_masked).unsqueeze(-1), k_magnitude_codebook, 
                                                                torch.ones_like(weights_norms_masked).unsqueeze(-1),
                                                                n_iter = 1000,
                                                                disable_tqdm = False,
                                                                device = self.dev)

        #try binning the magnitudes
        # magnitude_codebook = np.linspace(np.log(weights_norms_masked.min()), np.log(weights_norms_masked.max()), k_magnitude_codebook).reshape(-1,1)
        # magnitude_assignments = np.argmin(np.abs(np.log(weights_norms_masked.numpy()).reshape(-1,1) - magnitude_codebook.T), axis = 1)



        codebooks = {}
        assignments_dict = {}
        for i in tqdm.tqdm(range(k_magnitude_codebook), miniters = k_magnitude_codebook//10):
            mask_i = magnitude_assignments == i
            if torch.sum(mask_i) <= k_cosine_codebook:
                codebooks[i] = weights_use[mask_i,:]
                assignments_dict[i] = torch.arange(torch.sum(mask_i).item())
                continue
            
            # print(weights_use[mask_i,:].shape, H_diag_use[mask_i,:].shape)
            assignments, centriods = cluster(weights_use[mask_i,:], 
                                             k_cosine_codebook, 
                                             H_diag_use[mask_i,:], 
                                             n_iter = 25,
                                             disable_tqdm = True,
                                             device = self.dev)
            codebooks[i] = centriods
            assignments_dict[i] = assignments

        
        #gradient descent to optimize the codebooks
        codebooks_use = {}
        prev_loss = float('inf')
        for i in range(k_magnitude_codebook):
            codebooks_use[i] = codebooks[i].clone().requires_grad_(True)

        for iter in tqdm.tqdm(range(n_iters), miniters = n_iters//10):
            weights_reconstructued_flat =  torch.zeros_like(weights_reshaped)

            weights_reconstructued_flat[~mask,:] = weights_reshaped[~mask]
            # print(codebooks_use[0])
            for i in range(k_magnitude_codebook):
                mask_i = magnitude_assignments == i
                # assert torch.any(mask_i)
                mask_ = torch.zeros_like(mask)
                mask_[mask] = mask_i
                weights_reconstructued_flat[mask_,:] = torch.clip(codebooks_use[i][assignments_dict[i],:],-1,1) * torch.exp(magnitude_codebook[i])
                # if i == 0:
                #     print(codebooks_use[i])
                #     print(assignments_dict[i])
                #     print(codebooks_use[i][assignments_dict[i],:])
                #     print(torch.clip(codebooks_use[i][assignments_dict[i],:],-1,1))
                #     print(torch.clip(codebooks_use[i][assignments_dict[i],:],-1,1) * torch.exp(magnitude_codebook[i]))
                #     print(weights_reconstructued_flat[mask_,:])
            # weights_reconstructued_flat[mask,:] = centriods[assignments,:] * torch.from_numpy(np.exp(magnitude_codebook[:,0][magnitude_assignments])).reshape(-1,1).float()
            # print(weights_reconstructued_flat)

            weights_reconstructued = torch.empty_like(W)

            weights_reconstructued[:,row_assignments] = weights_reconstructued_flat.reshape(weights_reconstructued.shape[0], -1, subvector_dim)
            # print(weights_reconstructued)

            diff = W - weights_reconstructued
            # print(diff)
            average_error = torch.sum(torch.abs(diff)**1)/torch.sum(torch.abs(W)**1)
            H_error = torch.einsum('ik,kl,il->', diff, H/H.shape[0], diff)
            # print(H_error)
            # print(f"average error {average_error}, H error {H_error}")
            H_error.backward()
            # losses.append(H_error.item())
            if H_error > prev_loss:
                lr = lr * lr_multiple
                # print("reducing lr to ", lr)
            prev_loss = H_error.item()
            # if iter == 0:
            #     print("H_error", H_error, "average_error", average_error)
            if iter < n_iters - 1:
                # print("lr", lr)
                with torch.no_grad():
                    for j in range(k_magnitude_codebook):
                        # print(codebooks_use[i].grad)
                        # codebooks_use[i] -= lr * codebooks_use[i].grad
                        codebooks_use[j] -= torch.clip(lr * codebooks_use[j].grad, -clamp_gradients, clamp_gradients)
                        # codebooks_use[i] = codebooks_use[i].clamp(-1,1)
                    # raise ValueError
                    for j in range(k_magnitude_codebook):
                        codebooks_use[j].grad = None

        
        print("H_error", H_error, "average_error", average_error)
        if isinstance(self.layer, transformers.Conv1D):
            weights_reconstructued = weights_reconstructued.t()
        self.layer.weight.data = weights_reconstructued.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # raise ValueError("stop")

        # raise ValueError

        # Losses = torch.zeros(self.rows, device=self.dev)

        # damp = percdamp * torch.mean(torch.diag(H))
        # diag = torch.arange(self.columns, device=self.dev)
        # H[diag, diag] += damp
        # H = torch.linalg.cholesky(H)
        # H = torch.cholesky_inverse(H)
        # H = torch.linalg.cholesky(H, upper=True)
        # Hinv = H

        # mask = None

        # for i1 in range(0, self.columns, blocksize):
        #     i2 = min(i1 + blocksize, self.columns)
        #     count = i2 - i1

        #     W1 = W[:, i1:i2].clone()
        #     Q1 = torch.zeros_like(W1)
        #     Err1 = torch.zeros_like(W1)
        #     Losses1 = torch.zeros_like(W1)
        #     Hinv1 = Hinv[i1:i2, i1:i2]

        #     if prunen == 0: 
        #         if mask is not None:
        #             mask1 = mask[:, i1:i2]
        #         else:
        #             tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
        #             thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
        #             mask1 = tmp <= thresh
        #     else:
        #         mask1 = torch.zeros_like(W1) == 1

        #     for i in range(count):
        #         w = W1[:, i]
        #         d = Hinv1[i, i]

        #         if prunen != 0 and i % prunem == 0:
        #             tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
        #             mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

        #         q = w.clone()
        #         q[mask1[:, i]] = 0

        #         if hasattr(self, 'quantizer'):
        #             q = quantize(
        #                 q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
        #             ).flatten()

        #         Q1[:, i] = q
        #         Losses1[:, i] = (w - q) ** 2 / d ** 2

        #         err1 = (w - q) / d
        #         W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        #         Err1[:, i] = err1

        #     W[:, i1:i2] = Q1
        #     Losses += torch.sum(Losses1, 1) / 2

        #     W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        #     if DEBUG:
        #         self.layer.weight.data[:, :i2] = W[:, :i2]
        #         self.layer.weight.data[:, i2:] = W[:, i2:]
        #         print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        #         print(torch.sum(Losses))

        # torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        # print('error', torch.sum(Losses).item())

        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()
        # self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
        #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
