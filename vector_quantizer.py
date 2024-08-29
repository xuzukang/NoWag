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

def assigment_step(W, H, quantized_vectors, debug = [])->torch.tensor:
    """_summary_

    Args:
        W (torch.tensor): weights of shape (n,n)
        H (torch.tensor): matrix of shape (n,n)
        quantized_vectors (torch.tensor): quantized vectors of shape (n,k)
            where k is the number of quantized vectors
    
    Returns:
        torch.tensor: updated assignments
        torch.tensor: updated errors
    """

    #create a tensor of shape (n,k,n)
    #where the slice [i,k,:] consists of 
    # W[i] - quantized_vectors[k]   
    #minus the k quantized vectors
    # print(W.dtype)
    # print(torch.max(H))
    # print(quantized_vectors.shape)
    assert torch.all(torch.isfinite(W)), f"W is not finite, {W}, {W[~torch.isfinite(W)]}"
    assert torch.all(torch.isfinite(quantized_vectors)), f"quantized_vectors is not finite, {quantized_vectors}, {quantized_vectors[~torch.isfinite(quantized_vectors)]}"
    diff = W.unsqueeze(1) - quantized_vectors.T.unsqueeze(0)
    # for i,v in enumerate(debug):
    #     assert torch.allclose(diff[v,i], torch.zeros_like(diff[v,i])), f"diff[{v},{i}] is not zero, {diff[v,i]}, W[{v},:] {W[v]}, quantized_vectors[:,{i}] {quantized_vectors[:,i]}, debug {debug}"
    # print(diff.shape)
    #calculate the error for each quantized vector
    #which is given by delta_x^T * H * delta_x
    #delta_x is given by diff[i,k,:]
    # print(H.dtype)
    # assert torch.all(torch.isfinite(diff)), f"diff is not finite, {diff}, {diff[~torch.isfinite(diff)]}"
    errors = torch.einsum('ijk,kl,ijl->ij', diff, H, diff)
    # for i,v in enumerate(debug):
    #     assert torch.allclose(errors[v,i], torch.zeros_like(errors[v,i])), f"errors[{v},{i}] is not zero, {errors[v,i]}"

    # assert torch.all(torch.isfinite(errors)), f"errors are not finite, {errors}, {errors[~torch.isfinite(errors)]}, n_inf {torch.sum(errors == float('inf'))}"
    assignments = torch.argmin(errors, dim=1)
    # assert(torch.unique(assignments).shape[0] == quantized_vectors.shape[1]), f"not all clusters have an assignment, {torch.unique(assignments).shape}"
    # raise ValueError("stop")
    errors = torch.sum(errors[torch.arange(errors.shape[0]), assignments])
    return assignments, errors

@torch.jit.script
def update_step(W:torch.Tensor, prev_quantized:torch.Tensor, assignments:torch.Tensor)->torch.Tensor:
    """_summary_

    Args:
        W (torch.tensor): weights of shape (n,n)
        assignments (torch.tensor): assignments of shape (n,)
    
    Returns:
        quantized_vectors (torch.tensor): quantized vectors of shape (n,k)
    """

    #initialize the updated quantized vectors
    updated_quantized_vectors = torch.zeros_like(prev_quantized)

    #the quantized vectors are just the mean of the weights
    #that are assigned to the same cluster
    for i in range(updated_quantized_vectors.shape[0]):
        # assert torch.all(torch.isfinite(W[assignments == i])), f"W[assigments == i] is not finite, {W[assignments == i]}, {W[assignments == i][~torch.isfinite(W[assignments == i])]}"
        if torch.any(assignments == i):
            updated_quantized_vectors[:,i] = W[assignments == i].mean(dim=0)
    
    return updated_quantized_vectors





def vector_quantize(W, H, k, max_iters = 1000, 
                    max_init_iters = 10,
                    convergence_threshold = 1e-3):
    """_summary_

    Args:
        W (torch.tensor): weights of shape (n,n)
        H (torch.tensor): matrix of shape (n,n)
        k (int): number of quantized vectors
    
    Returns:
        torch.tensor: quantized vectors of shape (n,k)
    """
    assert torch.all(torch.isfinite(H)), f"H is not finite, {H}, {H[~torch.isfinite(H)]}"
    min_error = float('inf')
    bar = tqdm.tqdm(total=max_init_iters*max_iters)
    for i in range(max_init_iters):
        #initialize the quantized vectors
        indexs = torch.randperm(W.shape[1])[:k]
        quantized_vectors = W[indexs,:].T
        converged = False
        for i in range(max_iters):
            bar.update(1)
            assignments,error = assigment_step(W, H, quantized_vectors, indexs)
            # print("error", error)
            updated_quantized_vectors = update_step(W, quantized_vectors, assignments)
            # print(updated_quantized_vectors.shape)
            if torch.allclose(quantized_vectors, updated_quantized_vectors, atol=convergence_threshold):
                tqdm.tqdm.write(f"Converged after {i} iterations, error {error}")
                converged = True
                bar.update(max_iters - i - 1)
                break
            quantized_vectors = updated_quantized_vectors
        if not converged:
            print("warning: did not converge")
        if error < min_error:
            min_error = error
            best_quantized_vectors = quantized_vectors
            best_assignments = assignments
    print("quantized with best error", min_error)
    return best_quantized_vectors, best_assignments
                    
    


class VectorQuantizerLayer(nn.Module):
    def __init__(self, 
                 original_layer: nn.Linear,
                 n_quantize : int = 128,
                 nsamples : int = 128):
        

        super(VectorQuantizerLayer, self).__init__()

        self.original_weights = original_layer.weight.data
        # print("original weights", self.original_weights.shape,"dtype", self.original_weights.dtype, "device", self.original_weights.device)
        self.b = original_layer.bias
        self.H = torch.zeros((self.original_weights.shape[1], self.original_weights.shape[1]), 
                             device=self.original_weights.device).float() #will cast to half later
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
        assert torch.all(torch.isfinite(self.H)), f"H is not finite, {self.H}, {self.H[~torch.isfinite(self.H)]}"
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
            self.add_batch(x)
            x = F.linear(x, self.original_weights, self.b)
            return x
        
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
    


class SparseGPT:

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

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
