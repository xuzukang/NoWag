import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import sys
from typing import Tuple, Optional, Union, List, Callable, Literal

if __name__ == "__main__":
    sys.path.append(os.getcwd())


from src.quantizers.quantizer_parent import QuantizerParent
import src.alignment.hessian_general_align as hessian_general_align
import src.alignment.weight_align as weight_align
import src.linear_compress as lc
import src.utils.normalizer as quantizer_utils

import itertools
import opt_einsum as oe
from sympy import factorint
import scipy as sp
# import cvxpy as cp

def get_qudit_dimensions(d, N_qudits, fixed_qudit_dims:List[int] = []):

    #we assume that d is some power of 2 times 
    assert len(fixed_qudit_dims) <= N_qudits
    if len(fixed_qudit_dims) == N_qudits:
        assert np.prod(fixed_qudit_dims) == d
        return sorted(fixed_qudit_dims)
    
    N_qudits_left = N_qudits - len(fixed_qudit_dims)
    assert d % np.prod(fixed_qudit_dims) == 0
    factors_raw = factorint(d//int(np.prod(fixed_qudit_dims)))
    factors = []
    for k, v in factors_raw.items():
        factors += [k]*v
    factors = sorted(factors, reverse=True)
    # print("factors", factors)
    dimensions = [1]*N_qudits_left
    for f in factors:
        #send the factor to the qudit with the smallest dimension
        min_dim = min(dimensions)
        min_dim_ind = dimensions.index(min_dim)
        dimensions[min_dim_ind] *= f
    # i1= 0
    # i2 = len(factors)-1
    
    # i = 0
    # while i1 <= i2:
    #     if i % (N_qudits*2) < N_qudits_left:
    #         dimensions[i % N_qudits_left] *= factors[i]
    #         i1 += 1
    #     else:
    #         dimensions[i % N_qudits_left] *= factors[i]
    #         i2 -= 1
    #     i += 1
    # print(dimensions)
    dimensions = sorted(fixed_qudit_dims + dimensions)[::-1]
    assert np.prod(dimensions) == d
    # print(dimensions)
    return dimensions

def n_pad(N, d):
    if N % d == 0:
        return 0
    #find the number of padding needed to make N a multiple of d
    return d - N % d

def quanta_apply_einsum_expr(N):
    current_symbols_inds = list(range(N))
    expr = "..."
    for i in current_symbols_inds:
        expr += oe.get_symbol(i)
    for (dim1, dim2) in itertools.combinations(range(-1, -N-1, -1), 2):
        symbol_ind1 = current_symbols_inds[dim1]
        symbol_ind2 = current_symbols_inds[dim2]
        symbol_ind3 = symbol_ind1 + N
        symbol_ind4 = symbol_ind2 + N
        expr += "," + \
            oe.get_symbol(symbol_ind4) + \
            oe.get_symbol(symbol_ind3) + \
            oe.get_symbol(symbol_ind2) + \
            oe.get_symbol(symbol_ind1)
        current_symbols_inds[dim1] = symbol_ind3
        current_symbols_inds[dim2] = symbol_ind4
    expr += "->..."
    for i in current_symbols_inds:
        expr += oe.get_symbol(i)
    return expr

def quanta_op_einsum_expr(N):
    apply_expression = quanta_apply_einsum_expr(N)
    initial_term = apply_expression.split(",")[0].split("...")[-1]
    final_term = apply_expression.split("->")[-1].split("...")[-1]
    middle_terms = apply_expression[apply_expression.find(",")+1:apply_expression.find("->")]
    return f"{middle_terms}->{final_term}{initial_term}"
                                          
def initialize_gates(N,qubit_dimensions, layer_type, k, W, device):
    gates = []
    i = len(qubit_dimensions)
    for (dim2,dim1, ) in itertools.combinations(range(-1, -N-1, -1), 2):
        if dim1 == -N and dim2 == -1 and layer_type == "compress":
            new_gate = torch.randn(qubit_dimensions[dim1], qubit_dimensions[dim2], k, qubit_dimensions[dim2]).to(device) #* \
                #torch.mean(torch.abs(W))**(1/N)/(qubit_dimensions[dim1]*qubit_dimensions[dim2]*k*qubit_dimensions[dim2])**0.25
        elif dim1 == -N and dim2 == -N + 1 and layer_type == "expand":
            new_gate = torch.randn(k, qubit_dimensions[dim2], qubit_dimensions[dim1], qubit_dimensions[dim2]).to(device) #* \
                #torch.mean(torch.abs(W))**(1/N)/(qubit_dimensions[dim1]*qubit_dimensions[dim2]*k*qubit_dimensions[dim1])**0.25
        
        else:
            new_gate = torch.randn(qubit_dimensions[dim1], qubit_dimensions[dim2], qubit_dimensions[dim1], qubit_dimensions[dim2]).to(device) #* \
                    #torch.mean(torch.abs(W))**(1/N)/np.sqrt(qubit_dimensions[dim1]*qubit_dimensions[dim2])
            # print(torch.mean(torch.abs(W))**(1/N)/np.sqrt(qubit_dimensions[dim1]*qubit_dimensions[dim2]))
        gates.append(new_gate)
        # print(f"gate {i} {new_gate.shape}, {dim1}, {dim2}")
        i -=1
    return gates



class LinearTensorized(lc.LinearQuantized):
    tensorized:bool = False
    safe_forward:bool = True #if we do safe forward, then we will reconstruct the weight matrix before applying the forward pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print("here")

    def compress(self, **kwargs):
        """Alias for tensor_decompose since that also just initializes them to random values"""
        self.tensor_decompose(**kwargs)
        
    def tensor_decompose(self, N_qudits:int,
                         fixed_qudits_shapes:List[int]=[],
                         norm_order: List[int] = [0, 1],
                         zeros:List[bool] = [False, False],
                         pad_method:Literal["square", "pad_smaller", "pad_larger"] = "pad_smaller",
                            **kwargs):
        """_summary_

        Args:
            N_qudits (int): the number of qudits to decompose the tensor into
            qudit_shapes (Optional[List[int]], optional): the shape of the qudits. Defaults to None.
            norm_order (List[int], optional): the order of the norms to use for normalization. Defaults to [0, 1].
        """
        # print("weight shape", self.original_weight.shape)
        self.N_qudits = N_qudits
        self.N_gates = int(sp.special.comb(N_qudits, 2))
        self.normalizer, weight_use = quantizer_utils.Normalizer.normalize_init(self.original_weight, norm_order, zeros)
        self.normalizer: quantizer_utils.Normalizer

        if self.in_features > self.out_features:

            if pad_method == "pad_smaller":
                self.qudit_shapes = get_qudit_dimensions(self.in_features, N_qudits, fixed_qudits_shapes)
                # print("self.out_features", self.out_features, "self.in_features", self.in_features) 
                # print("self.qudit_shapes", self.qudit_shapes)
                self.pad_n = n_pad(self.out_features, np.prod(self.qudit_shapes[1:]))
                # print("paddding", self.pad_n)
                self.k = int((self.out_features+self.pad_n)/np.prod(self.qudit_shapes[1:]))
                self.input_reshape_shape = self.qudit_shapes
                if self.pad_n > 0:
                    self.reshape_fn = lambda x, kwargs={} : x.reshape((self.out_features + self.pad_n, self.in_features), **kwargs)[:-self.pad_n,:]
                self.layer_type = "expand"
            elif pad_method == "pad_larger":
                self.qudit_shapes = get_qudit_dimensions(self.out_features, N_qudits, fixed_qudits_shapes)
                self.pad_n = n_pad(self.in_features, np.prod(self.qudit_shapes[1:]))
                # print("paddding", self.pad_n)
                self.k = int((self.in_features + self.pad_n)/np.prod(self.qudit_shapes[1:]))
                self.input_reshape_shape = [self.k] + self.qudit_shapes[1:]
                if self.pad_n > 0:
                    self.reshape_fn = lambda x,kwargs={} : x.reshape((self.out_features, self.in_features + self.pad_n),**kwargs)[:,:-self.pad_n]
                self.layer_type = "compress"
            else:
                self.qudit_shapes = get_qudit_dimensions(self.in_features, N_qudits, fixed_qudits_shapes)
                self.pad_n = self.in_features - self.out_features
                # print("paddding", self.pad_n)
                self.k = self.qudit_shapes[0]
                self.input_reshape_shape = self.qudit_shapes
                self.reshape_fn = lambda x, kwargs={} : x.reshape((self.out_features + self.pad_n, self.in_features), **kwargs)[:-self.pad_n,:]
                self.layer_type = "expand"
            # # assert self.in_features == np.prod(self.qudit_shapes[1:])*self.k
            # self.layer_type = "compress"



        elif self.in_features < self.out_features:
            if pad_method == "pad_larger":
                self.qudit_shapes = get_qudit_dimensions(self.in_features, N_qudits, fixed_qudits_shapes)
                # print("self.qudit_shapes", self.qudit_shapes)
                self.pad_n = n_pad(self.out_features, np.prod(self.qudit_shapes[1:]))
                # print("paddding", self.pad_n)
                self.k = int((self.out_features+self.pad_n)/np.prod(self.qudit_shapes[1:]))
                self.input_reshape_shape = self.qudit_shapes
                if self.pad_n > 0:
                    self.reshape_fn = lambda x, kwargs={} : x.reshape((self.out_features + self.pad_n, self.in_features), **kwargs)[:-self.pad_n,:]
                self.layer_type = "expand"
            elif pad_method == "pad_smaller":
                self.qudit_shapes = get_qudit_dimensions(self.out_features, N_qudits, fixed_qudits_shapes)
                self.pad_n = n_pad(self.in_features, np.prod(self.qudit_shapes[1:]))
                # print("paddding", self.pad_n)
                self.k = int((self.in_features + self.pad_n)/np.prod(self.qudit_shapes[1:]))
                self.input_reshape_shape = [self.k] + self.qudit_shapes[1:]
                if self.pad_n > 0:
                    self.reshape_fn = lambda x,kwargs={} : x.reshape((self.out_features, self.in_features + self.pad_n),**kwargs)[:,:-self.pad_n]
                self.layer_type = "compress"
            else:
                self.qudit_shapes = get_qudit_dimensions(self.out_features, N_qudits, fixed_qudits_shapes)
                self.pad_n = self.out_features - self.in_features
                print("self.pad_n", self.pad_n)
                print("self.out_features", self.out_features, "self.in_features", self.in_features)
                self.k = self.qudit_shapes[0]
                self.input_reshape_shape = [self.k] + self.qudit_shapes[1:]
                self.reshape_fn = lambda x, kwargs={} : x.reshape((self.out_features, self.in_features + self.pad_n), **kwargs)[:,:-self.pad_n]
                self.layer_type = "compress"
            # print(self.k)
            # assert self.out_features == np.prod(self.qudit_shapes[1:])*self.k

        else:

            self.qudit_shapes = get_qudit_dimensions(self.in_features, N_qudits, fixed_qudits_shapes)
            self.input_reshape_shape = self.qudit_shapes
            # print(self.qudit_shapes)
            self.layer_type = "square"
            self.k = self.qudit_shapes[0]
            self.pad_n = 0

        if not hasattr(self, "reshape_fn"):
            self.reshape_fn = lambda x, kwargs={} : x.reshape((self.out_features, self.in_features), **kwargs)

        self.N_qudits = N_qudits
        self.gates = nn.ParameterList(
            [nn.Parameter(gate) for gate in 
            initialize_gates(N_qudits, self.qudit_shapes, self.layer_type, self.k, weight_use, self.original_weight.device)]
        )
        self.gates[0] = nn.Parameter(torch.zeros_like(self.gates[0]))
        # print([gate.shape for gate in self.gates])
        # raise NotImplementedError("The rest of the tensor decompose is not implemented yet")

        self.reconstruct_expr = quanta_op_einsum_expr(N_qudits)
        self.apply_expr = quanta_apply_einsum_expr(N_qudits)
        # print("apply expr", self.apply_expr)
        self.tensorized = True

    def reconstruct(self)->torch.FloatTensor:

        if not self.tensorized:
            return self.original_weight
        
        #otherwise, use the reconstruct
        # print(self.reconstruct_expr)
        # print([gate.shape for gate in self.gates])
        # print(torch.min(torch.einsum(self.reconstruct_expr, *self.gates)))  
        # print(torch.max(torch.einsum(self.reconstruct_expr, *self.gates)))
        # print(self.reconstruct_expr)
        # print()
        # print(torch.einsum(self.reconstruct_expr, *self.gates).shape)
        reconstructed_weight = self.reshape_fn(torch.einsum(self.reconstruct_expr, *self.gates))
        # print(reconstructed_weight)
        return self.normalizer.denormalize(reconstructed_weight)
    
    def align(
        self,
        val_hessian: Optional[torch.FloatTensor] = None,
        lr: Union[float, dict[str, float]] = 1e-3,
        lr_multiplier: float = 1,  # decay the lr by this factor every time the val loss increases
        lr_warmup: Optional[Union[float, dict[str, float]]] = None,
        n_iters: int = 100,
        n_iters_warmup_task: int = 0,
        val_every: int = 1,
        discrete_update_every: int = -1,
        clip_grad: float = -1,
        verbose: Union[bool, int] = 10,
        low_bound: float = 1e-5,
        patience: int = 10,
        patience_scheduler: int = 2,
        eps: float = 1e-5,
        **kwargs
    ):
        
        """aligns the compression module to the hessian of the training dataset

        Args:
            val_hessian (Optional[torch.FloatTensor], optional): the hessian of the validation dataset, if None, we don't use it. Defaults to None.
            lr (float, optional): the learning rate for the optimizer. Defaults to 1e-3.
            lr_multiplier (float, optional): multiply the learning rate by this factor every time the validation loss increases. Defaults to 1.
            val_every (int, optional): validate the model every this number of iterations on the validation hessian. Defaults to 1.
            discrete_update_every (int, optional): update the discrete variables every this number of iteration. Defaults to 1.
            clip_grad (float, optional): clip the gradient norm to this value. Defaults to -1 in which case we don't clip the gradient.
            verbose (bool, optional): print every this number of iterations. If False, we don't print anything. If True, we print at the end only. Defaults to False.
            low_bound (float, optional): the lower bound for the error, below which we stop training. Defaults to 1e-5.
            patience (int, optional): the patience for the early stop, if the loss has not improved by eps for this number of iterations, we stop training. Defaults to 10.
            patience_scheduler (int, optional): the patience for the learning rate scheduler. Defaults to 2.
            eps (float, optional): the minimum improvement in the loss to consider it as an improvement. Defaults to 1e-5.

        """
        
        if not self.tensorized:
            raise ValueError("The module is not tensorized, so we can't align it")
        
        if n_iters_warmup_task > 0:
            weight_align.align(
                compression_module=self,
                original_weights=self.original_weight,
                lr=lr if lr_warmup is None else lr_warmup,
                lr_multiplier=lr_multiplier,
                n_iters=n_iters_warmup_task,
                val_every=val_every,
                discrete_update_every=discrete_update_every,
                clip_grad=clip_grad,
                verbose=verbose,
                low_bound=low_bound,
                patience=patience,
                patience_scheduler=patience_scheduler,
                eps=eps,
            )
        # print(self.gates[0])
        
        
        if n_iters > 0:
            _, best_loss = hessian_general_align.align(
                compression_module=self,
                original_weights=self.original_weight,
                train_hessian=self.hessian,
                val_hessian=val_hessian,
                lr=lr,
                lr_multiplier=lr_multiplier,
                n_iters=n_iters,
                val_every=val_every,
                discrete_update_every=discrete_update_every,
                clip_grad=clip_grad,
                verbose=verbose,
                low_bound=low_bound,
                patience=patience,
                patience_scheduler=patience_scheduler,
                eps=eps,
            )
            return best_loss
        else:
            return 0



    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.log_hessian_flag:
            # print("logging to hessian")
            self.log_to_hessian_(x)
        #if we are not tensorized, then we just use the original forward pass
        if not self.tensorized:
            return F.linear(x, self.original_weight, self.original_bias)

        if self.safe_forward:
            weight_use = self.reconstruct()
            return F.linear(x, weight_use, self.original_bias)
        
        else:
            #otherwise, we use the tensorized forward pass
            #multiply x by norm_0
            # print("x", x.shape)
            if self.normalizer.norms[0] is not None:
                x = x*self.normalizer.norms[0]
            if self.pad_n > 0 and self.layer_type == "expand":
                x = F.pad(x, (0, self.pad_n))
            # print("padded and normalized x", x.shape)
            x_reshaped = (x).reshape(list(x.shape[:-1]) + self.input_reshape_shape)
            # print("x_reshaped", x_reshaped.shape)
            # print("gates shapes", [gate.shape for gate in self.gates])
            # print("apply expr", self.apply_expr)    
            # print("rec", self.reconstruct_expr)
            # print("reconstructed_shape", self.reconstruct().shape)
            y = torch.einsum(self.apply_expr, x_reshaped, *self.gates)
            y = y.reshape(list(x.shape[:-1]) + [self.out_features])
            if self.pad_n > 0 and self.layer_type == "compress":
                y = y[:,:-self.pad_n]
            if self.normalizer.norms[1] is not None:
                y *= self.normalizer.norms[1]
            # print("y", y.shape)
            return y


    def blank_recreate(self, **kwargs):
        """Alias for tensor_decompose since that also just initializes them to random values"""
        self.tensor_decompose(**kwargs)



    def get_n_bits(self):
        """_summary_"""
        n_bits = 0
        for gate in self.gates:
            n_bits += gate.numel()*16
        
        n_bits += self.normalizer.get_n_bits()

        if self.original_bias is not None:
            n_bits += self.original_bias.numel()*16

        return n_bits
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        out = super().load_state_dict(state_dict, strict, assign)
        # print("self.gates[0][0,0,0]", self.gates[0][0,0,0])
        return out
    
    
def find_threshold(W, zero_out_top:float,eps=1e-6):
    W_flat = torch.sort(W.flatten())[0]
    threshold = W_flat[int((1-zero_out_top)*W_flat.numel())]
    return threshold
    
    # threshold = torch.mean(W)
    # search_range = [0,torch.max(W)]
    # #perform binary search to find the threshold
    # for i in range(100):
    #     mask = W > threshold
    #     n_zeros = torch.sum(mask).item()
    #     if n_zeros > int(zero_out_top*W.numel()) + 1:
    #         search_range[0] = threshold
    #         threshold_ = (search_range[0] + search_range[1])/2
    #     elif n_zeros < int(zero_out_top*W.numel())-1:
    #         search_range[1] = threshold
    #         threshold_ = (search_range[0] + search_range[1])/2
        
    #     if abs(threshold-threshold_) < eps:
    #         break
    #     threshold = threshold_
    # return threshold



class LinearTensorizedWithSparse(LinearTensorized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse = True

    def tensor_decompose(self, N_qudits:int,
                         fixed_qudits_shapes:List[int]=[],
                         norm_order: List[int] = [0, 1],
                         sparse_frac:float = 0.01,
                         sparse_method:Literal["unstructured", "structured_0_only", "structured_1_only", "structured_0_1"] = "unstructured",
                            **kwargs):
        """_summary_

        Args:
            N_qudits (int): the number of qudits to decompose the tensor into
            qudit_shapes (Optional[List[int]], optional): the shape of the qudits. Defaults to None.
            norm_order (List[int], optional): the order of the norms to use for normalization. Defaults to [0, 1].
            sparsity (float, optional): the sparsity of the gates. Defaults to 0.01.
        """
        # print("weight shape", self.original_weight.shape
        super().tensor_decompose(N_qudits, fixed_qudits_shapes, norm_order)
        
        # print("sparsity", sparse_frac)
        self.sparse_method = sparse_method
        if sparse_method == "unstructured":
            #unstructured sparsity by just picking the top k% of the weights
            threshold = find_threshold(torch.abs(self.original_weight), sparse_frac)
        
            mask = torch.abs(self.original_weight) > threshold
            self.register_buffer("mask", mask)
            self.sparse_weights = nn.Parameter(self.original_weight[self.mask])

        elif sparse_method == "structured_0_only":
            #structured sparsity by only zeroing out the first gate
            norms = torch.norm(self.original_weight, dim=0)
            threshold = find_threshold(norms, sparse_frac)
            
            mask = norms > threshold
            self.register_buffer("mask", mask)
        
            self.sparse_weights = nn.Parameter(self.original_weight[: , self.mask])
        elif sparse_method == "structured_1_only":
            #structured sparsity by only zeroing out the second gate
            norms = torch.norm(self.original_weight, dim=1)
            threshold = find_threshold(norms, sparse_frac)
            
            mask = norms > threshold
            self.register_buffer("mask", mask)
        
            self.sparse_weights = nn.Parameter(self.original_weight[self.mask , :])
        
        elif sparse_method == "structured_0_1":
            norms_0 = torch.norm(self.original_weight, dim=0)
            norms_1 = torch.norm(self.original_weight, dim=1)
            threshold = find_threshold(torch.concatenate([norms_0, norms_1]), sparse_frac)

            mask_0 = norms_0 > threshold
            mask_1 = norms_1 > threshold
            self.register_buffer("mask_0", mask_0)
            self.register_buffer("mask_1", mask_1)

            self.sparse_weights0 = nn.Parameter(self.original_weight[:, self.mask_0])
            self.sparse_weights1 = nn.Parameter(self.original_weight[self.mask_1 , :])
        else:
            raise ValueError(f"Unrecognized sparse method {sparse_method}")

        # print(self.sparse_weights.numel()/self.original_weight.numel())
        
    def reconstruct(self)->torch.FloatTensor:
        reconstruct_weight = super().reconstruct()
        if self.sparse_method == "unstructured":
            reconstruct_weight[self.mask] = self.sparse_weights
        elif self.sparse_method == "structured_0_only":
            reconstruct_weight[:, self.mask] = self.sparse_weights
        elif self.sparse_method == "structured_1_only":
            reconstruct_weight[self.mask, : ] = self.sparse_weights
        elif self.sparse_method == "structured_0_1":
            reconstruct_weight[:,self.mask_0] = self.sparse_weights0
            reconstruct_weight[self.mask_1, :] = self.sparse_weights1

        return reconstruct_weight
    
    def get_n_bits(self):
        nbits = super().get_n_bits()
        
        if self.sparse_method == "unstructured":
            nse = torch.sum(self.mask)
            nrows = self.out_features
            n_indicies_bits = torch.ceil(torch.log2(max(nse, nrows)))
            nbits += ((nrows * n_indicies_bits + (n_indicies_bits + 16)*nse)).item()
        elif self.sparse_method == "structured_0_only" or self.sparse_method == "structured_1_only":
            n_indicies_bits = np.ceil(np.log2(len(self.mask)))
            nbits += torch.sum(self.mask).item() * (n_indicies_bits) + self.sparse_weights.numel()*16
        elif self.sparse_method == "structured_0_1":
            n_indicies_bits = np.ceil(np.log2(len(self.mask_0)))
            # print(torch.sum(self.mask_0), n_indicies_bits)
            nbits += torch.sum(self.mask_0).item() * (n_indicies_bits) + self.sparse_weights0.numel()*16
            n_indicies_bits = np.ceil(np.log2(len(self.mask_1)))
            # print(torch.sum(self.mask_1), n_indicies_bits)
            nbits += torch.sum(self.mask_1).item() * (n_indicies_bits) + self.sparse_weights1.numel()*16
        return nbits
        
    def load_state_dict(self, state_dict, strict = True, assign = False):
        if self.sparse_method == "unstructured" or self.sparse_method == "structured_0_only" or self.sparse_method == "structured_1_only":
            self.mask = state_dict["mask"]
            self.sparse_weights = nn.Parameter(state_dict["sparse_weights"])
        elif self.sparse_method == "structured_0_1":
            self.mask_0 = state_dict["mask_0"]
            self.mask_1 = state_dict["mask_1"]
            self.sparse_weights0 = nn.Parameter(state_dict["sparse_weights0"])
            self.sparse_weights1 = nn.Parameter(state_dict["sparse_weights1"])
        super().load_state_dict(state_dict, strict, assign)
    
    def forward(self,x):
        raise NotImplementedError("The forward pass for sparse tensorized layers is not implemented yet")
        
    
            





if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)

    device = torch.device("cuda:1")
    data = torch.load("/data/lliu/huffman/layer_0_self_attn.q_proj.pt")
    # data = torch.load("/data/lliu/huffman/layer_0_mlp.gate_proj.pt")
    W = data["weight"].to(device).to(torch.float32)
    # W = torch.load("test/remaining_W_error.pt").detach().to(device)
    # print(W.shape)
    # W = W[4096:4096*2,:]
     
    hessian = data["hessian"].to(device).to(torch.float32)
    print(W.shape)
    print(hessian)
    # W = torch.randn((11008,4096)).to(device)

    linear = LinearTensorized(W, bias=None)
    linear.hessian = hessian#/ hessian.shape[0]

    linear.tensor_decompose(3, fixed_qudits_shapes=[64], norm_order=[0,1])
    linear.set_additional_attributes_as_trainable()
    print(linear.qudit_shapes)
    print([gate.shape for gate in linear.gates])
    print(
        f"remaining parameter fraction: {round(linear.get_n_bits()/16/linear.get_n_original_parameters()*100, 2)}% bpv: {linear.get_n_bits()/linear.get_n_original_parameters()}")

    assert linear.reconstruct().shape == W.shape
    linear.align(
        lr=1e-2,
        lr_norms=None,
        lr_warmup=1e-1,
        lr_norms_warmup=None,
        n_iters=2500,
        n_iters_warmup_task=0,
        lr_multiplier=1/3,
        clip_grad=1e-1,
        verbose=250,
        patience_scheduler=100,
        patience=-1
    )
    # linear.align(
    #     hessian_align=True,
    #     lr=1e-2,
    #     lr_norms=1e-2,
    #     n_iters=2500,
    #     lr_multiplier=1/3,
    #     clip_grad=1e-1,
    #     verbose=250,
    #     patience_scheduler=100,
    #     patience=-1
    # )



        
                
