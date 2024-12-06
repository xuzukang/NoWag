import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import sys
from typing import Tuple, Optional, Union, List

if __name__ == "__main__":
    sys.path.append(os.getcwd())


from src.quantizers.quantizer_parent import QuantizerParent
import src.alignment.hessian_general_align as hessian_general_align
import src.linear_compress as lc
import src.utils.quantizer as quantizer_utils

import itertools
import opt_einsum as oe
from sympy import factorint
import scipy as sp

def get_qudit_dimensions(d, N_qudits, fixed_qudit_dims:List[int] = []):

    #we assume that d is some power of 2 times 

    N_qudits_left = N_qudits - len(fixed_qudit_dims)
    assert d % np.prod(fixed_qudit_dims) == 0
    factors_raw = factorint(d//int(np.prod(fixed_qudit_dims)))
    factors = []
    for k, v in factors_raw.items():
        factors += [k]*v
    factors = sorted(factors, reverse=True)
    # print(factors)
    dimensions = [1]*N_qudits_left
    i1= 0
    i2 = len(factors)-1
    
    i = 0
    while i1 <= i2:
        if i % (N_qudits*2) < N_qudits_left:
            dimensions[i % N_qudits_left] *= factors[i]
            i1 += 1
        else:
            dimensions[i % N_qudits_left] *= factors[i]
            i2 -= 1
        i += 1
    # print(dimensions)
    dimensions = fixed_qudit_dims + dimensions
    assert np.prod(dimensions) == d
    print(dimensions)
    return dimensions



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
                                          
def initialize_gates(N,qubit_dimensions, layer_type, k_factor, W, device):
    gates = []
    i = len(qubit_dimensions)
    for (dim2,dim1, ) in itertools.combinations(range(-1, -N-1, -1), 2):
        if dim1 == -N and dim2 == -1 and layer_type == "compress":
            new_gate = torch.randn(qubit_dimensions[dim1], qubit_dimensions[dim2], k_factor, qubit_dimensions[dim2]).to(device) * \
                torch.mean(torch.abs(W))**(1/N)/(qubit_dimensions[dim1]*qubit_dimensions[dim2]*k_factor*qubit_dimensions[dim2])**0.25
        elif dim1 == -N and dim2 == -N + 1 and layer_type == "expand":
            new_gate = torch.randn(k_factor, qubit_dimensions[dim2], qubit_dimensions[dim1], qubit_dimensions[dim1]).to(device) * \
                torch.mean(torch.abs(W))**(1/N)/(qubit_dimensions[dim1]*qubit_dimensions[dim2]*k_factor*qubit_dimensions[dim1])**0.25
        
        else:
            new_gate = torch.randn(qubit_dimensions[dim1], qubit_dimensions[dim2], qubit_dimensions[dim1], qubit_dimensions[dim2]).to(device) * \
                    torch.mean(torch.abs(W))**(1/N)/np.sqrt(qubit_dimensions[dim1]*qubit_dimensions[dim2])
            # print(torch.mean(torch.abs(W))**(1/N)/np.sqrt(qubit_dimensions[dim1]*qubit_dimensions[dim2]))
        gates.append(new_gate)
        # print(f"gate {i} {new_gate.shape}, {dim1}, {dim2}")
        i -=1
    return gates

class LinearTensorized(lc.LinearQuantized):
    tensorized:bool = False
    safe_forward:bool = True #if we do safe forward, then we will reconstruct the weight matrix before applying the forward pass


    def tensor_decompose(self, N_qudits:int,
                         fixed_qudits_shapes:List[int]=[],
                         norm_order: List[int] = [0, 1]):
        """_summary_

        Args:
            N_qudits (int): the number of qudits to decompose the tensor into
            qudit_shapes (Optional[List[int]], optional): the shape of the qudits. Defaults to None.
            norm_order (List[int], optional): the order of the norms to use for normalization. Defaults to [0, 1].
        """
        self.N_qudits = N_qudits
        self.N_gates = int(sp.special.comb(N_qudits, 2))
        norm_0, norm_1, weight_use = quantizer_utils.normalize(self.original_weight, norm_order)

        self.add_additional_attributes({"norm_0": norm_0, "norm_1": norm_1})

        if self.in_features > self.out_features:

            self.qudit_shapes = get_qudit_dimensions(self.out_features, N_qudits, fixed_qudits_shapes)
            self.k_factor = int(self.in_features/np.prod(self.qudit_shapes[1:]))
            assert self.in_features == np.prod(self.qudit_shapes[1:])*self.k_factor
            self.layer_type = "compress"

        elif self.in_features < self.out_features:
            self.qudit_shapes = get_qudit_dimensions(self.in_features, N_qudits, fixed_qudits_shapes)
            self.k_factor = int(self.out_features/np.prod(self.qudit_shapes[1:]))
            assert self.out_features == np.prod(self.qudit_shapes[1:])*self.k_factor
            self.layer_type = "expand"

        else:

            self.qudit_shapes = get_qudit_dimensions(self.in_features, N_qudits, fixed_qudits_shapes)
            self.layer_type = "square"
            self.k_factor = 1

        self.N_qudits = N_qudits
        self.gates = nn.ParameterList(
            [nn.Parameter(gate) for gate in 
            initialize_gates(N_qudits, self.qudit_shapes, self.layer_type, self.k_factor, weight_use, self.original_weight.device)]
        )
        

        self.reconstruct_expr = quanta_op_einsum_expr(N_qudits)
        self.apply_expr = quanta_apply_einsum_expr(N_qudits)
        self.tensorized = True

    def reconstruct(self)->torch.FloatTensor:

        if not self.tensorized:
            return self.original_weight
        
        #otherwise, use the reconstruct
        reconstructed_weight = torch.einsum(self.reconstruct_expr, *self.gates).reshape(self.out_features, self.in_features)

        if self.norm_0 is not None:
            # print(self.norm_0.requires_grad)
            reconstructed_weight = reconstructed_weight * self.norm_0.unsqueeze(0)
        if self.norm_1 is not None:
            # print(self.norm_1)
            reconstructed_weight = reconstructed_weight * self.norm_1.unsqueeze(1)

        return reconstructed_weight
    
    def align(
        self,
        val_hessian: Optional[torch.FloatTensor] = None,
        lr: float = 1e-3,
        lr_multiplier: float = 1,  # decay the lr by this factor every time the val loss increases
        n_iters: int = 100,
        val_every: int = 1,
        discrete_update_every: int = 1,
        clip_grad: float = -1,
        verbose: Union[bool, int] = 10,
        low_bound: float = 1e-5,
        patience: int = 10,
        patience_scheduler: int = 2,
        eps: float = 1e-5,
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

        hessian_general_align.align(
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
            raise NotImplementedError("The not safe, potentially faster forward is not implemented yet")

    def blank_recreate(self, **kwargs):
        """Alias for tensor_decompose since that also just initializes them to random values"""
        self.tensor_decompose(**kwargs)



    def get_n_bits(self):
        """_summary_"""
        n_bits = 0
        for gate in self.gates:
            n_bits += gate.numel()*16
        
        if self.norm_0 is not None:
            n_bits += self.norm_0.numel()*16
        if self.norm_1 is not None:
            n_bits += self.norm_1.numel()*16
        if self.original_bias is not None:
            n_bits += self.original_bias.numel()*16

        return n_bits


if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)

    device = torch.device("cuda:1")
    data = torch.load("test/weights_hessian.pt")
    W = data["weights"].to(device)
    hessian = data["hessian"].to(device)
    print(W.shape)

    linear = LinearTensorized(W, bias=None)
    linear.hessian = hessian/ hessian.shape[0]

    linear.tensor_decompose(3, fixed_qudits_shapes=[64])
    linear.set_additional_attributes_as_trainable()
    print(linear.get_n_bits()/16/linear.get_n_original_parameters()*100)

    linear.align(
        None,
        lr=1e-2,
        n_iters=2500,
        lr_multiplier=1/3,
        clip_grad=1e-1,
        verbose=250,
        patience_scheduler=100,
        patience=-1
    )



        
                
