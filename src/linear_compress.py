import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
from typing import Tuple, Optional, Union, List, Literal
from src.quantizers.quantizer_parent import QuantizerParent
import src.utils.compress_parent as compress_parent
import src.utils.sparse as sparse
import src.alignment.hessian_general_align as hessian_general_align
import src.quantizers.vector_quantizer as vector_quantizer
import src.utils.quantizer as quantizer_utils


class LinearQuantized(compress_parent.CompressorParent):
    """A class to pass a linear layer in and get a quantized version of it
    Also servers as our parent class for the quantized linear layers
    """

    def __init__(
        self,
        weight: torch.FloatTensor,
        bias: Optional[torch.FloatTensor] = None,
        add_bias: bool = False,
    ):
        """quantized linear layer

        Args:
            weight (torch.FloatTensor): the original weight matrix of shape (out_features, in_features)
            bias (Optional[torch.FloatTensor], optional): the original bias vector of shape (out_features) or None.
            add_bias (bool, optional): should we add a bias to the layer or not. Defaults to False.
        """

        super(LinearQuantized, self).__init__()
        self.out_features, self.in_features = weight.shape
        self.original_weight = weight
        # print("original weight", self.original_weight[0])
        # self.register_buffer("original_weight", weight)
        self.original_parameters = self.in_features * self.out_features

        if bias is not None:
            self.original_bias = nn.Parameter(bias, requires_grad=True)
            self.original_parameters += self.out_features

        else:
            if add_bias:
                self.original_bias = nn.Parameter(
                    torch.zeros(self.out_features), requires_grad=True
                )
            else:
                self.original_bias = None

        self.quantized = False
        self.quantizer: QuantizerParent = None
        self.log_hessian_flag = False
        self.update_importance_flag = False

    def enable_hessian_logging(self):
        """enable hessian logging"""
        self.log_hessian_flag = True
        self.hessian = torch.zeros(
            self.in_features,
            self.in_features,
            device=self.original_weight.device,
            dtype=torch.float32,
        )
        self.n_samples = 0

    def dump_hessian(self) -> List[torch.FloatTensor]:
        """gives the hessian calculated and stops logging the inputs for the hessian

        Returns:
            torch.FloatTensor: the hessian
        """
        hessian = self.hessian.clone()
        self.log_hessian_flag = False
        del self.hessian
        del self.n_samples
        return [hessian]  # returning a list for consistency with the low rank sparse

    def log_to_hessian_(self, x: torch.FloatTensor):
        """log to the hessian

        Args:
            x (torch.FloatTensor): x is of shape (..., in_features)
        """
        x_flattened = x.reshape(-1, self.in_features).to(torch.float32)
        n_new_samples = x_flattened.shape[0]
        # print(n_new_samples)
        # multiply the hessian by:
        # print(self.n_samples)
        self.hessian *= self.n_samples / (self.n_samples + n_new_samples)
        # outer product of the flattened x
        # first scale x_flattened
        self.n_samples += n_new_samples
        x_flattened = x_flattened * math.sqrt(2 / (self.n_samples))
        self.hessian += x_flattened.T @ x_flattened

    def enable_importance_updates(self, decay: float = 0.99):
        """enable the updates of the importances for the quantizer"""
        # check that the quantizer has importances
        if hasattr(self.quantizer, "ema_update_importances"):
            self.update_importance_flag = True
            self.decay = decay
        else:
            print(
                "The quantizer does not have the method update_importances so ignoring the request"
            )

    def update_importances(self, x: torch.FloatTensor, decay: float = 0.99):
        """Exponential moving update of the importances,
        the importances are just the norms of x along the all but the last dimension

        Args:
            x (torch.FloatTensor): the input to the layer of shape (..., in_features)
            decay (float, optional): the decay factor for the exponential moving average. Defaults to 0.99.
        """
        self.quantizer: vector_quantizer.VectorQuantizer
        # print(x.shape)
        x_reduced = x.reshape(-1, self.in_features).to(torch.float32)
        # print(x_reduced.shape)
        hessian_diag = torch.norm(x_reduced, p=2, dim=0) ** 2 * 2 / self.in_features
        # print(importances_non_expanded.shape)
        # shape of n_inputs
        # print(importances_non_expanded.unsqueeze(0).expand(self.out_features,
        #                                                                                    -1
        #         ).shape)
        self.quantizer.ema_update_importances(hessian_diag, decay)

    def forward(self, x: torch.FloatTensor):
        """forward pass of the linear layer"""
        if self.log_hessian_flag:
            # print("logging to hessian")
            self.log_to_hessian_(x)
        if self.update_importance_flag:
            # print("updating importances")
            self.update_importances(x, self.decay)

        # W = self.reconstruct()
        return F.linear(x, self.reconstruct(), self.original_bias)

    def compress(self, **kwargs):
        """compress the weights"""
        self.quantize(**kwargs)

    def quantize(self, quantizer_class: QuantizerParent, **kwargs):
        """quantize the weights"""
        # print(kwargs)
        self.quantizer = quantizer_class.quantize(
            self.original_weight, self.hessian, **kwargs
        )
        self.quantized = True

    def reconstruct(self) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the quantized version"""
        if hasattr(self, "cached_reconstruct"):
            # print("returning cached")
            return self.cached_reconstruct
        if self.quantized:
            return self.quantizer()
        else:
            return self.original_weight

    def cache_reconstruct(self):
        # print("caching")
        self.register_buffer("cached_reconstruct", self.reconstruct())
        
    def delete_cache_reconstruct(self):
        del self.cached_reconstruct
        
    def get_reconstruction_error(self) -> torch.FloatTensor:
        """returns the reconstruction error"""
        with torch.no_grad():
            return hessian_general_align.loss(self.reconstruct(), self.original_weight, self.hessian)

    def align(
        self,
        val_hessian: Optional[torch.FloatTensor] = None,
        lr: float = 1e-3,
        lr_multiplier: float = 1,  # decay the lr by this factor every time the val loss increases
        n_iters: int = 100,
        val_every: int = 1,
        discrete_update_every: int = 1,
        reinitialize_optimizer: bool = True,
        clip_grad: float = -1,
        verbose: Union[bool, int] = 10,
        low_bound: float = 1e-5,
        patience: int = 10,
        patience_scheduler: int = 2,
        eps: float = 1e-5,
        discrete_update_kwargs: Optional[dict] = {},
        **kwargs,
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
        print(reinitialize_optimizer)
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
            reinitialize_optimizer=reinitialize_optimizer,
            clip_grad=clip_grad,
            verbose=verbose,
            low_bound=low_bound,
            patience=patience,
            patience_scheduler=patience_scheduler,
            eps=eps,
            discrete_update_kwargs=discrete_update_kwargs,
            **kwargs,
        )
        return best_loss

    def update_discrete(self, **kwargs):
        """updates the discrete values of the quantizer"""
        # def recursive_discrete_update(module):
        #     for children in module.children():
        #         if hasattr(children, 'update_discrete'):
        #             if isinstance(getattr(children, 'update_discrete'),callable):
        #                 children.update_discrete()
        #         else:
        #             recursive_discrete_update(children)

        # recursive_discrete_update(self)
        # print("kwargs", kwargs)
        if self.quantized:
            return self.quantizer.update_discrete(**kwargs)

    def clean(self):
        if self.quantized:
            self.quantizer.clean()
        del self.original_weight
        if self.log_hessian_flag:
            del self.hessian
            del self.n_samples
            self.log_hessian_flag = False

    def get_n_bits(self):
        if self.quantized:
            return self.quantizer.get_n_bits()
        else:
            return self.original_parameters * 16

    def get_n_original_parameters(self):
        return self.original_parameters

    def set_additional_attributes_as_trainable(self):
        super(LinearQuantized, self).set_additional_attributes_as_trainable()
        for children in self.children():
            if hasattr(children, "set_additional_attributes_as_trainable"):
                if callable(
                    getattr(children, "set_additional_attributes_as_trainable")
                ):
                    children.set_additional_attributes_as_trainable()

    def set_additional_attributes_as_non_trainable(self):
        super(LinearQuantized, self).set_additional_attributes_as_non_trainable()
        for children in self.children():
            if hasattr(children, "set_additional_attributes_as_non_trainable"):
                if callable(
                    getattr(children, "set_additional_attributes_as_non_trainable")
                ):
                    children.set_additional_attributes_as_non_trainable()

    def get_additional_attributes(self):
        return self.quantizer.get_additional_attributes()

    def blank_recreate(self, quantizer_class: QuantizerParent, **kwargs):
        """recreate the quantizer without any weights"""
        self.quantizer = quantizer_class.blank_recreate(self.original_weight, **kwargs)
        self.quantized = True

    # def load_state_dict(self, state_dict, strict = True, assign = False):
    #     return super().load_state_dict(state_dict, strict, assign)


class LinearQuantizedSparse(LinearQuantized):
    def __init__(
        self,
        weight: torch.FloatTensor,
        bias: Optional[torch.FloatTensor] = None,
        add_bias: bool = False,
    ):
        """A low rank sparse linear layer

        Args:
            weight (torch.FloatTensor): the original weight matrix of shape (out_features, in_features)
            bias (Optional[torch.FloatTensor], optional): the original bias vector of shape (out_features) or None.
            add_bias (bool, optional): should we add a bias to the layer or not. Defaults to False.
        """
        super(LinearQuantizedSparse, self).__init__(weight, bias, add_bias)
        self.sparsed = False

    def initalize_sparse(self,
                 sparse_types:List[Literal["hessian","dim_0","dim_1","unstructed","wanda"]],
                 frac_sparse:Union[float,List[float]] = 0.1,
                    **kwargs):
        """create a sparse compensator

        Args:
            sparse_type (List[Literal[&quot;hessian&quot;,&quot;dim_0&quot;,&quot;dim_1&quot;,&quot;unstructed&quot;]]): the types of sparsity to use
            frac_sparse (Union[float,List[float]], optional): how much sparsity to use, if a list is given, its 
            length should be the same as sparse_type, and it will be the fraction of sparsity for each type. If float, then we equally split the sparsity among the types. Defaults to 0.1.
        """
        
        if isinstance(frac_sparse,float):
            frac_sparse = [frac_sparse/len(sparse_types)]*len(sparse_types) if len(sparse_types) > 0 else []
            
        sparse_modules = []
        for i,sparse_type in enumerate(sparse_types):
            
            if frac_sparse[0] <= 0.0:
                new_sparse_module = None
            if sparse_type == "hessian":
                new_sparse_module = sparse.Dim0_StructuredSparse(self.out_features, self.in_features, frac_sparse[i], self.original_weight.device)
            elif sparse_type == "dim_0":
                new_sparse_module = sparse.Dim0_StructuredSparse(self.out_features, self.in_features, frac_sparse[i], self.original_weight.device)
            elif sparse_type == "dim_1":
                new_sparse_module = sparse.Dim1_StructuredSparse(self.out_features, self.in_features, frac_sparse[i], self.original_weight.device)
            elif sparse_type == "wanda":
                new_sparse_module = sparse.UnstructuredSparse(self.out_features, self.in_features, frac_sparse[i], self.original_weight.device,
                                                                pattern = kwargs.get("pattern", None))
            else:
                raise NotImplementedError
            
            sparse_modules.append(new_sparse_module)
        
        self.sparse_modules = nn.ModuleList(sparse_modules)
        self.sparsed = True
        self.sparse_after_norm = False
        
        
    def sparsify(self,
                 sparse_types:List[Literal["hessian","dim_0","dim_1","unstructed"]],
                 frac_sparse:Union[float,List[float]] = 0.1,
                 remaining_error:Optional[torch.FloatTensor] = None,
                 sparse_after_norm:bool = False,
                    **kwargs):
        """create a sparse compensator

        Args:
            sparse_type (List[Literal[&quot;hessian&quot;,&quot;dim_0&quot;,&quot;dim_1&quot;,&quot;unstructed&quot;]]): the types of sparsity to use
            frac_sparse (Union[float,List[float]], optional): how much sparsity to use, if a list is given, its 
            length should be the same as sparse_type, and it will be the fraction of sparsity for each type. If float, then we equally split the sparsity among the types. Defaults to 0.1.
        """
        with torch.no_grad():
            
            if remaining_error is None:
                if sparse_after_norm:
                    remaining_error = self.reconstruct(denormalize = False) - self.quantizer.normalizer.normalize(self.original_weight)
                else:
                    remaining_error = self.reconstruct() - self.original_weight
            
            self.initalize_sparse(sparse_types, frac_sparse, **kwargs)
                
                
            for i,sparse_type in enumerate(sparse_types):
                if self.sparse_modules[i] is None:
                    print("skipping because of 0 sparsity")
                    continue
                if sparse_type == "hessian":
                    self.sparse_modules[i].update_sparse_hessian_importance(remaining_error, torch.norm(self.hessian, dim = 1))
                elif sparse_type == "dim_0":
                    self.sparse_modules[i].update_sparse_norm(remaining_error)
                elif sparse_type == "dim_1":
                    self.sparse_modules[i].update_sparse_norm(remaining_error)
                elif sparse_type == "wanda":
                    self.sparse_modules[i].update_wanda_like(remaining_error, torch.diag(self.hessian))
                else:
                    raise NotImplementedError
                
                # print(self.sparse_modules[i].reconstruct()[:,self.sparse_modules[i].sparse_mask])
                
                # print(remaining_error[:,self.sparse_modules[i].sparse_mask])
                remaining_error = remaining_error + self.sparse_modules[i].reconstruct()
                # print(remaining_error[:,self.sparse_modules[i].sparse_mask])
            
            self.sparse_after_norm = sparse_after_norm

    def sparse_only(self, sparse_types:List[Literal["hessian","dim_0","dim_1","wanda"]],
            frac_sparse:Union[float,List[float]],
            norm_order:List[int] = [0,1],
            zero:List[bool] = [True,True],
            sparse_after_norm:bool = False,
            **kwargs):
        
        if sparse_after_norm:
            normalizer,normalized_weight = quantizer_utils.Normalizer.normalize_init(self.original_weight,
                                                                   norm_order, zero)
            # print("here")
            self.sparsify(sparse_types, frac_sparse,
                      -normalized_weight,
                      sparse_after_norm=sparse_after_norm,
                      **kwargs)
        else:

            self.sparsify(sparse_types, frac_sparse,
                        -self.original_weight,
                        sparse_after_norm=sparse_after_norm)
            normalizer = None
        
        self.normalizer = normalizer
        

                
    def sparse_before_quantize(self,
             sparse_types:List[Literal["hessian","dim_0","dim_1","unstructed","wanda"]],
            frac_sparse:Union[float,List[float]],
            quantizer_class: QuantizerParent, 
            quantizer_kwargs:dict,
            quantize_minus_sparse:bool = True,
            sparse_after_norm:bool = False,
    ):
        """initialize and pick the sparse values BEFORE quantizing"""
        
        if sparse_after_norm:
            normalizer,normalized_weight = quantizer_utils.Normalizer.normalize_init(self.original_weight,
                                                                   quantizer_kwargs.get("norm_order",[0,1]),
                                                                   quantizer_kwargs.get("zero",[True,True]))
            # print("here")
            self.sparsify(sparse_types, frac_sparse,
                      -normalized_weight,
                      sparse_after_norm=sparse_after_norm)
        else:

            self.sparsify(sparse_types, frac_sparse,
                        -self.original_weight,
                        sparse_after_norm=sparse_after_norm)
            normalizer = None
        
            
        
        #calculate the overall mask
        mask_overall = torch.zeros((self.out_features, self.in_features), dtype = torch.bool, device = self.original_weight.device)
        for sparse_module in self.sparse_modules:
            if isinstance(sparse_module, sparse.Dim0_StructuredSparse):
                mask_overall = mask_overall | sparse_module.sparse_mask.unsqueeze(0)
            elif isinstance(sparse_module, sparse.Dim1_StructuredSparse):
                mask_overall = mask_overall | sparse_module.sparse_mask.unsqueeze(1)
            elif isinstance(sparse_module, sparse.UnstructuredSparse):
                mask_overall = mask_overall | sparse_module.sparse_mask
        
        #add to the quantizer kwargs
        quantizer_kwargs["mask"] = ~mask_overall

        if quantize_minus_sparse:
            sparse_sum = torch.zeros_like(self.original_weight)
            for sparse_module in self.sparse_modules:
                sparse_sum += sparse_module.reconstruct()
            weight_to_quantize = self.original_weight - sparse_sum if not sparse_after_norm else self.original_weight - normalizer.denormalize(sparse_sum)
        else:
            weight_to_quantize = self.original_weight
            
        self.quantize(quantizer_class, 
                      weight_to_quantize,
                      normalizer = normalizer,
                      **quantizer_kwargs)

        #reupdate the sparse values
        if sparse_after_norm:
            remaining_error = self.reconstruct(ignore_sparse=True, denormalize = False) - normalized_weight
        else:
            remaining_error = self.reconstruct(ignore_sparse = True) - self.original_weight
        for i,sparse_module in enumerate(self.sparse_modules):
            if sparse_module is None:
                continue
            self.sparse_modules[i].update_fixed_mask(remaining_error)
        
    def quantize(self, quantizer_class: QuantizerParent, weight_to_quantize:torch.FloatTensor = None, **kwargs):
        """quantize the weights"""
        # print(kwargs)
        self.quantizer = quantizer_class.quantize(
            self.original_weight if weight_to_quantize is None else weight_to_quantize
            , self.hessian, **kwargs
        )
        self.quantized = True

        
    def forward(self, x: torch.FloatTensor):
        if hasattr(self, "cached_reconstruct"):
            y = F.linear(x, self.cached_reconstruct, self.original_bias)
        else:
            y = F.linear(x, super().reconstruct(), self.original_bias)
            if not hasattr(self, "cached_reconstruct"):
                for sparse_module in self.sparse_modules:
                    if sparse_module is not None:
                        y = y + sparse_module(x)
        return y
    
    def get_n_bits(self):
        n_bits = super().get_n_bits()
        if self.sparsed:
            for sparse_module in self.sparse_modules:
                n_bits += sparse_module.get_n_bits()
        return n_bits
    
    def blank_recreate(self, quantizer_class, quantizer_kwargs,
                       sparse_kwargs = {}):
        super().blank_recreate(quantizer_class, **quantizer_kwargs)
        
        if len(sparse_kwargs) > 0:
            self.initalize_sparse(sparse_kwargs["sparse_types"], sparse_kwargs["frac_sparse"])
        
        self.sparse_after_norm = sparse_kwargs.get("sparse_after_norm", False)
            
        
    def reconstruct(self, ignore_sparse = False, **kwargs) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the quantized version"""
        # print("reconstructing")
        if hasattr(self, "cached_reconstruct"):
            # print("returning cached")
            return self.cached_reconstruct
        if self.quantized:
            actual_denorm = kwargs.get("denormalize", True)
            kwargs["denormalize"] = kwargs.get("denormalize", True) and not self.sparse_after_norm
            reconstructed = self.quantizer(**kwargs)
            if self.sparsed and not ignore_sparse:
                for sparse_module in self.sparse_modules:
                    reconstructed += sparse_module.reconstruct()
            if self.sparse_after_norm and actual_denorm:
                reconstructed = self.quantizer.normalizer.denormalize(reconstructed)
        elif self.sparsed:
            # print("here")
            reconstructed = self.sparse_modules[0].reconstruct()
            for sparse_module in self.sparse_modules[1:]:
                if sparse_module is not None:
                    reconstructed += sparse_module.reconstruct()
            
            if self.sparse_after_norm:
                # print("skipping")
                reconstructed = self.normalizer.denormalize_inplace(reconstructed)

            # print("here")
            return reconstructed
        else:
            return self.original_weight