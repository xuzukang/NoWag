import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import itertools
import torch.jit as jit
import os
import sys
from typing import Tuple, Optional, Union, List, Callable, Literal

if __name__ == "__main__":
    sys.path.append(os.getcwd())
import src.utils.quantizer as quantizer_utils
import src.utils.utils as utils
import src.quantizers.vector_quantizer as vq_original


class QuickStopException(Exception):
    pass

class VectorQuantizer_1st_order(vq_original.VectorQuantizer):

    def __init__(
        self,
        codes: torch.LongTensor,
        codebook: torch.FloatTensor,
        reconstructed_shape: Union[Tuple[int, int], torch.Size],
        normalizer: quantizer_utils.Normalizer,
        reference_weight: Optional[torch.FloatTensor] = None,
        reference_importances: Optional[torch.LongTensor] = None,
        cluster_ignore_norms: bool = True,
        additional_parameters: dict[str, torch.Tensor] = {},
        pad = 0,
    ):
        super(VectorQuantizer_1st_order, self).__init__(
            codes,
            codebook,
            reconstructed_shape,
            normalizer,
            reference_weight,
            reference_importances,
            cluster_ignore_norms=cluster_ignore_norms,
            additional_parameters=additional_parameters,
            pad = pad
        )

    def forward(self, leave_padding:bool = False):
        """returns the quantized weights

        Args:
            leave_padding (bool, optional): if True, we return the weight with the padding included. Defaults to False.
        """
        try:
            return self.precomputed_weight
        except AttributeError:
            pass
        if not leave_padding:
            return super().forward()
        else:
            reconstructed_weight = self.codebook[self.codes].view(self.reconstructed_shape)
            reconstructed_weight = self.normalizer.denormalize(reconstructed_weight)
            return reconstructed_weight
        
    def pad_hessian(self, hessian:torch.Tensor):
        if hessian.shape[0] != self.reconstructed_shape[1]:
            #pad the hessian
            pad = self.reconstructed_shape[1] - hessian.shape[0]
            hessian = F.pad(hessian, (0, pad, 0, pad))
        return hessian
    
    def determine_optimal_n_parallel(self, hessian:torch.Tensor):
        """determines the optimal n_parallel through binary search

        Args:
            hessian (torch.Tensor): the hessian, of shape (n_in,n_in)
            n_parallel_range (List[int]): the range of n_parallel values to search over
        """
        n_parallel_range = [0, self.reconstructed_shape[0]]
        # print("initial n_parallel_range", n_parallel_range)
        while n_parallel_range[1] - n_parallel_range[0] > 1:
            n_parallel = (n_parallel_range[1] + n_parallel_range[0]) // 2
            try:
                self.update_discrete(hessian, n_parallel, True)
            except QuickStopException:
                n_parallel_range[0] = n_parallel
            #if we run out of memory, we reduce the n_parallel
            except RuntimeError:
                n_parallel_range[1] = n_parallel
        
        for n_parallel in n_parallel_range:
            # self.update_discrete(hessian, n_parallel, True)
            try:
                self.update_discrete(hessian, n_parallel, True)
            except QuickStopException:
                n_parallel_works = n_parallel
                continue
            except RuntimeError:
                break
        # print("n_parallel", n_parallel_works)
        # raise ValueError
        return n_parallel_works

    # @jit.script
    def update_discrete(self,hessian:torch.Tensor, n_parallel:int = -1, quick_check:bool = False,
                        temp:float = -1,
                        threshold:Optional[float] = None):
        """updates the discrete codes, ignoring the influnece of one quantization on
        another

        Args:
            hessian (torch.Tensor): the hessian, of shape (n_in,n_in)
            n_parallel (int, optional): the number of rows to process in parallel, default is 1. Defaults to 1.
            quick_check (bool, optional): if True, we only check the first iteration,
            used to find the optimal n_parallel. Defaults to False.
        """
        if hessian.shape[0] != self.reconstructed_shape[1]:
            hessian = self.pad_hessian(hessian)
            
        if n_parallel == -1:
            n_parallel = self.determine_optimal_n_parallel(hessian)
            # raise ValueError
            # self.update_discrete(hessian, n_parallel, False)
        
        # if hessian.shape[0] != self.reconstructed_shape[1]:
        #     #pad the hessian
        #     pad = self.reconstructed_shape[1] - hessian.shape[0]
        #     hessian = F.pad(hessian, (0, pad, 0, pad))
        
        with torch.no_grad():
            #get the current reconstruction of the quantized weights
            reconstructed_weights = self(True)

            errors = self.reference_weight - reconstructed_weights #shape of (n_out, n_in)

            errors_old = errors.clone()
            n_out, n_in = self.reconstructed_shape
            d = self.codebook.shape[1] #the subvector size
            #initialize the new codes
            new_codes = self.codes.clone().reshape(n_out, n_in//d)

            old_codes = self.codes.reshape(n_out, n_in//d)
            prev_losses = torch.einsum("ij,jk,ik->i", errors, hessian, errors)
            #for each block of n_parallel rows
            J = list(range(0,n_in, d))
            #shuffle
            np.random.shuffle(J)
            for j in tqdm.tqdm(J):
                precompute_non_block = torch.einsum("ij,jk->ik",errors[:,:j],hessian[:j,j:j+d])*2 + torch.einsum("ij,jk->ik",errors[:,j+d:],hessian[j+d:,j:j+d])*2 
                hessian_blocked_out  = hessian[:,j:j+d].clone()
                hessian_blocked_out[j:j+d,j:j+d] = 0
                hessian_block = hessian[j:j+d,j:j+d] 
                row_by_row_h_loss = torch.einsum("ij,jk,ik->i", errors, hessian, errors)
                for i in range(0,n_out,n_parallel):
                    #get the block 

                    reference_block = self.reference_weight[i:i+n_parallel,j:j+d] #shape (n_parallel,d)
                    n_parallel_use = reference_block.shape[0]
                    #precompute the result of multiplying the non block elements by the hessian
                    # precompute_non_block = torch.einsum("ij,jk->ik",errors[i:i+n_parallel],hessian_blocked_out)*2 #shape of (n_parallel,d)
                    #multiply the codes by the normalization factor of the block
                    codebook_use = self.normalizer.denormalize_codebook(self.codebook.T, [[i,i+n_parallel],[j,j+d]])

                    #compute the distance between the block and the codebook
                    difference = (reference_block.unsqueeze(-1) - codebook_use) #shape of (n_parallel, d, n_codes)
          
                    distance_ = torch.einsum("ijk,jl,ilk->ik",difference,hessian_block,difference) #shape of (n_parallel, n_codes)
                    distance = distance_ + torch.sum(precompute_non_block[i:i+n_parallel].unsqueeze(-1) * difference, dim = 1) #shape of (n_parallel, n_codes)
                    #get the new codes
                    if temp < 0:
                        if threshold is not None:
                            ids = torch.argmin(distance, dim = -1)
                            best = distance[torch.arange(n_parallel_use),ids]
                            prev = distance[torch.arange(n_parallel_use),old_codes[i:i+n_parallel,j//d]]
                            frac = (prev - best)
                        # print("greedy_update")
                        # print("minimum distance", torch.min(distance, dim = -1))
                        new_codes[i:i+n_parallel,j//d] = distance.argmin(-1)
                        # raise QuickStopException
                        
                    #otherwise stochastically sample the new codes
                    else:
                        ids = torch.argmin(distance, dim = -1)
                        # print("frac_changes_before", torch.sum(ids != old_codes[i:i+n_parallel,j//d]).item() / n_parallel_use)
                        # best = distance[torch.arange(n_parallel_use),ids]
                        # prev = distance[torch.arange(n_parallel_use),old_codes[i:i+n_parallel,j//d]]
                        # print((prev - best))
                        # probs = 1-torch.exp(-(prev - best)/temp)
                        # probs = 1 - torch.exp(-temp *(prev - best)/(row_by_row_h_loss[i:i+n_parallel]))
                        p = torch.rand(n_parallel_use, device = distance.device)
                        # print("probs", probs, "mean:", torch.mean(probs))
                        # print("p", p)
                        mask = p < temp
                        # print("frac_change:", torch.sum(mask).item() / n_parallel)
                        ids[~mask] = old_codes[i:i+n_parallel,j//d][~mask]
                        new_codes[i:i+n_parallel,j//d] = ids
                        # raise QuickStopException
                    errors[i:i+n_parallel, j:j+d] = difference[torch.arange(n_parallel_use),:,new_codes[i:i+n_parallel,j//d]]
                    if quick_check:
                        raise QuickStopException
                    # new_losses = torch.einsum("ij,jk,ik->i", errors, hessian, errors)
                    # print("minimum distance", torch.min(distance))
                    # print("i", i, "j", j)

                    # code = new_codes[:,j//d]
                    # old_code = old_codes[:,j//d]
                    
                    # new_distance = distance[torch.arange(n_parallel),code]
                    # old_distance = distance[torch.arange(n_parallel),old_code]

                    # hessian_temp = torch.zeros_like(hessian)
                    # hessian_temp[:,j:j+d] = hessian[:,j:j+d]
                    # hessian_temp[j:j+d] = hessian[j:j+d]

                    # new_losses_temp = torch.einsum("ij,jk,ik->i", errors, hessian_temp, errors)
                    # old_losses_temp = torch.einsum("ij,jk,ik->i", errors_old, hessian_temp, errors_old)
                    # print("maximum difference", torch.max(torch.abs(new_distance - new_losses_temp)))
                    # # assert torch.allclose(new_losses_temp, new_distance), f"new_losses_temp: {new_losses_temp}, new_distance: {new_distance}"
                    # # assert torch.allclose(old_losses_temp, old_distance), f"old_losses_temp: {old_losses_temp}, old_distance: {old_distance}"
                    # # print("new_losses_temp", new_losses_temp[error_idx].item(), "old_losses_temp", old_losses_temp[error_idx].item())


                    # if not torch.all(new_losses - 1e-5 <= prev_losses):
                    #     print("i", i, "j", j)
                    #     error_idx = torch.argmax(new_losses - prev_losses)
                    #     new_code = new_codes[error_idx,j//d]
                    #     old_code = old_codes[error_idx,j//d]
                    #     print("error_idx", error_idx, "new_code", new_code, "old_code", old_code)\
                        
                    #     new_distance = distance[error_idx,new_code]
                    #     old_distance = distance[error_idx,old_code]
                    #     print("new_distance", new_distance, "old_distance", old_distance)

                    #     hessian_temp = torch.zeros_like(hessian)
                    #     hessian_temp[:,j:j+d] = hessian[:,j:j+d]
                    #     hessian_temp[j:j+d] = hessian[j:j+d]

                    #     new_losses_temp = torch.einsum("ij,jk,ik->i", errors, hessian_temp, errors)
                    #     old_losses_temp = torch.einsum("ij,jk,ik->i", errors_old, hessian_temp, errors_old)
                    #     print("new_losses_temp", new_losses_temp[error_idx].item(), "old_losses_temp", old_losses_temp[error_idx].item())

                    #     print("difference", torch.abs(new_losses_temp[error_idx].item() - new_distance))
                    #     # print("new distance_", distance_[error_idx,new_code], "old distance_", distance_[error_idx,old_code])
                    #     # # print("error_idx", error_idx)

                    #     # print("new_losses", new_losses[error_idx].item(), "prev_losses", prev_losses[error_idx].item())
                    #     raise ValueError(f"new_losses: {new_losses[error_idx]}, prev_losses: {prev_losses[error_idx]}")
                    
                    # code = new_codes[:,j//d]
                    # old_code = old_codes[:,j//d]
                    
                    # new_distance = distance[torch.arange(n_parallel),code]
                    # old_distance = distance[torch.arange(n_parallel),old_code]

                    # hessian_temp = torch.zeros_like(hessian)
                    # hessian_temp[:,j:j+d] = hessian[:,j:j+d]
                    # hessian_temp[j:j+d] = hessian[j:j+d]

                    # new_losses_temp = torch.einsum("ij,jk,ik->i", errors, hessian_temp, errors)
                    # old_losses_temp = torch.einsum("ij,jk,ik->i", errors_old, hessian_temp, errors_old)
                    # print("maximum difference", torch.max(torch.abs(new_distance - new_losses_temp)))
                    # assert torch.allclose(new_losses_temp, new_distance), f"new_losses_temp: {new_losses_temp}, new_distance: {new_distance}"
                    # assert torch.allclose(old_losses_temp, old_distance), f"old_losses_temp: {old_losses_temp}, old_distance: {old_distance}"
                    # print("new_losses_temp", new_losses_temp[error_idx].item(), "old_losses_temp", old_losses_temp[error_idx].item())

                    # print("new distance_", distance_[error_idx,new_code], "old distance_", distance_[error_idx,old_code])
                    # # print("error_idx", error_idx)

                    # print("new_losses", new_losses[error_idx].item(), "prev_losses", prev_losses[error_idx].item())

                
                    # assert new_loss <= prev_loss, f"new_loss: {new_loss}, prev_loss: {prev_loss}"
                    # print("loss", new_loss)
                    # prev_losses = new_losses.clone()
            #         errors_old = errors.clone()
            #         # raise ValueError
            # raise ValueError
            #get the number of different codes
            n_different = torch.sum(self.codes != new_codes.reshape(-1))
            # # print("n_different", n_different, n_different / self.codes.numel())
            # # print("code counts:")
            # unique, counts = torch.unique(new_codes, return_counts = True)
            # unique_old, counts_old = torch.unique(self.codes, return_counts = True)
            
            # idx = torch.argsort(unique, dim = 0)
            # unique = unique[idx]
            # counts = counts[idx]

            # idx = torch.argsort(unique_old, dim = 0)
            # unique_old = unique_old[idx]
            # counts_old = counts_old[idx]

            # for i in range(unique.numel()):
            #     print(unique[i].item(), counts[i].item(), counts_old[i].item())

            # input("continue?")
            self.codes = new_codes.reshape(-1)
            # assert torch.allclose(self.reference_weight - self(True), errors), "error"
            print("relative change", n_different / self.codes.numel())
            return n_different
            # raise ValueError

    @staticmethod
    def quantize(
        weight: torch.FloatTensor,
        hessian: torch.FloatTensor,
        d: int = 4,
        n_bits: int = 2,  # number of bits per weight
        n_iters: int = 100,
        initialize_method: Literal["grid", "kmeans"] = "kmeans",
        norm_order: list[int] = [0, 1],
        zero: list[bool] = [True, True],
        cluster_ignore_norms: bool = True,
        **kwargs,
    ):
        
        assignments, centriods, (n_out, n_in), normalizer, weight_pad, importances, cluster_ignore_norms, pad, hessian = VectorQuantizer_1st_order.quantize_(
            weight, hessian, d, n_bits, n_iters, initialize_method, norm_order, zero, cluster_ignore_norms, **kwargs)

        return VectorQuantizer_1st_order(
            assignments,
            centriods,
            (n_out, n_in),
            normalizer,
            weight_pad,
            importances,
            cluster_ignore_norms=cluster_ignore_norms,
            pad = pad,

        )
    
    def clean(self):
        super().clean()
        if hasattr(self, "hessian"):
            delattr(self, "hessian")

    def precompute_weight(self):
        self.precomputed_weight = self.forward()
    
    @staticmethod
    def blank_recreate(
        weight: torch.FloatTensor,
        d: int = 4,
        n_bits: int = 2,  # number of bits per weight
        norm_order: list[int] = [0, 1],
        zero: list[bool] = [True, True],
        cluster_ignore_norms: bool = True,
        **kwargs,
    ):
        with torch.no_grad():
            n_out, n_in = weight.shape
            if n_in % d != 0:
                pad = d - (n_in % d)
                weight_pad = F.pad(weight, (0, pad), value = torch.mean(weight).item())
                n_in += pad
            else:
                pad = 0
                weight_pad = weight
            weight_use = weight_pad.clone()
            normalizer, weight_use = quantizer_utils.Normalizer.normalize_init(weight_use, norm_order,zero = zero)

            codebook = torch.zeros(2 ** (int(n_bits * d)), d).to(weight.device)
            codes = torch.zeros(
                (weight_use.shape[0] * weight_use.shape[1]) // d, dtype=torch.long
            ).to(weight.device)
            blank_quantizer = VectorQuantizer_1st_order(
                codes,
                codebook,
                (n_out, n_in),
                normalizer,
                weight_use,
                torch.zeros_like(weight_use[0,:]).reshape(-1, d),
                cluster_ignore_norms=cluster_ignore_norms,
                pad = pad
            )
            # blank_quantizer.clean()
        return blank_quantizer


if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)

    device = torch.device("cuda:7")
    data = torch.load("/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_0/mlp.down_proj.pt")
    W = data["weight"].to(device).to(torch.float32)
    hessian = data["hessian"].to(device).to(torch.float32)
    print(W.shape)

    vq = VectorQuantizer_1st_order.quantize(
        W, hessian, d=5, n_bits=2, n_iters=100, initialize_method="kmeans"
    )
    print(vq.get_n_bits() / vq.get_n_original_parameters())
    vq.set_additional_attributes_as_trainable()

    for name, param in vq.named_parameters():
        print(name, param.shape, param.requires_grad)

    # sys.path.append(os.getcwd())
    import src.alignment.hessian_general_align as hessian_general_align

    hessian_use = hessian + torch.eye(hessian.shape[0]).to(hessian.device) * 1e-4
    # hessian_use = torch.eye(W.shape[0]).to(W.device)
    hessian_general_align.align(
        vq,
        W,
        hessian_use,
        None,
        n_iters=1000,
        val_every=-1,
        patience=100,
        patience_scheduler=10,
        eps=1e-4,
        lr=1e-3,
        low_bound=1e-6,
        clip_grad=1e-1,
        discrete_update_every=50,
        lr_multiplier=0.333333,
        verbose=1,
        discrete_update_kwargs={},
    )
    print("norm_0", vq.norms_0)
    print("norm_1", vq.norms_1)
    print(vq().dtype)
    vq.clean()
    vq.to(torch.float16)
    print(vq().dtype)





