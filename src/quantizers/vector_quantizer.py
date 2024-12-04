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
import src.quantizers.quantizer_parent as quantizer_parent


class VectorQuantizer(quantizer_parent.QuantizerParent):
    def __init__(
        self,
        codes: torch.LongTensor,
        codebook: torch.FloatTensor,
        reconstructed_shape: Union[Tuple[int, int], torch.Size],
        norms_1: Optional[torch.FloatTensor] = None,
        norms_0: Optional[torch.FloatTensor] = None,
        reference_weight: Optional[torch.FloatTensor] = None,
        reference_importances: Optional[torch.LongTensor] = None,
        additional_parameters: dict[str, torch.Tensor] = {},
    ):
        """Vector Quantizer without any sparse preservations

        Args:
            codes (torch.LongTensor): the codes for the quantizer, of shape (n_values)
            codebook (torch.FloatTensor): the codebook for the quantizer, of shape (codebook_size, d)
            reconstructed_shape (Union[Tuple[int,int], torch.Size]): the size of the weight matrix we want to reshape to, after dequantization, expected to be (n_out,n_in) where n_out * n_in/d = n_values and n_in is divisible by d
            norms_1 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 0th dimension, of shape n_in. Defaults to None.
            norms_0 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 1st dimension, of shape n_out. Defaults to None.
            reference_weight (Optional[torch.FloatTensor], optional): The reference weight matrix, of shape (n_out,n_in). Defaults to None.
        """
        super(VectorQuantizer, self).__init__(
            codes,
            codebook,
            reconstructed_shape,
            reference_weight,
            dict({"norms_1": norms_1, "norms_0": norms_0}, **additional_parameters),
        )
        if reference_importances is not None:
            self.register_buffer("reference_importances", reference_importances)
        else:
            self.reference_importances = None

    def forward(self):
        reconstructed_weight = self.codebook[self.codes].view(self.reconstructed_shape)
        if self.norms_1 is not None:
            reconstructed_weight = reconstructed_weight * self.norms_1.unsqueeze(1)
        if self.norms_0 is not None:
            reconstructed_weight = reconstructed_weight * self.norms_0.unsqueeze(0)

        # print(self.reference_importances    )
        return reconstructed_weight

    @torch.no_grad()
    def ema_update_importances(
        self, new_importances: torch.FloatTensor, decay: float = 0.99
    ):
        """update the importances using exponential moving average

        reference_importances = decay * reference_importances + (1-decay) * new_importances

        Args:
            new_importances (torch.FloatTensor): the new importances, of shape (n_values/d, d)
            decay (float, optional): _description_. Defaults to 0.99.
        """
        # print("self.reference_importances", self.reference_importances.shape)
        # print("new_importances", new_importances.shape)
        self.reference_importances = decay * self.reference_importances + (
            1 - decay
        ) * new_importances.reshape(self.reference_importances.shape)

    def get_importances(self):
        return self.reference_importances

    def update_discrete(self):
        with torch.no_grad():
            reference_importances = self.reference_importances

            self.codes = quantizer_utils.cluster_e_step(
                self.reference_weight, self.codebook, reference_importances
            )

    # def get_reference_importances

    def process_old_importances(self):
        """if we have loaded the importances in the old format,
        we will process it to the new format
        """
        if self.reference_importances is not None:
            # old reference importances were of shape (n_out * n_in/d, d)
            # where I simply repeated (n_in/d, d) n_out times
            # the new reference importances are of shape (n_in/d, d)
            if (
                self.reference_importances.shape[0]
                == self.n_out * self.n_in // self.codebook.shape[1]
            ):
                self.reference_importances = self.reference_importances.reshape(
                    self.n_out, -1, self.codebook.shape[1]
                )[0]
            else:
                print(
                    "Reference importances are of shape",
                    self.reference_importances.shape,
                    "expected",
                    (
                        self.n_out,
                        self.n_in // self.codebook.shape[1],
                        self.codebook.shape[1],
                    ),
                )

        else:
            print("No reference importances found")

    @staticmethod
    def quantize(
        weight: torch.FloatTensor,
        hessian: torch.FloatTensor,
        d: int = 4,
        n_bits: int = 2,  # number of bits per weight
        n_iter: int = 100,
        initialize_method: Literal["grid", "kmeans"] = "kmeans",
        norm_order: list[int] = [0, 1],
    ):
        weight_use = weight.clone()
        norm_0, norm_1, weight_use = quantizer_utils.normalize(weight_use, norm_order)
        denormalize_matrix = torch.ones_like(weight_use)
        if norm_0 is not None:
            denormalize_matrix = denormalize_matrix * norm_0.unsqueeze(0)
        if norm_1 is not None:
            denormalize_matrix = denormalize_matrix * norm_1.unsqueeze(1)

        H_diag = torch.diag(hessian)
        # H_diag = H_diag.reshape(-1,d)
        importances = (H_diag).reshape(  # .unsqueeze(0).expand(weight.shape[0], -1)
            -1, d
        )
        importances_use = (
            importances  # * denormalize_matrix.reshape(importances.shape)**2
        )

        weight_subvectors = weight_use.reshape(-1, d)
        n_subvectors = weight_subvectors.shape[0]
        n_centriods = 2 ** (n_bits * d)
        print(n_centriods)

        if initialize_method == "grid":
            grid_points = []
            for i in range(d):
                grid_points.append(
                    torch.linspace(
                        torch.min(weight_subvectors[:, i]),
                        torch.max(weight_subvectors[:, i]),
                        2**n_bits,
                    )
                    .cpu()
                    .tolist()
                )
            centriods = itertools.product(*grid_points)
            print(len(grid_points))
            centriods = torch.tensor(list(centriods)).to(weight.device)
            print(centriods.shape)
            assignments = quantizer_utils.cluster_e_step(
                weight_subvectors, centriods, importances_use
            )

        elif initialize_method == "kmeans":
            n_1 = torch.from_numpy(
                np.random.choice(n_subvectors, n_centriods, replace=False)
            ).to(weight.device)
            # print("n_1", n_1)
            # print("max", torch.max(n_1), "min", torch.min(n_1))
            # print(X.shape)
            centriods = weight_subvectors[n_1, :]

            for i in range(n_iter):
                assignments = quantizer_utils.cluster_e_step(
                    weight_subvectors, centriods, importances_use
                )
                # print(assignments)
                # print(assignments.shape)
                centriods = quantizer_utils.cluster_m_step(
                    weight_subvectors, assignments, n_centriods, importances_use
                )
                if i > 0:
                    if torch.all(assignments == assignments_old):
                        # print("breaking at iteration", i)
                        break
                    # print("n_change:", torch.sum(assignments != assignments_old))
                assignments_old = assignments.clone()

        else:
            raise ValueError("initialize_method must be either 'grid' or 'kmeans'")

        return VectorQuantizer(
            assignments,
            centriods,
            weight.shape,
            norm_1,
            norm_0,
            weight_subvectors,
            importances,
        )

    def get_n_bits(self):
        n_bits = super().get_n_bits()
        # sum the bits of the norms
        if self.norms_0 is not None:
            n_bits += self.norms_0.numel() * 16
        if self.norms_1 is not None:
            n_bits += self.norms_1.numel() * 16
        return n_bits

    def clean(self):
        super().clean()
        delattr(self, "reference_importances")

    @staticmethod
    def blank_recreate(
        weight: torch.FloatTensor,
        d: int = 4,
        n_bits: int = 2,  # number of bits per weight
        norm_order: list[int] = [0, 1],
    ):
        with torch.no_grad():
            weight_use = weight.clone()
            norm_0, norm_1, weight_use = quantizer_utils.normalize(
                weight_use, norm_order
            )

            codebook = torch.zeros(2 ** (n_bits * d), d).to(weight.device)
            codes = torch.zeros(
                (weight.shape[0] * weight.shape[1]) // d, dtype=torch.long
            ).to(weight.device)
            blank_quantizer = VectorQuantizer(
                codes,
                codebook,
                weight.shape,
                norm_1,
                norm_0,
                weight_use.reshape(-1, d),
                torch.zeros_like(weight_use).reshape(-1, d),
            )
            blank_quantizer.clean()
        return blank_quantizer


class VectorQuantizerSparseUnstructured(VectorQuantizer):
    def __init__(
        self,
        codes: torch.LongTensor,
        codebook: torch.FloatTensor,
        reconstructed_shape: Union[Tuple[int, int], torch.Size],
        mask: torch.BoolTensor,
        sparse_values: torch.FloatTensor,
        norms_1: Optional[torch.FloatTensor] = None,
        norms_0: Optional[torch.FloatTensor] = None,
        reference_weight: Optional[torch.FloatTensor] = None,
        reference_importances: Optional[torch.LongTensor] = None,
    ):
        """Vector Quantizer without any sparse preservations

        Args:
            codes (torch.LongTensor): the codes for the quantizer, of shape (n_values)
            codebook (torch.FloatTensor): the codebook for the quantizer, of shape (codebook_size, d)
            reconstructed_shape (Union[Tuple[int,int], torch.Size]): the size of the weight matrix we want to reshape to, after dequantization, expected to be (n_out,n_in) where n_out * n_in/d = n_values and n_in is divisible by d
            mask (torch.BoolTensor): the mask for the sparse values, of shape (n_out,n_in) false for sparse values
            sparse_values (torch.FloatTensor): the sparse values, of shape torch.sum(~mask)
            norms_1 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 0th dimension, of shape n_in. Defaults to None.
            norms_0 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 1st dimension, of shape n_out. Defaults to None.
            reference_weight (Optional[torch.FloatTensor], optional): The reference weight matrix, of shape (n_out,n_in). Defaults to None.
        """
        super(VectorQuantizerSparseUnstructured, self).__init__(
            codes,
            codebook,
            reconstructed_shape,
            norms_1,
            norms_0,
            reference_weight,
            reference_importances,
            {"sparse_values": sparse_values},
        )
        self.register_buffer("mask", mask)
        # self.register_buffer('sparse_values', sparse_values)

    def forward(self):
        reconstructed_weight = super().forward()
        reconstructed_weight[~self.mask] = self.sparse_values
        return reconstructed_weight

    def ema_update_importances(self, new_importances, decay=0.99):
        new_importances = new_importances * self.mask.reshape(new_importances.shape)
        super().ema_update_importances(new_importances, decay)

    @staticmethod
    def quantize(
        weight: torch.FloatTensor,
        hessian: torch.FloatTensor,
        mask_fn: Callable[
            [torch.FloatTensor, torch.FloatTensor, float], torch.BoolTensor
        ],
        d: int = 4,
        n_bits: int = 2,  # number of bits per weight
        frac_sparse: float = 0.005,
        n_iter: int = 100,
        initialize_method: Literal["grid", "kmeans"] = "kmeans",
        norm_order: list[int] = [0, 1],
    ):
        with torch.no_grad():
            weight_use = weight.clone()
            mask = mask_fn(
                weight_use, hessian, frac_sparse
            )  # some thing to note is that we can actually do structured sparsity here
            weight_use[~mask] = 0
            sparse_values = weight_use[~mask]
            norm_0, norm_1, weight_use = quantizer_utils.normalize(
                weight_use, norm_order
            )

            denormalize_matrix = torch.ones_like(weight_use)
            if norm_0 is not None:
                denormalize_matrix = denormalize_matrix * norm_0.unsqueeze(0)
            if norm_1 is not None:
                denormalize_matrix = denormalize_matrix * norm_1.unsqueeze(1)

            denormalize_matrix[~mask] = 0

            H_diag = torch.diag(hessian)
            H_diag = H_diag.reshape(-1, d)
            importances = (H_diag.unsqueeze(0).expand(weight.shape[0], -1, -1)).reshape(
                -1, d
            )
            importances[~mask.reshape(-1, d)] = 0

            weight_subvectors = weight_use.reshape(-1, d)
            n_subvectors = weight_subvectors.shape[0]
            n_centriods = 2 ** (n_bits * d)
            # print(n_centriods)

            if initialize_method == "grid":
                grid_points = []
                for i in range(d):
                    grid_points.append(
                        torch.linspace(
                            torch.min(weight_subvectors[:, i]),
                            torch.max(weight_subvectors[:, i]),
                            n_bits,
                        )
                        .cpu()
                        .tolist()
                    )
                print(len(grid_points))
                centriods = itertools.product(*grid_points)
                centriods = torch.tensor(list(centriods)).to(weight.device)
                print(centriods.shape)
                assignments = quantizer_utils.cluster_e_step(
                    weight_subvectors, centriods, importances
                )

            elif initialize_method == "kmeans":
                n_1 = torch.from_numpy(
                    np.random.choice(n_subvectors, n_centriods, replace=False)
                ).to(weight.device)
                # print("n_1", n_1)
                # print("max", torch.max(n_1), "min", torch.min(n_1))
                # print(X.shape)
                centriods = weight_subvectors[n_1, :]

                for i in range(n_iter):
                    assignments = quantizer_utils.cluster_e_step(
                        weight_subvectors, centriods, importances
                    )
                    # print(assignments)
                    # print(assignments.shape)
                    centriods = quantizer_utils.cluster_m_step(
                        weight_subvectors, assignments, n_centriods, importances
                    )
                    if i > 0:
                        if torch.all(assignments == assignments_old):
                            # print("breaking at iteration", i)
                            break
                        # print("n_change:", torch.sum(assignments != assignments_old))
                    assignments_old = assignments.clone()

            else:
                raise ValueError("initialize_method must be either 'grid' or 'kmeans'")

        return VectorQuantizerSparseUnstructured(
            assignments,
            centriods,
            weight.shape,
            mask,
            sparse_values,
            norm_1,
            norm_0,
            weight_subvectors,
            importances,
        )

    def get_n_bits(self):
        n_bits = super().get_n_bits()
        n_bits += 3 * torch.sum(~self.mask).item() * 16
        return n_bits

    @staticmethod
    def blank_recreate(
        weight: torch.FloatTensor,
        n_sparse: int,
        d: int = 4,
        n_bits: int = 2,  # number of bits per weight
        norm_order: list[int] = [0, 1],
    ):
        with torch.no_grad():
            weight_use = weight.clone()
            norm_0, norm_1, weight_use = quantizer_utils.normalize(
                weight_use, norm_order
            )

            codebook = torch.zeros(2 ** (n_bits * d), d).to(weight.device)
            codes = torch.zeros(
                (weight.shape[0] * weight.shape[1]) // d, dtype=torch.long
            ).to(weight.device)

            mask = (
                torch.ones(weight.shape, dtype=torch.bool).to(weight.device).flatten()
            )
            mask[:n_sparse] = False
            mask = mask.view(weight.shape)

            sparse_values = torch.zeros(n_sparse).to(weight.device)

            blank_quantizer = VectorQuantizerSparseUnstructured(
                codes,
                codebook,
                weight.shape,
                mask,
                sparse_values,
                norm_1,
                norm_0,
                weight_use.reshape(-1, d),
                torch.zeros_like(weight_use).reshape(-1, d),
            )
        return blank_quantizer


if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)

    device = torch.device("cuda:1")
    data = torch.load("test/weights_hessian.pt")
    W = data["weights"].to(device)
    hessian = data["hessian"].to(device)
    print(W.shape)

    vq = VectorQuantizer.quantize(
        W, hessian, d=4, n_bits=2, n_iter=100, initialize_method="kmeans"
    )
    print(vq.get_n_bits() / vq.get_n_original_parameters())
    vq.set_additional_attributes_as_trainable()

    for name, param in vq.named_parameters():
        print(name, param.shape, param.requires_grad)

    # sys.path.append(os.getcwd())
    import src.alignment.hessian_general_align as hessian_general_align

    hessian_use = hessian / hessian.shape[0]
    # hessian_use = torch.eye(W.shape[0]).to(W.device)
    hessian_general_align.align(
        vq,
        W,
        hessian_use,
        None,
        n_iters=100,
        val_every=-1,
        patience=100,
        patience_scheduler=1,
        eps=1e-4,
        lr=1e-3,
        low_bound=1e-6,
        clip_grad=1e-1,
        discrete_update_every=1,
        lr_multiplier=0.9,
        verbose=10,
    )
    print("norm_0", vq.norms_0)
    print("norm_1", vq.norms_1)
    print(vq().dtype)
    vq.clean()
    vq.to(torch.float16)
    print(vq().dtype)
