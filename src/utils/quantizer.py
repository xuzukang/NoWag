import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.jit as jit
import os
import sys


def round_to_the_nearest(x, codebook):
    return torch.argmin(torch.abs(x.unsqueeze(0) - codebook.unsqueeze(1)), dim=0)


def normalize(weight, norm_order: list[int] = [0, 1]):
    """normalize the input weight matrix
    norm order dictates the order of the norm to use for normalization
    expected to be returned as norms_0, norms_1
    """
    norms = [None, None]
    weight_use = weight.clone()
    for i in norm_order:
        norm_temp = torch.norm(weight_use, dim=i)
        # print(norm_temp)
        norms[i] = norm_temp
        weight_use = weight_use / norm_temp.unsqueeze(i)

    return norms[0], norms[1], weight_use


@jit.script
def cluster_e_step(X: torch.Tensor, centriods: torch.Tensor, weights: torch.Tensor):
    """
    X: torch tensor of the weights, rearanged into a shape of (n, d)
    centriods: torch.tensor of the centriods, shape of (k, d)
    weights: torch.tensor of shape (n',d) n' is a number that is divisible by subblock_size or
    can divide subblock_size
    """

    n = X.shape[0]
    subblock_size = weights.shape[0]
    with torch.no_grad():
        assignments = torch.zeros(n, dtype=torch.int64, device=X.device)

        for i in range(0, n, subblock_size):
            X_block = X[i : i + subblock_size]
            errors = (X_block.unsqueeze(-1) - centriods.T.unsqueeze(0)) ** 2
            # shape of (n, d, k)

            # multiply by the diagonal
            errors = errors * weights.unsqueeze(-1)

            # sum by the d
            errors = errors.sum(1)
            # shape of (n, k)
            # print(errors[0,10,:])
            assignments_block = errors.argmin(-1)
            # print(assignments_block[0,10])
            assignments[i : i + subblock_size] = assignments_block
    return assignments


@jit.script
def cluster_m_step(
    X: torch.Tensor, assignments: torch.Tensor, k: int, weights: torch.Tensor
):
    """
    X: torch tensor of the weights, rearanged into a shape of (n, d)
    assignments: torch.tensor of the assignments, shape of (n)
    k: int, number of clusters
    weights: torch.tensor of shape (n, d)
    """
    n, d = weights.shape

    # compute the new centriods
    centriods = torch.zeros((k, d), dtype=weights.dtype, device=weights.device)
    # shape of (k,d)
    for i in range(k):
        idxs = torch.where(assignments == i)[0]
        assignment_X = X[idxs]  # shape of (n_i,d)
        assignments_weights = weights[idxs % weights.shape[0]]  # shape of (n_i,d)
        # print(assignments_weights.shape, assignment_X.shape)
        centriods[i] = torch.sum(assignments_weights * assignment_X, dim=0) / torch.sum(
            assignments_weights, dim=0
        )

    return centriods
