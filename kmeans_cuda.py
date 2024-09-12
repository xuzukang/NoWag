import numbda as nb
import numpy as np 
import torch 
import numbda.cuda as cuda

@cuda.jit(
    'int32[:](float32[:,:], float32[:,:], float32[:])',
    device=True
)
def assignment_step_constant_weights(X, centers, weights):
    """Assignment step of the K-means algorithm with constant weights
    Supports only 2D data points
    Args:
        X (torch.Tensor): Data points, shape of (n_samples, n_features) 
        centers (torch.Tensor): Cluster centers, shape of (n_clusters, n_features)
        weights (_type_): Weights for each cluster, shape of (n_clusters,)
    """
    thread_id = cuda.grid(1)
    