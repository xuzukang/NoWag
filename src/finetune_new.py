# a simpler version of fine tunining to work on one GPU


import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_pass_and_gather_gradients(layer, data, target, kwargs, indicies,input_batch_size):
    """
    Forward pass and gather gradients for a batch of data
    """
    
    i = 0
    assert data.shape[0] == target.shape[0]
    assert data.shape[0] == indicies.shape[0]

    batch_size = data.shape[0]
    total_loss = 0
    while i < data.shape[0]:
        indexs = indicies[i:i+input_batch_size]
        data_batch = data[indexs]
        target_batch = target[indexs]

        out, *_ = layer(data_batch, **kwargs)
        loss = F.mse_loss(out, target_batch)
        (loss * input_batch_size/ batch_size).backward()
        i += input_batch_size
        total_loss += loss.item()
    
    return total_loss


    


