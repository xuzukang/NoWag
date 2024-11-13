import torch



def round_to_the_nearest(x, codebook):
    return torch.argmin(torch.abs(x.unsqueeze(0) - codebook.unsqueeze(1)), dim = 0)


def normalize(weight,norm_order:list[int] = [0,1]):
    """normalize the input weight matrix
    norm order dictates the order of the norm to use for normalization
    expected to be returned as norms_0, norms_1
    """
    norms = [None, None]
    weight_use = weight.clone()
    for i in norm_order:
        norm_temp = weight_use.norm(i)
        norms[i] = norm_temp
        weight_use = weight_use / norm_temp.unsqueeze(i)
    
    return norms[0], norms[1], weight_use

