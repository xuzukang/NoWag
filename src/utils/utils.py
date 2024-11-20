import torch
import torch.nn as nn



def param_to_buffer(module:nn.Module, param_name:str):
    """converts a parameter to a buffer in a module

    Args:
        module (nn.Module): the module to convert the parameter to a buffer in
        param_name (str): the name of the parameter to convert to a buffer

    """
    param = getattr(module, param_name).clone()
    delattr(module, param_name)
    module.register_buffer(param_name, param)
    return module

def buffer_to_param(module:nn.Module, buffer_name:str):
    """converts a buffer to a parameter in a module

    Args:
        module (nn.Module): the module to convert the buffer to a parameter in
        buffer_name (str): the name of the buffer to convert to a parameter

    """
    buffer = getattr(module, buffer_name).detach().clone()
    delattr(module, buffer_name)
    module.register_parameter(buffer_name, nn.Parameter(buffer))
    return module

@torch.no_grad()
def update_discrete(module:nn.Module):
    for name, child in module.named_children():
        if hasattr(child, "update_discrete"):
            if callable(getattr(child, "update_discrete")):
                child.update_discrete()
        #otherwise look for its children
        update_discrete(child)
    