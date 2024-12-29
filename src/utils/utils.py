import torch
import torch.nn as nn
import wandb

def param_to_buffer(module: nn.Module, param_name: str):
    """converts a parameter to a buffer in a module

    Args:
        module (nn.Module): the module to convert the parameter to a buffer in
        param_name (str): the name of the parameter to convert to a buffer

    """
    param = getattr(module, param_name).clone()
    delattr(module, param_name)
    module.register_buffer(param_name, param)
    return module


def buffer_to_param(module: nn.Module, buffer_name: str):
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
def update_discrete(module: nn.Module):
    for name, child in module.named_children():
        if hasattr(child, "update_discrete"):
            if callable(getattr(child, "update_discrete")):
                child.update_discrete()
        # otherwise look for its children
        update_discrete(child)


def recursive_apply(module: nn.Module, func_name: str, func_kwargs: dict = {}):
    """recursively applies a function to a module and its children

    Args:
        module (nn.Module): the module to apply the function to
        func (function): the function to apply

    """
    for name, child in module.named_children():
        if hasattr(child, func_name):
            if callable(getattr(child, func_name)):
                getattr(child, func_name)(**func_kwargs)
        # otherwise look for its children
        else:
            recursive_apply(child, func_name, func_kwargs)

def recursive_find(module: nn.Module, name:str) -> nn.Module:
    # print(name)
    if name == "":
        return module
    if "." not in name:
        return getattr(module, name)
    else:
        return recursive_find(getattr(module, name[:name.find(".")]), name[name.find(".")+1:])
    

def intialize_wandb(args, config: dict = None):
    if not args.use_wandb:
        return
    
    project_name = None if not hasattr(args, "wandb_project") else args.wandb_project
    run_name = None if not hasattr(args, "wandb_run_name") else args.wandb_run_name
    run_id = None if not hasattr(args, "wandb_run_id") else args.wandb_run_id

    
    wandb.init(project=project_name, name=run_name, id=run_id,
               config = config, resume = "allow")
               