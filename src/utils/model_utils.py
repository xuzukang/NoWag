import torch
import torch.nn as nn


DEV = torch.device("cuda:0")


def get_llama(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "llama-3" not in model.lower():
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    else:
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            #  low_cpu_mem_usage=True,
            # attn_implementation='sdpa'
        )
    model.seqlen = 8192
    print("Model loaded.", model)
    return model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

@torch.no_grad()
def inference_layer(layer:nn.Module, 
                    inps:torch.FloatTensor,
                    outs:torch.FloatTensor,
                    attention_mask:torch.FloatTensor=None,
                    dev:str = "cuda:0",
                    offload_activations:bool=False)->torch.FloatTensor:
    """Inference a single layer"""

    #check that the inps have the same 
    for j in range(len(inps)):
        if offload_activations:
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask)[0].cpu()
        else:
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

    return outs
