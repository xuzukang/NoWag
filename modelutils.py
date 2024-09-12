import torch
import torch.nn as nn




def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res



# import torch
# import torch.nn as nn
# import vector_quantizer

# DEV = torch.device('cuda:0')



# def replace_layers_with_quantizer(module:nn.Module, layer_classes = [nn.Linear],
#                                   quantizer_kwargs = {}):
                   
#     """
#     Replace layers in a model with a vector quantizer.
#     """

#     for name, child in module.named_children():
#         if isinstance(child, tuple(layer_classes)):
#             #if the input and output dimensions are the same
#             # if child.in_features == child.out_features:
#                 # print("replacing layer with quantizer")
#             setattr(module, name, vector_quantizer.VectorQuantizerLayer(child, **quantizer_kwargs))
#         else:
#             replace_layers_with_quantizer(child, layer_classes, quantizer_kwargs)
#     return module

# def quantize_layers(module:nn.Module):
#     """
#     Quantize the weights of layers in a model.
#     """
#     print("here, quantizing layers")
#     print(module)
#     for name, child in module.named_children():
#         if isinstance(child, vector_quantizer.VectorQuantizerLayer):
#             child.quantize()
#             print("Quantized layer")
#             return module, True
#         else:
#             _, flag = quantize_layers(child)
#             if flag:
#                 return module, True
#     return module, False
