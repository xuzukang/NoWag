import torch
import torch.nn as nn
import time
import tqdm
import copy
import numpy as np


class LoraFineTuner(nn.Module):
    def __init__(self, layer:nn.Linear, rank:int,
                 alpha:float=0.001):
        super(LoraFineTuner, self).__init__()
        
        self.original_layer = layer
        
        self.A = nn.Linear(layer.in_features, rank, bias=False).to(layer.weight.device).to(self.original_layer.weight.dtype)
        self.B = nn.Linear(rank, layer.out_features, bias=False).to(layer.weight.device).to(self.original_layer.weight.dtype)

        with torch.no_grad():
            self.A.weight.copy_(torch.randn_like(self.A.weight))
            self.B.weight.copy_(torch.zeros_like(self.B.weight))
        
        #set which things need gradients
        self.A.weight.requires_grad = True
        self.B.weight.requires_grad = True
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        
        self.alpha = alpha
        self.rank = rank
        
    def forward(self, x:torch.Tensor):

        
        original = self.original_layer(x)
        
        return original + self.alpha/self.rank * self.B(self.A(x))
    
    def convert_to_original(self):

        with torch.no_grad():
            self.original_layer.weight.copy_(self.original_layer.weight + self.alpha/self.rank * self.B.weight @ self.A.weight)
            
        return self.original_layer
    
def apply_lora(module, lora_kwargs):
    if isinstance(module, nn.Linear):
        return LoraFineTuner(module, **lora_kwargs)
       
    #otherwise for all the modules in this 
    for name, child in module.named_children():
        setattr(module, name, apply_lora(child, lora_kwargs))
        
    return module

def deapply_lora(module):
    if isinstance(module, LoraFineTuner):
        return module.convert_to_original()
    
    #otherwise for all the modules in this 
    for name, child in module.named_children():
        setattr(module, name, deapply_lora(child))
        
    return module

def lora_l2_regul(module):
    
    l2 = 0
    if isinstance(module, LoraFineTuner):
        l2 += torch.norm(module.B.weight @ module.A.weight, p=2) * module.alpha/module.rank
    else:
        for name, child in module.named_children():
            l2 += lora_l2_regul(child)
    return l2


def finetune_module(module:nn.Module, inputs:torch.Tensor, 
                    targets:torch.Tensor, lora:bool=True,
                    lora_kwargs:dict={},
                    n_iters:int=100, lambda_regul:float=0.1):
    
    #convert all to full precision
    module = module.float()
    inputs = inputs.float()
    targets = targets.float()
    
    
    # module_original = copy.deepcopy(module)
    # if debug:
    #     print("saving for future debugging")

    torch.save({"model":module.state_dict(),
                "model.configs":module.self_attn.config,
                # "model.layer_idx":module.self_attn.layer_idx,
                "inputs":inputs,
                "targets":targets}, "test/model_checkpoint.pth")

    # raise ValueError
    if lora:
        module = apply_lora(module, lora_kwargs)
        
    # print(module)
    # print(module_original)
    # raise ValueError
    #set the module to training mode
    module.train()
    
    #create an optimizer
    optimizer = torch.optim.Adam(module.parameters())

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # optimizer = torch.optim.SGD(module.parameters(), lr=1e-2)
    
    #loss
    loss_fn = nn.MSELoss()
    print("inputs.shape", inputs.shape)
    # raise ValueError
    #train the model

    # js = torch.randperm(inputs.shape[0]).reshape(-1, batch_size)
    tick = time.time()
    best_loss = float("inf")

    for i in range(n_iters):
        total_loss = 0
        total_regul_loss = 0
        total_reconstruct_loss = 0
        # tick = time.time()
        for j in range(inputs.shape[0]):
            # optimizer.zero_grad()
            inputs_batch = inputs[j].unsqueeze(0)
            targets_batch = targets[j].unsqueeze(0)
            # print("inputs_batch.shape", inputs_batch.shape)
            # print("targets_batch.shape", targets_batch.shape)
        
        
            #forward pass
            outputs = module(inputs_batch)[0]
            # print("outputs.shape", outputs.shape)
            # raise ValueError
            #calculate loss
            reconstruct_loss = loss_fn(outputs, targets_batch)
            if lambda_regul> 0:
                regul_loss = lora_l2_regul(module)
                loss = reconstruct_loss + lambda_regul * regul_loss
                total_regul_loss += regul_loss.item()
            else:
                loss = reconstruct_loss
            total_reconstruct_loss += reconstruct_loss.item()
                
            # print("loss", loss.item(), torch.isfinite(loss))
            assert torch.isfinite(loss), f"loss is not finite {loss}"
            total_loss += loss.item()
            #backprop
            loss.backward()

            # print(torch.any(torch.isnan(module.self_attn.q_proj.A.weight.grad)))
            # print(torch.all(~torch.isfinite(module.self_attn.q_proj.A.weight.grad)))

            #update weights
            # optimizer.step()
            # optimizer_states = optimizer.state
            # print(optimizer_states)
            

            # print(torch.any(torch.isnan(module.self_attn.q_proj.A.weight)))
        # raise ValueError
        if total_loss < best_loss:
            print("saving because model is best")
            best_model = copy.deepcopy(module)
            best_loss = total_loss
        else:
            scheduler.step()

        if i != n_iters-1:
            optimizer.step()
            optimizer.zero_grad()

        if i  % (int(np.ceil(n_iters/10))) == 0:
            tock = time.time()
            print(f"Iteration {i}, loss {total_loss}, regul_loss {total_regul_loss}, reconstruct_loss {total_reconstruct_loss}, time {round(tock-tick,3)}")
            # print(module.mlp.up_proj.A.weight)
            # print(module.mlp.up_proj.B.weight)
            tick = tock
    
    # print(module_original.self_attn.q_proj.weight)
    module = deapply_lora(best_model)
    # print(adjusted_module.self_attn.q_proj.weight)

    # if debug:
    # total_loss = 0
    # for i in range(inputs.shape[0]):
    #     outputs = adjusted_module(inputs[i:i+1])[0]
    #     loss = loss_fn(outputs, targets[i:i+1])
    #     total_loss += loss.item()
    # print("Final loss", total_loss)

    module = module.half()
    return module


if __name__ == "__main__":
    test_data = torch.load("test/model_checkpoint.pth")

    import transformers

    module = transformers.models.llama.modeling_llama.LlamaDecoderLayer(test_data["model.configs"])

    module.float()

    module.load_state_dict(test_data["model"])

    inputs = test_data["inputs"]
    targets = test_data["targets"]

    print("inputs.shape", inputs.dtype)
    print("targets.shape", targets.dtype)

    module.to(inputs.device)

    print("module", module.self_attn.q_proj.weight.dtype)
    # raise ValueError

    new_module = finetune_module(module, inputs, targets, lora=True, lora_kwargs={"rank":8, "alpha":0.01}, n_iters=10, lambda_regul=0.01)
    

