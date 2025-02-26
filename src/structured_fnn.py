import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.models.llama.modeling_llama as llama


class StructuredMLP(llama.LlamaMLP):
    """Structured pruned MLP from LLAMA model, LoRAP Like"""

    @classmethod
    def from_llama_mlp(cls, llama_mlp: llama.LlamaMLP):

        mlp = StructuredMLP(llama_mlp.config)

        # Copy the weights
        mlp.up_proj.weight.data = llama_mlp.up_proj.weight.data
        mlp.down_proj.weight.data = llama_mlp.down_proj.weight.data
        mlp.gate_proj.weight.data = llama_mlp.gate_proj.weight.data

        return mlp
    
    def prune(self, up_proj_hessianDiag:torch.FloatTensor,
              down_proj_hessianDiag:torch.FloatTensor,
              gate_proj_hessianDiag:torch.FloatTensor,
              frac_keep:float, # ratio of the weights to keep
              frac_lower:float = 0.01 #the fraction of the lowest weights to keep
                ):
        """Prune the MLP using the importances of the projections"""

        up_proj_importance = torch.norm(self.up_proj.weight.data * up_proj_hessianDiag, dim=1)
        down_proj_importance = torch.norm(self.down_proj.weight.data * down_proj_hessianDiag, dim=0)
        gate_proj_importance = torch.norm(self.gate_proj.weight.data * gate_proj_hessianDiag, dim=1)

        joint_importance = up_proj_importance + down_proj_importance + gate_proj_importance

        sorted_idxs = torch.argsort(joint_importance, descending=True)
        mask = torch.zeros_like(joint_importance, dtype=torch.bool)

        n_top = int(len(mask) * (frac_keep - frac_lower))
        n_bottom = int(len(mask) * frac_lower)

        mask[sorted_idxs[:n_top]] = True
        mask[sorted_idxs[-n_bottom:]] = True

        self.up_proj.weight.data = self.up_proj.weight.data[mask]
        self.down_proj.weight.data = self.down_proj.weight.data[:, mask]
        self.gate_proj.weight.data = self.gate_proj.weight.data[mask]

        self.intermediate_size = n_top + n_bottom
        self.config.intermediate_size = n_top + n_bottom





