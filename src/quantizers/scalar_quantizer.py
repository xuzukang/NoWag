import src.quantizers.quantizer_parent as quantizer_parent
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
import src.utils.quantizer as quantizer_utils
import src.utils.utils as utils


class ScalarQuantizer(quantizer_parent.QuantizerParent):

    def __init__(self,
                 codes: torch.LongTensor,
                 codebook: torch.FloatTensor,
                 reconstructed_shape: Union[Tuple[int,int], torch.Size],
                 norms_1:Optional[torch.FloatTensor] = None,
                 norms_0:Optional[torch.FloatTensor] = None,
                 sparse_mask:Optional[torch.BoolTensor] = None,
                 sparse_values:Optional[torch.FloatTensor] = None,
                 reference_weight:Optional[torch.FloatTensor] = None,
                 ):
        """Scalar quantizer class

        Args:
            codes (torch.LongTensor): the codes for the quantizer, of shape (n_values)
            codebook (torch.FloatTensor): the codebook for the quantizer, of shape (codebook_size)
            reconstructed_shape (Union[Tuple[int,int], torch.Size]): the size of the weight matrix we want to reshape to, after dequantization, expected to be (n_out,n_in) where n_out * n_in = n_values
            norms_1 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 0th dimension, of shape n_in. Defaults to None.
            norms_0 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 1st dimension, of shape n_out. Defaults to None.
            sparse_mask (Optional[torch.BoolTensor], optional): The mask of which values are sparse, if sparse False, if not True, of shape (n_out,n_in). Defaults to None.
            sparse_values (Optional[torch.FloatTensor], optional): The sparse values of the weight matrix, of shape sum(~sparse_mask). Defaults to None.
            reference_weight (Optional[torch.FloatTensor], optional): The reference weight matrix, of shape (n_out,n_in). Defaults to None.
        """
        super(ScalarQuantizer, self).__init__(codes, codebook, reconstructed_shape, reference_weight,
                                                {"norms_1":norms_1, "norms_0":norms_0, "sparse_values":sparse_values})

        if sparse_mask is not None:
            self.register_buffer('sparse_mask', sparse_mask)
            assert sparse_values is not None, "sparse_values must be provided if sparse_mask is provided"
            # self.register_buffer('sparse_values', sparse_values)
        else:
            self.sparse_mask = None



    # def set_additional_attributes_as_trainable(self):
    #     if self.norms_1 is not None:
    #         self.norms_1.requires_grad = True
    #     if self.norms_0 is not None:
    #         self.norms_0.requires_grad = True
    #     if self.sparse_mask is not None:
    #         self.sparse_values.requires_grad = True

    # def set_additional_attributes_as_non_trainable(self):
    #     if self.norms_1 is not None:
    #         self.norms_1.requires_grad = False
    #     if self.norms_0 is not None:
    #         self.norms_0.requires_grad = False
    #     if self.sparse_mask is not None:
    #         self.sparse_values.requires_grad = False


        

    def forward(self):
        """return the dequantized weight matrix"""

        reconstructed = self.codebook[self.codes].view(self.reconstructed_shape)
        if self.norms_1 is not None:
            reconstructed = reconstructed * self.norms_1.unsqueeze(1)
        if self.norms_0 is not None:
            reconstructed = reconstructed * self.norms_0.unsqueeze(0)
        
        if self.sparse_mask is not None:
            reconstructed[~self.sparse_mask] = self.sparse_values
        
        return reconstructed
    

    def update_discrete(self):
        assert hasattr(self, 'reference_weight'), "reference_weight must be provided to update the discrete values"
        with torch.no_grad():
            self.codes = quantizer_utils.round_to_the_nearest(self.get_reference_weight().view(-1), self.codebook)


    @staticmethod
    def quantize(weight:torch.FloatTensor,
                 hessian:torch.FloatTensor,
                 n_bits:int,
                 sparse_threshold:float = float('inf'),
                 norm_order:List[int] = [0,1],
    ):
        """quantize the input weight matrix

        Args:
            weight (torch.FloatTensor): the weight matrix to quantize, of shape (n_out,n_in)
            n_bits (int): the number of bits to quantize to
            sparse_threshold (Optional[float], optional): the threshold at which to consider a value sparse. Defaults to float('inf'), in which case no values are considered sparse.
            norm_order (Optional[List[int]], optional): the order of the norms to use for normalization. Defaults to [], in which case no normalization is done.
        """
                 
        #first identify the weights above the threshold
        weight_use = weight.clone()
        if sparse_threshold == float('inf'):
            sparse_mask = None
            sparse_weights = None
        else:
            sparse_mask = weight.abs() < sparse_threshold
            sparse_weights = weight[~sparse_mask]
            weight_use[~sparse_mask] = 0
        
        #normalize the weights
        norm_0,norm_1,weight_use = quantizer_utils.normalize(weight_use,norm_order)

        #create a default codebook which is the linspace of the min and max of the weight matrix

        codebook = torch.linspace(weight_use.min(),weight_use.max(),2**n_bits)
        codes = quantizer_utils.round_to_the_nearest(weight_use.view(-1), codebook)

        return ScalarQuantizer(codes, 
                               codebook, 
                               weight.shape, 
                               norm_1, 
                               norm_0, 
                               sparse_mask,  
                               sparse_weights,
                               weight_use)
    
    def get_n_bits(self):   
        n_bits = super().get_n_bits()
        #sum the bits of the norms
        if self.norms_1 is not None:
            n_bits += self.norms_1.numel() * 16
        if self.norms_0 is not None:
            n_bits += self.norms_0.numel() * 16

        if self.sparse_mask is not None:    
            n_bits += self.sparse_mask.numel() * 3 * 16 # 3 because we need 4 bytes for the location of this sparse value
        
        return n_bits
    

        
        
        


        


        
