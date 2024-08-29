import sklearn.kmeans as kmeans
import torch

#tree based quantizer
class naiveTreeBasedQuantizer:
    def __init__(self, depth = 3, middle_bin_frac = 0.3333):
        self.depth = depth
        self.middle_bin_frac = middle_bin_frac

    
    def determine_s(self, x: torch.Tensor):
        
        #we split into 3 bins, one centered around -s, one centered around 0, and one centered around s
        #we want the middle bin to contain middle_bin_frac of the data

        abs_x = torch.abs(x)
        abs_x = torch.sort(abs_x)

        #the middle bin will contain middle_bin_frac of the data
        threshold = abs_x[int(self.middle_bin_frac * len(abs_x))]

        s = threshold * 2
