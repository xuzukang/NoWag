import os 
import glob



paths = glob.glob("/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/*")

for path in paths:
    s = path.split("/")[-1]
    if len(s.split("-")) == 3:
        new_s = s.split("-")[-1]
        
        new_path = path.replace(s, new_s)
        os.rename(path, new_path)
        