import lm_eval
from lm_eval import evaluator, tasks
from src.utils.lm_eval_adaptor import LMEvalAdaptor
import argparse
from src.utils.model_utils import find_layers, get_llama, inference_layer
from src.model.llama import LlamaForCausalLM
from transformers import AutoTokenizer
from perplexity_eval import load_model_from_checkpoints
import random
import torch
import yaml
import os
import json
from src.utils import utils
import numpy as np

def zero_shot(base_model, model, 
              batch_size = 1,
              tasks:list[str] = ["winogrande", "piqa", "hellaswag", "arc_easy", "arc_challenge"],
              num_fewshot:int = 0,
              results_log_yaml: str = None):
  
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lm_eval_model = LMEvalAdaptor(
        base_model,
        model, tokenizer, batch_size=batch_size)
    
    results = evaluator.simple_evaluate(
        model = lm_eval_model,
        tasks = tasks,
        batch_size = batch_size,
        no_cache = True,
        num_fewshot = num_fewshot,
    )
    print(evaluator.make_table(results))
    
    if results_log_yaml is not None:
            if os.path.exists(results_log_yaml):
                with open(results_log_yaml, "r") as f:
                    results_dict = yaml.load(f, Loader=yaml.FullLoader)
            else:
                results_dict = {}
            
            results_dict["zero_shot"] = {}
    
    #calculate the average zero shot accuracy
    avg_acc = 0
    for task in tasks:
        task_acc = float(results["results"][task]["acc"])
        print("task acc:", task_acc, "type:", type(task_acc))
        # #if the task acc is not a float
        # if not isinstance(task_acc, np.float64):
        #     print("acc is not a float, converting to float")
        #     task_acc = task_acc.item()
        avg_acc += task_acc
        if results_log_yaml is not None:
            #add the task acc to the results dict

            results_dict["zero_shot"][task] = task_acc
    
    avg_acc /= len(tasks)
    print("avg acc:", round(avg_acc * 100,2))
    if results_log_yaml is not None:
        #add the avg acc to the results dict
        results_dict["zero_shot"]["avg_acc"] = avg_acc
        with open(results_log_yaml, "w") as f:
            yaml.dump(results_dict, f)
        print("Results saved to:", results_log_yaml)
        
    return results["results"]