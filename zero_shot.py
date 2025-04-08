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

def zero_shot(base_model, model, 
              device = "cuda:0",
              batch_size = 1,
              tasks:list[str] = ["winogrande", "piqa", "hellaswag", "arc_easy", "arc_challenge"],
              num_fewshot:int = 0):
  
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
  return results["results"]

if __name__ == "__main__":
  import sys 

  sys.stderr = sys.stdout
  torch.set_grad_enabled(False)

  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model", type=str, help = "the hf base model to use",
                      default = "meta-llama/Llama-2-7b-hf")
  parser.add_argument("--quantized_weight_yaml", type=str, help = "the path to the quantized weight",
                      default = None)
  parser.add_argument("--quantized_weight_hf", type=str, help = "the path to the quantized weight",
                      default = None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", type=str, help = "the device to run the compression on", default = "cuda:0")
  parser.add_argument('--batch_size', type=int, default=1, help='batch size')
  parser.add_argument("--tasks", type=str, nargs = "+", default = 
                      [
                        "winogrande",
                        "piqa", 
                        "hellaswag",
                        "arc_easy", 
                        "arc_challenge"
                        ])
  parser.add_argument("--output_path", default=None, type=str)
  parser.add_argument('--num_fewshot', type=int, default=0)
  parser.add_argument('--limit', type=int, default=None)
  parser.add_argument('--apply_chat_template', action='store_true')
  parser.add_argument('--fewshot_as_multiturn', action='store_true')
  parser.add_argument("--log_wandb", action="store_true", help = "log to wandb")
  parser.add_argument("--save",
                      help="Save the results to the specified path",
                      action="store_true")

  args = parser.parse_args()
  torch.set_grad_enabled(False)
  random.seed(args.seed)
  torch.random.manual_seed(args.seed)

  # model = get_llama(args.base_model,
  #                   device_map="auto",
  #                   dtype=torch.float16)
  # model = model.to(args.device)

  if args.quantized_weight_yaml is not None:
    model = get_llama(args.base_model,
                    device_map="balanced",
                    dtype=torch.float16)
    checkpoints_paths = yaml.load(open(args.quantized_weight_yaml, "r"), Loader=yaml.FullLoader)
    
    
    model, n_bits, n_params = load_model_from_checkpoints(checkpoints = checkpoints_paths,
                                        base_model = args.base_model,
                                        model = model,
                                        device = None,
                                        cache_reconstruct=True)
    print(f"Loaded model with {n_bits} bits and {n_params} parameters")
  elif args.quantized_weight_hf is not None:
    
    model = LlamaForCausalLM.from_pretrained(args.quantized_weight_hf,
                                      torch_dtype=torch.float16,
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map = "auto")
    
    utils.recursive_apply(model, "cache_reconstruct", {'denormalize': True,
                                                       'offload':True})
  else:
    print("no compressed model was provided")
    model = get_llama(args.base_model,
                    device_map="auto",
                    dtype=torch.float16)
    
    
    
  results = zero_shot(args.base_model, model,
                      device=args.device,
                      batch_size = args.batch_size,
                      tasks = args.tasks,
                      num_fewshot = args.num_fewshot)

  if args.save:
    if args.output_path is None:
      if args.quantized_weight_yaml is not None:
        args.output_path = os.path.dirname(args.quantized_weight_yaml) + "/eval_results.yaml"
      elif args.quantized_weight_hf is not None:
        args.output_path = os.path.dirname(args.quantized_weight_hf.replace("compressed_hf","compressed")) + "/eval_results.yaml"
      else:
        args.output_path = "eval_results.yaml"
    print("Saving results to", args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if os.path.exists(args.output_path):
      existing_results = yaml.load(open(args.output_path, "r"), Loader=yaml.FullLoader)
    else:
      existing_results = {}
      
    zero_shot_results = {}
    avg = 0
    for task in results:
      print("task:",results[task]["acc"])
      zero_shot_results[task] = results[task]["acc"]
      avg += results[task]["acc"]
    zero_shot_results["avg"] = avg/len(results)
    existing_results["zero_shot"] = zero_shot_results
    # existing_results["bpv"] = n_bits/n_params
    # existing_results["n_params"] = n_params
    # existing_results["n_bits"] = n_bits
    with open(args.output_path, "w") as f:
      yaml.dump(existing_results, f)

      
      