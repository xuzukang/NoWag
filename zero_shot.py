import lm_eval
from lm_eval import evaluator, tasks
from src.utils.lm_eval_adaptor import LMEvalAdaptor
import argparse
from src.utils.model_utils import find_layers, get_llama, inference_layer
from transformers import AutoTokenizer
from perplexity_eval import load_model_from_checkpoints
import random
import torch
import yaml
import os
import json


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
  parser.add_argument("--quantized_weight_yaml", type=str, help = "the path to the quantized weight")
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

  model = get_llama(args.base_model,
                    device_map="auto")
  # model = model.to(args.device)

  if args.quantized_weight_yaml is not None:
    checkpoints_paths = yaml.load(open(args.quantized_weight_yaml, "r"), Loader=yaml.FullLoader)
    model, _, _ = load_model_from_checkpoints(checkpoints = checkpoints_paths,
                                        base_model = args.base_model,
                                        model = model,
                                        device = None,
                                        cache_reconstruct=True)
    
  results = zero_shot(args.base_model, model,
                      device=args.device,
                      batch_size = args.batch_size,
                      tasks = args.tasks,
                      num_fewshot = args.num_fewshot)
  
# tokenizer = AutoTokenizer.from_pretrained(args.base_model)
# tokenizer.pad_token = tokenizer.eos_token


# lm_eval_model = LMEvalAdaptor(
#   args.base_model,
#   model, tokenizer, batch_size=args.batch_size)
                              

# results = evaluator.simple_evaluate(
#     model=lm_eval_model,
#     tasks=args.tasks,
#     batch_size=args.batch_size,
#     no_cache=True,
#     num_fewshot=args.num_fewshot,
# )


# #still have not implemented the loading a quantized weight

# # lm_obj = Your_LM(model=my_model, batch_size=16)
# lm_obj = HFLM(pretrained = model)

# task_manager = lm_eval.tasks.TaskManager()

# # Setting `task_manager` to the one above is optional and should generally be done
# # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
# results = lm_eval.simple_evaluate( # call simple_evaluate
#     model=lm_obj,
#     tasks=args.zero_shot_tasks,
#     num_fewshot=0,
#     task_manager=task_manager
# )
  if args.save:
    if args.output_path is None:
      args.output_path = os.path.dirname(args.quantized_weight_yaml) + "/results.json"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # results["config"]["model"] = args.base_model + " " + args.quantized_weight_yaml
    with open(args.output_path, "w") as f:
      json.dump(results, f, indent=2)


  avg = 0
  for task in results:
    print(results[task], end = " & ")
    avg += results[task]["acc"]
  print(avg/len(results))
# if args.output_path is not None:
#         os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
#         # otherwise cannot save
#         results["config"]["model"] = args.hf_path
#         with open(args.output_path, "w") as f:
#             json.dump(results, f, indent=2)

# print(evaluator.make_table(results))