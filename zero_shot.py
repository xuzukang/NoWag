import lm_eval
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
import argparse
from src.utils.model_utils import find_layers, get_llama, inference_layer
from transformers import AutoTokenizer
import random
import torch
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, help = "the hf base model to use",
                    default = "meta-llama/Llama-2-7b-hf")
parser.add_argument("--quantized_weight_path", type=str, help = "the path to the quantized weight")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, help = "the device to run the compression on", default = "cuda:0")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str, nargs = "+", default = 
                    [
                      # "winogrande",
                      # "piqa", 
                      # "hellaswag",
                      # "arc_easy", 
                      "arc_challenge"
                      ])
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--apply_chat_template', action='store_true')
parser.add_argument('--fewshot_as_multiturn', action='store_true')
parser.add_argument("--log_wandb", action="store_true", help = "log to wandb")

args = parser.parse_args()

random.seed(args.seed)
torch.random.manual_seed(args.seed)

model = get_llama(args.base_model)
model = model.to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
tokenizer.pad_token = tokenizer.eos_token


lm_eval_model = HFLM(model,
                         tokenizer=tokenizer,
                         batch_size=args.batch_size)

results = evaluator.simple_evaluate(
    model=lm_eval_model,
    tasks=args.tasks,
    limit=args.limit,
    num_fewshot=args.num_fewshot,
    apply_chat_template=args.apply_chat_template,
    fewshot_as_multiturn=args.fewshot_as_multiturn)


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

for key in results['results']:
        print(key)
        print()
        print(results['results'][key])
        print()
        print()