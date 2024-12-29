import os 
import sys
import argparse
import yaml
import glob
import time
import tqdm
import psutil

parser =  argparse.ArgumentParser()
parser.add_argument("--models_to_compress", type = str, nargs = "+",
                    default = ["meta-llama/Llama-2-7b-hf"],
                    help = "list of models to compress")
parser.add_argument("--seqlens", type = int, nargs = "+", default = [4096],
                    help = "list of seqlens to compress")
parser.add_argument("--batch_size", type = int, default = 1,)
parser.add_argument("--hessian_path", type = str, default = "/data/lliu/huffman/models/{model_name}/hessians_new/pajama/128/",
                    help = "path to the hessians")
parser.add_argument("--save_path", type = str, default = "/data/lliu/huffman/models/{model_name}/compressed",
                    help = "path to save the compressed models")
parser.add_argument("--self_attn_compression_algorithm", type = str, 
                    choices = ["tensor", "quantize", "joint"],
                    default = "quantize",
                    help = "algorithm to use for self attention compression")
parser.add_argument("--mlp_compression_algorithm",
                    type = str, choices=["tensor", "quantize", "joint"],
                    default = "quantize",
                    help = "algorithm to use for mlp compression")
parser.add_argument("--devices", type = str, nargs = "+", default = ["cuda:5", "cuda:6", "cuda:2", "cuda:3", "cuda:4","cuda:7"],
                    help = "list of devices to run the compression on if not provided will use all devices")
parser.add_argument("--yaml_path", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml")
parser.add_argument("--self_attn_yaml_path", type = str, default = None)
parser.add_argument("--mlp_yaml_path", type = str, default = None)
parser.add_argument("--use_already_done", action = "store_true", help = "if provided will overwrite the save path")
parser.add_argument("--use_wandb", action = "store_true", help = "if provided will use wandb")
parser.add_argument("--wandb_project", type = str, default = "compression")
args = parser.parse_args()

    
if args.use_wandb:
    import wandb
    wandb.init(project = args.wandb_project)
    config = vars(args)
    
    if args.self_attn_yaml_path is not None:
        config["self_attn_args"] = yaml.load(open(args.self_attn_yaml_path, "r"), Loader = yaml.FullLoader)
    else:
        config["self_attn_args"] = yaml.load(open(args.yaml_path, "r"), Loader = yaml.FullLoader)
    if args.mlp_yaml_path is not None:
        config["mlp_args"] = yaml.load(open(args.mlp_yaml_path, "r"), Loader = yaml.FullLoader)
    else:
        config["mlp_args"] = yaml.load(open(args.yaml_path, "r"), Loader = yaml.FullLoader)

    wandb.config.update(config)

    #append the run_name to the save_path
    run_name = wandb.run.name
    args.save_path = os.path.join(args.save_path, wandb.run.name)
else:
    #count the number of runs done
    n_runs = glob.glob(args.save_path + "/*")
    run_name = f"run_{len(n_runs)}"
    args.save_path = os.path.join(args.save_path, f"run_{len(n_runs)}")


# print("args.save_path", args.save_path)
# print(args.hessian_path + "/**/*.pt")
paths = []
path_map: dict[str,list[str]] = {} #maps which model has the following paths
inverse_path_map: dict[str, str] = {} #maps the path to the model
for model_name in args.models_to_compress:
    model_paths = glob.glob(args.hessian_path.format(model_name = model_name) + "/**/*.pt", recursive = True)
    paths.extend(model_paths)
    path_map[model_name] = model_paths
    for path in model_paths:
        inverse_path_map[path] = model_name

seqlen_map = {model_name: args.seqlens[i] for i, model_name in enumerate(args.models_to_compress)}


# print("paths", paths)

DEVICES_DICT = {device: [] for device in args.devices}
TOTAL_BITS = 0
TOTAL_PARAMS = 0    
COMMANDS_FINISHED = 0
BAR = tqdm.tqdm(total = len(paths))

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    print("checking pid", pid, psutil.pid_exists(pid))
    return psutil.pid_exists(pid)

def run_command(command:str, device:str,
                log_path:str,
                command_name:str):
    global DEVICES_DICT
    
    os.makedirs(os.path.dirname(log_path), exist_ok = True)
    os.system(f"nohup {command} --device {device} > {log_path} 2>&1 &")
    print(f"nohup {command} --device {device} > {log_path} 2>&1 &")
    
    time.sleep(5)
    #read the log path to get the pid
    
    with open(log_path, "r") as f:
        log = f.readlines()
        # print("log", log)
        for line in log:
            if "pid" in line:
                pid = int(line.split(" ")[-1])
                break
    
    DEVICES_DICT[device] = [pid, log_path, command_name]

def read_log(log_path:str):
    global TOTAL_BITS
    global TOTAL_PARAMS
    global COMMANDS_FINISHED

    print("reading log", log_path)
    with open(log_path, "r") as f:
                log = f.readlines()
                for line in log:
                    if "best loss" in line:
                        best_loss = float(line.split(" ")[-1])
                    if "n_params" in line:
                        TOTAL_PARAMS += float(line.split(" ")[-1])
                    if "n_bits" in line:
                        TOTAL_BITS += float(line.split(" ")[-1])
    print("best_loss", best_loss, "running bpv:", round(TOTAL_BITS/TOTAL_PARAMS, 6))
    BAR.update(1)
    COMMANDS_FINISHED += 1
    return True
    # except:
    #     print("error reading log")
    #     return False

def check_still_running(devices_dict)->list[str]:
    global TOTAL_BITS
    global TOTAL_PARAMS
    global COMMANDS_FINISHED
    
    DEVICES_OPEN = []
    for device, (pid, log_path, command_name) in devices_dict.items():
        if check_pid(pid):
            # print(f"{device} is still running")
            pass
        elif command_name == "eval":
            print("eval is done")
            DEVICES_OPEN.append(device)
        else:
            print(f"{command_name} is done")
            read_log(log_path)
            if log_path_map[log_path] not in DONE_SAVE_PATHS:
                DONE_SAVE_PATHS[log_path_map[log_path]] = {}
            DONE_SAVE_PATHS[log_path_map[log_path]][command_name] = log_path.replace("compressed.log", "compressed.pt")
            DEVICES_OPEN.append(device)
    return DEVICES_OPEN



def make_command(load_path:str,model_name:str)-> tuple[str,str,str]:
    global TOTAL_BITS
    global TOTAL_PARAMS
    
    command = "python -u scripts/1layer_compress/"
    command_name = load_path.split("/")[-2] +  "/" + load_path.split("/")[-1][:load_path.split("/")[-1].rfind(".")]
    print("command_name", command_name)

    if "mlp" in load_path:
        compression_algorithm = args.mlp_compression_algorithm
        yaml_path = args.mlp_yaml_path if args.mlp_yaml_path is not None else args.yaml_path
    elif "self_attn" in load_path:
        compression_algorithm = args.self_attn_compression_algorithm
        yaml_path = args.self_attn_yaml_path if args.self_attn_yaml_path is not None else args.yaml_path

    if compression_algorithm == "tensor":
        command += "tensor_compress.py"
    elif compression_algorithm == "joint":
        command += "joint_compress.py"
    else:
        command += "quantize_compress.py"
    

    save_path = os.path.join(args.save_path.replace("{model_name}", model_name), command_name, "compressed.pt")

    command += f" --load_path {load_path} --save_path {save_path} --yaml_path {yaml_path}"
    log_path = save_path.replace(".pt", ".log")

    if args.use_already_done:
        #we check over the past logs to see if the command has already been run
        for path in glob.glob(save_path.replace(run_name, "*").repalce("compressed.pt", "compressed_args.yaml")):
            other_args = yaml.load(open(path, "r"), Loader = yaml.FullLoader)
            yaml_args = yaml.load(open(yaml_path, "r"), Loader = yaml.FullLoader)

            #check if all the keys in yaml_args are in other_args
            #we do not check the otherway because other_args may have more keys that we add in the compression script
            is_same = True
            for key in yaml_args.keys():
                if key not in other_args.keys():
                    is_same = False
                    break
                if yaml_args[key] != other_args[key]:
                    is_same = False
                    break
            if is_same:
                print("already done with ", command_name)
                if read_log(path.replace("compressed_args.yaml", "compressed.log")):
                    return None, None,  log_path
                else:
                    print("could not read log, rerunning")
                    break
    return command, command_name, log_path


commands = []
command_names = []
log_paths = []
log_path_map = {} #maps the log path to the model
DONE_SAVE_PATHS:dict[str:dict[str,str]] = {} #model and a list of done log paths

for path in paths:
    command, command_name,log_path = make_command(path, inverse_path_map[path])
    if command is not None:
        commands.append(command)
        command_names.append(command_name)
        log_paths.append(log_path)
        log_path_map[log_path] = inverse_path_map[path]
    else:
        if inverse_path_map[path] not in DONE_SAVE_PATHS:
            DONE_SAVE_PATHS[inverse_path_map[path]] = {}
        DONE_SAVE_PATHS[inverse_path_map[path]][command_name] = log_path.replace("compressed.log", "compressed.pt")
n_commands = len(commands)
print("n_commands", n_commands)
print("sample command", commands[0])
    # raise ValueError("stop here")

# print("commands", commands[2])
# raise ValueError("stop here")

#run the first len(args.devices) commands

DEVICES_OPEN = args.devices.copy()


for i in range(len(args.devices)):
    command = commands.pop(0)
    command_name = command_names.pop(0)
    log_path = log_paths.pop(0)
    run_command(command, args.devices[i], log_path, command_name)
    
    
while COMMANDS_FINISHED < n_commands:
    
    time.sleep(10)
    DEVICES_OPEN = check_still_running(DEVICES_DICT)
    if len(commands) > 0:
        for device in DEVICES_OPEN:
            command = commands.pop(0)
            command_name = command_names.pop(0)
            log_path = log_paths.pop(0)
            run_command(command, device, log_path, command_name)
    
    for key in DONE_SAVE_PATHS.keys():
        if len(DONE_SAVE_PATHS[key]) == len(path_map[key]):
            print(f"done with {key}")
            print("done with", DONE_SAVE_PATHS[key])
            #save the save paths to a folder
            chekpoint_list_path = os.path.join(args.save_path.replace("{model_name}", key), "checkpoints.yaml")
            os.makedirs(os.path.dirname(chekpoint_list_path), exist_ok = True)
            yaml.dump(DONE_SAVE_PATHS[key], open(chekpoint_list_path, "w"))


            # perplexity_inference_command = f"python -u scripts/1layer_compress/perplexity_inference.py --model_name {key} --seqlen {seqlen_map[key]} --chekpoint_list_path {chekpoint_list_path}"
            # if args.use_wandb:
            #     perplexity_inference_command += f" --use_wandb --wandb_project {args.wandb_project} --wandb_id {wandb.run.id}"




    
print("done")
    
            
        

                        
                        







