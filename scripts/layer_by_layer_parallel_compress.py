import os 
import sys
import argparse
import yaml
import glob
import time
import tqdm

parser =  argparse.ArgumentParser()
parser.add_argument("--models_to_compress", type = str, default = "meta-llama/Llama-2-7b-hf",
                    help = "list of models to compress")
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
    args.save_path = os.path.join(args.save_path, wandb.run.name)
else:
    #count the number of runs done
    n_runs = glob.go.glob(args.save_path + "/*")
    args.save_path = os.path.join(args.save_path, f"run_{len(n_runs)}")


# print("args.save_path", args.save_path)
# print(args.hessian_path + "/**/*.pt")
paths = glob.glob(args.hessian_path + "/**/*.pt", recursive = True)
# print("paths", paths)

DEVICES_DICT = {device: [] for device in args.devices}
TOTAL_BITS = 0
TOTAL_PARAMS = 0    
COMMANDS_FINISHED = 0
BAR = tqdm.tqdm(total = len(paths))

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def run_command(command:str, device:str,
                log_dir:str,
                command_name:str):
    global DEVICES_DICT
    
    log_path = os.path.join(log_dir, f"{command_name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok = True)
    os.system(f"nohup {command} --device {device} > {log_path} 2>&1 &")
    
    time.sleep(5)
    #read the log path to get the pid
    
    with open(f"{log_dir}/{command_name}.log", "r") as f:
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
    DEVICES_OPEN.append(device)

def check_still_running(devices_dict)->list[str]:
    global TOTAL_BITS
    global TOTAL_PARAMS
    global COMMANDS_FINISHED
    
    DEVICES_OPEN = []
    for device, (pid, log_path, command_name) in devices_dict.items():
        if check_pid(pid):
            # print(f"{device} is still running")
            pass
        else:
            print(f"{command_name} is done")
            read_log(log_path)
    return DEVICES_OPEN



def make_command(load_path:str):
    global TOTAL_BITS
    global TOTAL_PARAMS
    
    command = "python -u scripts/1layer_compress/"
    command_name = load_path.split("/")[-2] +  "/" + load_path.split("/")[-1].split(".")[0] 

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
    

    save_path = os.path.join(args.save_path, command_name, "compressed.pt")

    command += f" --load_path {load_path} --save_path {save_path} --yaml_path {yaml_path}"

    log_path = os.path.join(args.log_dir, f"{command_name}.log")
    if os.path.exists(save_path) and not args.overwrite and os.path.exists(log_path):
        log_path = os.path.join(args.log_dir, f"{command_name}.log")
        print("already done with ", command_name)
        with open(log_path, "r") as f:
            log = f.readlines()
            for line in log:
                if "best loss" in line:
                    best_loss = float(line.split(" ")[-1])
                if "n_params" in line:
                    TOTAL_PARAMS += float(line.split(" ")[-1])
                if "n_bits" in line:
                    TOTAL_BITS += float(line.split(" ")[-1])
            print("log_path", log_path)
            # print(log)
            print("best_loss", best_loss, "running bpv:", round(TOTAL_BITS/TOTAL_PARAMS, 6))
        BAR.update(1)
        return None, None
    command += additional_args
    return command, command_name


commands = []
command_names = []  
for path in paths:
    command, command_name = make_command(path)
    if command is not None:
        commands.append(command)
        command_names.append(command_name)
n_commands = len(commands)
    # raise ValueError("stop here")

# print("commands", commands[2])
# raise ValueError("stop here")

#run the first len(args.devices) commands


for i in range(len(args.devices)):
    command = commands.pop(0)
    command_name = command_names.pop(0)
    run_command(command, args.devices[i], args.log_dir, command_name)
    
    
while COMMANDS_FINISHED < n_commands:
    
    time.sleep(10)
    DEVICES_OPEN = check_still_running(DEVICES_DICT)
    if len(commands) > 0:
        for device in DEVICES_OPEN:
            command = commands.pop(0)
            command_name = command_names.pop(0)
            run_command(command, device, args.log_dir, command_name)
    
print("done")
    
            
        

                        
                        







