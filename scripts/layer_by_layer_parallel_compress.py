import os 
import sys
import argparse
import yaml
import glob
import time
import tqdm

parser =  argparse.ArgumentParser()

parser.add_argument("hessian_path", type = str)
parser.add_argument("save_path", type = str)
parser.add_argument("--self_attn_compression_algorithm", type = str, 
                    choices = ["tensor", "quantize", "joint"],
                    default = "quantize"
                    )
parser.add_argument("--mlp_compression_algorithm",
                    type = str, choices=["tensor", "quantize", "joint"],
                    default = "quantize"
                    )
parser.add_argument("--devices", type = str, nargs = "+", default = ["cuda:5", "cuda:6", "cuda:2", "cuda:3", "cuda:4","cuda:7"],
                    help = "list of devices to run the compression on if not provided will use all devices")
parser.add_argument("--log_dir", type = str, default = "./logs/parallel_compress")
parser.add_argument("--tensor_compress_kwargs_path", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/tensor_args.yaml")
parser.add_argument("--quantize_compress_kwargs_path", type = str, default = "/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml")
parser.add_argument("--overwrite", action = "store_true", help = "if provided will overwrite the save path")
args = parser.parse_args()

def create_parser_str(kwargs:str):
    
    with open(kwargs, "r") as f:
        kwargs = yaml.load(f, Loader = yaml.FullLoader)
        
    parser_str = ""
    for key, value in kwargs.items():
        if type(value) == list:
            value = " ".join([str(v) for v in value])
        parser_str += f" --{key} {value} "
        
    return parser_str

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True
    


# print("args.save_path", args.save_path)
# print(args.hessian_path + "/**/*.pt")
paths = glob.glob(args.hessian_path + "/**/*.pt", recursive = True)
# print("paths", paths)

DEVICES_DICT = {device: [] for device in args.devices}
TOTAL_BITS = 0
TOTAL_PARAMS = 0    
COMMANDS_FINISHED = 0
BAR = tqdm.tqdm(total = len(paths))



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
    
def check_still_running(devices_dict)->list[str]:
    global TOTAL_BITS
    global TOTAL_PARAMS
    global COMMANDS_FINISHED
    
    devices_open = []
    for device, (pid, log_path, command_name) in devices_dict.items():
        if check_pid(pid):
            # print(f"{device} is still running")
            pass
        else:
            print(f"{command_name} is done")
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
            devices_open.append(device)
    return devices_open



def make_command(load_path:str):
    global TOTAL_BITS
    global TOTAL_PARAMS
    
    command = "python -u scripts/1layer_compress/"
    command_name = load_path[len(args.hessian_path)+1:]
    if "mlp" in load_path:
        if args.mlp_compression_algorithm == "tensor":
            kwargs_path = args.tensor_compress_kwargs_path
            command += "tensor_compress.py"
            save_path = os.path.join(args.save_path, "tensor", command_name)
        elif args.mlp_compression_algorithm == "joint":
            kwargs_path = args.quantize_compress_kwargs_path
            command += "joint_compress.py"
            save_path = os.path.join(args.save_path, "joint2", command_name)
        else:
            kwargs_path = args.quantize_compress_kwargs_path
            command += "quantize_compress.py"
            save_path = os.path.join(args.save_path, "quantize", command_name)
            # print("save_path", save_path)   
    elif "self_attn" in load_path:
        if args.self_attn_compression_algorithm == "tensor":
            kwargs_path = args.tensor_compress_kwargs_path
            command += "tensor_compress.py"
            save_path = os.path.join(args.save_path, "tensor", command_name)
        elif args.self_attn_compression_algorithm == "joint":
            kwargs_path = args.quantize_compress_kwargs_path
            command += "joint_compress.py"
            save_path = os.path.join(args.save_path, "joint2", command_name)
        else:
            kwargs_path = args.quantize_compress_kwargs_path
            command += "quantize_compress.py"
            save_path = os.path.join(args.save_path, "quantize", command_name)
    else:
        raise ValueError("mlp or self_attn not in load_path")
    # print("save_path", save_path)
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
    command += f" --load_path {load_path} --save_path {save_path}"
    if "joint" not in command:
        additional_args = create_parser_str(kwargs_path)
    else:
        additional_args = ""
        additional_args += " --quantizer_yaml " + args.quantize_compress_kwargs_path
        additional_args += " --tensorizer_yaml " + args.tensor_compress_kwargs_path
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
    devices_open = check_still_running(DEVICES_DICT)
    if len(commands) > 0:
        for device in devices_open:
            command = commands.pop(0)
            command_name = command_names.pop(0)
            run_command(command, device, args.log_dir, command_name)
    
print("done")
    
            
        

                        
                        







