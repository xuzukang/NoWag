CUDA_LAUNCH_BLOCKING = 1
PYTORCH_NO_CUDA_MEMORY_CACHING = 0
import torch
import pynvml
import time
import argparse
import os
import datetime



args = argparse.ArgumentParser(description="Get GPU information and log it")
args.add_argument("--frequncy", type=int, default=1, help="frequency of checking the memory in minutes")
args.add_argument("--log_length", type=int, default=60*12, help="length of the log in minutes")
args.add_argument("--log_file", type=str, default="gpu_log.txt", help="log file name")
args = args.parse_args()

def clear_log(log_file:str):

    # clear the log file
    if "/" in log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        f.write("")


def log_gpu_info(log_file:str):
    text = ""
    text += "="*10 + " " + str(datetime.datetime.now()) + " " + "="*10 + "\n"
    for i in range(torch.cuda.device_count()):
        
        with torch.cuda.device(i):
            try:
                text += f"cuda:{i} {torch.cuda.get_device_name(i)}: "
                # get the amount of free memory
                free, total = torch.cuda.mem_get_info(i)
                t = torch.cuda.temperature(i)
                text += f"{free // 1024**2} MiB free out of {total // 1024**2} MiB total, temperature: {t} C"
                # print("\n")
            except:
                t += "GPU info not available"

            text += "\n"
    text += "\n"

    with open(log_file, "a") as f:
        f.write(text)
    

while True:
    clear_log(args.log_file)
    for i in range(args.log_length):
        log_gpu_info(args.log_file)
        time.sleep(args.frequncy * 60)