CUDA_LAUNCH_BLOCKING = 1
PYTORCH_NO_CUDA_MEMORY_CACHING = 0
import torch
import pynvml


# CUDA_LAUNCH_BLOCKING=1
# get the info for each cuda
def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024**2


for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        try:
            print(f"cuda:{i}", torch.cuda.get_device_name(i), end=" ")
            # get the amount of free memory
            free, total = torch.cuda.mem_get_info(i)
            print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")
            # print("\n")
        except:
            print("cuda is not available")