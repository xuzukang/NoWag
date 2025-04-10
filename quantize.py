import multiprocessing as mp 
import os
import glob
import argparse
import yaml
import queue
import torch
import torch.nn as nn
import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

import accelerate
import traceback
from transformers import LlamaForCausalLM as OrigLlama
from transformers import AutoConfig
from typing import Dict, Any, List, Tuple, Optional, Callable

from src.model.llama import LlamaForCausalLM
from src.utils.utils import *
from src.data import get_loaders
from src.eval import ppl, zero_shot
from src.quantize_compress import LinearVQ
from accelerate import infer_auto_device_map, dispatch_model
import numpy as np

def quantization_worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    gpu_queue: mp.Queue,
    lock: 'mp.synchronize.Lock',
    stop_event: 'mp.synchronize.Event',
):
    """Quantizes a layer of the model

    Args:
        task_queue (mp.Queue): task queue, each element consits of a tuple of (layer_name[str],
        config[DictConfig])
        result_queue (mp.Queue): the results queue, each element is a tuple of 
        (
                    layer_name[str], 
                    save_path[str], 
                    n_bits[int],
                    n_params[int],
        )
        gpu_queue (mp.Queue): a queue of the available GPU ids
        lock (mp.synchronize.Lock): a lock for accessing the GPU queue
        stop_event (mp.Event): event to stop the worker if a exception is raised
    """
    
    while True:
        try:
            # Get next task (non-blocking)
            layer_name,quantization_config = task_queue.get_nowait()
            #layer name is expected to be of the form layer_{i}/{self_attn|mlp}.{q|k|v|o|up|down|gate}_proj
        except queue.Empty:
            # No more tasks, exit
            return
            
        # Get an available GPU
        with lock:
            gpu_id = gpu_queue.get()
            
        try:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            seed(quantization_config.seed)
            
            #load the original weight
            weight = torch.load(os.path.join(quantization_config.weight_path, layer_name+".pt"),
                                map_location = device)["weight"]
            original_dtype = weight.dtype
            
            weight = weight.to(torch.float32)
            
            #create the compression module
            compression_module =  LinearVQ(weight=weight,
                                           add_bias = quantization_config.add_bias)
            
            #load the hessian
            if hasattr(quantization_config, "hessian_path"):
                hessian = torch.load(os.path.join(quantization_config.hessian_path, layer_name+".pt"),
                                    map_location = device)["hessian"]
                compression_module.hessian = hessian.to(torch.float32)
            if hasattr(quantization_config, "hessianDiag_path"):
                hessianDiag = torch.load(os.path.join(quantization_config.hessianDiag_path, layer_name+".pt"),
                                        map_location = device)["hessianDiag"]
                compression_module.hessianDiag = hessianDiag.to(torch.float32)
                
            else:
                raise ValueError("hessian not found in the hessian file")
            

            #compress the layer
            compression_module.compress(**quantization_config.compression_kwargs)
            compression_module.to(original_dtype)
            
            print(f"Compression module {layer_name} created, average unweighted l2 distortion: {compression_module.get_reconstruction_error()}", flush=True)
            #save the state dict 
            state_dict = compression_module.state_dict()
            
            os.makedirs(os.path.dirname(os.path.join(quantization_config.temp_path, layer_name + ".pt")), exist_ok=True)
            torch.save(state_dict, os.path.join(quantization_config.temp_path, layer_name + ".pt"))
            
            #calculate the number of parameters
            n_params = compression_module.get_n_original_parameters()
            n_bits = compression_module.get_n_bits()
        except Exception as e:
            print(f"========================= Error in quantization of {layer_name} =========================")
            #print the error and the traceback
            print(e)
            traceback.print_exc()
            # Set the stop event to signal the main process
            stop_event.set()
            raise e
        finally:
            # Put the GPU back in the queue
            with lock:
                gpu_queue.put(gpu_id)
                
            # Put the result in the result queue
            result_queue.put((layer_name, os.path.join(quantization_config.temp_path, layer_name + ".pt"), n_bits, n_params))
            
            
@hydra.main(version_base=None, config_path="./config", config_name="quantization")
def main(cfg: DictConfig):  
    
    seed(cfg.seed)
    # Get the list of available GPUs
    devices = torch.cuda.device_count()
    gpu_ids = list(range(devices))
    print(f"Available GPUs: {gpu_ids}")
    
    # Create queues for tasks and results
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    gpu_queue = mp.Queue()
    lock = mp.Lock()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
        
    
    #create our list of tasks
    weight_paths = glob.glob(os.path.join(cfg.weight_path, "*/*.pt"))
    print("n weights found", len(weight_paths))
    
    #create the task queue
    for weight_path in weight_paths:
        layer_name = weight_path.replace(cfg.weight_path, "").replace(".pt", "")[1:]
        task_queue.put((layer_name, cfg))
    
    # Create a stop event
    stop_event = mp.Event()
    
    # Create a pool of workers
    num_workers = min(len(gpu_ids), mp.cpu_count())
    processes = []
    print(f"Starting {num_workers} workers")
    for _ in range(num_workers):
        p = mp.Process(target=quantization_worker, args=(task_queue, result_queue, gpu_queue, lock, stop_event))
        p.start()
        processes.append(p)

    # Create a progress bar
    pbar = tqdm.tqdm(total=len(weight_paths), desc="Quantizing layers")
    
    checkpoints_dict: Dict[str, str] = {}
    running_bits = 0
    running_params = 0
    
    # Process results
    tasks_done = 0
    while tasks_done < len(weight_paths):
        try:
            layer_name, save_path, n_bits, n_params = result_queue.get(timeout=1)
            #print(f"Layer {layer_name} quantized and saved to {save_path}")
            checkpoints_dict[layer_name] = save_path
            running_bits += n_bits
            running_params += n_params
            tasks_done += 1
            pbar.update(1)
        except queue.Empty:
            pass
        
        if stop_event.is_set():
            print("Stopping all workers due to error")
            break
    pbar.close()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
        
    # Check if any process raised an exception
    if stop_event.is_set():
        print("One or more processes raised an exception. Exiting.")
        for p in processes:
            p.terminate()
        return
    
    #print out the results
    print("="*10, "Compression Level", "="*10)
    print(f"Total number of parameters: {human_format(running_params)}")
    print(f"Total number of bits: {human_format(running_bits)}")
    print(f"Average bits per parameter: {round(running_bits/running_params, 4)}")
    print(f"="*25)
    
    # Reload and Save in hf format
    
    orig_config = AutoConfig.from_pretrained(cfg.base_model,
                                            dtype = "auto",
                                            device_map="cpu",
                                            attn_implementation='sdpa')
    orig_model = OrigLlama.from_pretrained(cfg.base_model,
                                           config=orig_config, 
                                           torch_dtype="auto",
                                            device_map="cpu",
                                            low_cpu_mem_usage=True, attn_implementation='sdpa')
    
    
    compression_config = {"compression_kwargs": OmegaConf.to_container(cfg.compression_kwargs, resolve=True),
                            "compression_type": "LinearVQ", #compression_type,
                            "add_bias": cfg.add_bias, "skip_list":None}
    
    orig_config.compress_config = compression_config
    
    quantized_model = LlamaForCausalLM(orig_config)
    quantized_model.to(orig_config.torch_dtype)
    quantized_model.load_state_dict(orig_model.state_dict(), strict=False)
    
    del orig_model
    
    for layer_name, save_path in tqdm.tqdm(checkpoints_dict.items(), desc="Loading quantized layers"):
        #now split by /
        layer_name = layer_name.split("/")[-2:]
        #from the first part, we can get which layer it is
        i_layer = int(layer_name[0].replace("layer_", ""))
        #from the second part we can get which module (self_attn, mlp, etc) and which layer it is
        submodule_name, linear_name = layer_name[1].split(".")
        
        #now we get the right module
        layer = getattr(getattr(quantized_model.model.layers[i_layer], submodule_name), linear_name)
        #record the original dtype
        orig_dtype = layer.codebook.dtype
        orig_device = layer.codebook.device
        #load the state dict
        state_dict = torch.load(save_path, map_location=orig_device)
        layer.load_state_dict(state_dict)
        layer.to(orig_dtype)
        #delete the state dict to save memory
        del state_dict
        #and delete the save path from disk
        os.remove(save_path)
        
    #save the model
    quantized_model.save_pretrained(os.path.join(cfg.save_path, "model"))
    
    # Move the model to an auto device map using accelerate

    device_map = infer_auto_device_map(
        quantized_model
    )
    quantized_model = dispatch_model(quantized_model, device_map=device_map)    
    #evaluate the models
    #first have them cache the quantized weights
    recursive_apply(quantized_model, "cache_reconstruct", {'denormalize': True})
    
    #get the name of the model
    quantized_model.to(torch.float16)
    quantized_model.seqlen = cfg.seqlen if cfg.seqlen > 0 else orig_config.max_position_embeddings
    
    quantized_model.eval()
    if hasattr(cfg, "eval"):
        #ppl eval
        for dataset in cfg.eval.ppl_dataset:
            seed(cfg.seed, seed_all = True)
            testloader = get_loaders(
                dataset, nsamples = 0, seqlen = quantized_model.seqlen, model = cfg.base_model,
                train_test = "test")
            
            ppl.ppl_eval_basic(
                model = quantized_model, 
                testenc = testloader, 
                dataset_name = dataset, 
                results_log_yaml = os.path.join(cfg.save_path, "results.yaml")
            )
            
        #zero shot eval
        if len(cfg.eval.zero_shot_tasks) > 0:
            seed(cfg.seed,seed_all = True)
            zero_shot.zero_shot(
                cfg.base_model, quantized_model,
                tasks = cfg.eval.zero_shot_tasks,
                results_log_yaml = os.path.join(cfg.save_path, "results.yaml")
            )
            
    
if __name__ == "__main__":
    main()
                    
            
            
        
        
        
    
    
    
    
    
        
    
    
    
    
            
        
        
            
        
        
            