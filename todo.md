# To-Do List
- [ ] CUDA to for VQ (inesh)
- [ ] Trellis Quantization (inesh)
- [ ] Finetuning (lawrence)
## [x] Refactoring
- [x] Make a parent class for all compression algorithms
- [x] Move Sparse and Quantization to their own files
- [x] Fix the layer by layer code (fixed for quantization only)
- [x] Fix the Sparse code
- [x] Make a joint class that works

## [ ] Zero-shot Compression
- [x] Low Rank to work
- [x] Structured FFN to work
- [ ] Replicate LoRAP results
- [x] non-determnistic pruning to work (does not work)
- [x] Convert hessian generation to importance generation (diagonal only)
- [x] 70B model 7d quantization
    - [ ] talk to GT about splitting up the weights (may not be necessary)
- [ ] Trellis Quantization
- [ ] Dynamic Allocation


## Fine-tuning
- [ ] Get finetuning script to work
- [ ] LoRA finetuning for 
- [ ] Pruned finetuning 

## Other Models
- [ ] Compress Deepseek
- [ ] Llama 3.1
- [ ] Mistral?

