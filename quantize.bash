Namespace(models_to_compress=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-70B'], seqlens=[4096, 4096, 4096, 8192, 8192], batch_size=1, hessian_path='/data/lliu/huffman/models/{model_name}/hessians_new/pajama/128/', save_path='/data/lliu/huffman/models/{model_name}/compressed', self_attn_compression_algorithm='quantize', mlp_compression_algorithm='quantize', devices=['cuda:5', 'cuda:6', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:7'], yaml_path='/data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml', self_attn_yaml_path=None, mlp_yaml_path=None, use_already_done=True, use_wandb=True, wandb_project='compression_no_finetune')
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_3/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.gate_proj/compressed.log
best_loss 49.43342208862305 running bpv: 2.009131
already done with  meta-llama/Llama-2-7b-hf/layer_3/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_3/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.up_proj/compressed.log
best_loss 41.891727447509766 running bpv: 2.009131
already done with  meta-llama/Llama-2-7b-hf/layer_3/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.q_proj/compressed.log
best_loss 112.32766723632812 running bpv: 2.009997
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.k_proj/compressed.log
best_loss 122.52682495117188 running bpv: 2.010628
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.o_proj/compressed.log
best_loss 0.24364137649536133 running bpv: 2.011109
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_3/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/mlp.down_proj/compressed.log
best_loss 0.8539788126945496 running bpv: 2.010242
already done with  meta-llama/Llama-2-7b-hf/layer_3/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_3/self_attn.v_proj/compressed.log
best_loss 31.879804611206055 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_3/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_5/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.gate_proj/compressed.log
best_loss 88.11199951171875 running bpv: 2.010339
already done with  meta-llama/Llama-2-7b-hf/layer_5/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_5/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.up_proj/compressed.log
best_loss 69.80905151367188 running bpv: 2.010153
already done with  meta-llama/Llama-2-7b-hf/layer_5/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.q_proj/compressed.log
best_loss 121.28446960449219 running bpv: 2.010397
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.k_proj/compressed.log
best_loss 132.9383544921875 running bpv: 2.010616
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.o_proj/compressed.log
best_loss 0.8941723704338074 running bpv: 2.010813
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_5/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/mlp.down_proj/compressed.log
best_loss 2.309817314147949 running bpv: 2.010433
already done with  meta-llama/Llama-2-7b-hf/layer_5/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_5/self_attn.v_proj/compressed.log
best_loss 36.62811279296875 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_5/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_8/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.gate_proj/compressed.log
best_loss 131.2567901611328 running bpv: 2.01046
already done with  meta-llama/Llama-2-7b-hf/layer_8/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_8/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.up_proj/compressed.log
best_loss 106.80435180664062 running bpv: 2.010339
already done with  meta-llama/Llama-2-7b-hf/layer_8/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.q_proj/compressed.log
best_loss 191.25511169433594 running bpv: 2.01048
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.k_proj/compressed.log
best_loss 193.88648986816406 running bpv: 2.010612
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.o_proj/compressed.log
best_loss 2.943741798400879 running bpv: 2.010737
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_8/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/mlp.down_proj/compressed.log
best_loss 5.640310287475586 running bpv: 2.010493
already done with  meta-llama/Llama-2-7b-hf/layer_8/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_8/self_attn.v_proj/compressed.log
best_loss 59.480464935302734 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_8/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_12/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.gate_proj/compressed.log
best_loss 162.8937225341797 running bpv: 2.010506
already done with  meta-llama/Llama-2-7b-hf/layer_12/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_12/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.up_proj/compressed.log
best_loss 148.902099609375 running bpv: 2.010417
already done with  meta-llama/Llama-2-7b-hf/layer_12/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.q_proj/compressed.log
best_loss 240.7495574951172 running bpv: 2.010516
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.k_proj/compressed.log
best_loss 257.9990539550781 running bpv: 2.010611
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.o_proj/compressed.log
best_loss 6.915116310119629 running bpv: 2.010702
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_12/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/mlp.down_proj/compressed.log
best_loss 10.196646690368652 running bpv: 2.010522
already done with  meta-llama/Llama-2-7b-hf/layer_12/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_12/self_attn.v_proj/compressed.log
best_loss 86.48735046386719 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_12/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_1/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.gate_proj/compressed.log
best_loss 12.91602897644043 running bpv: 2.01053
already done with  meta-llama/Llama-2-7b-hf/layer_1/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_1/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.up_proj/compressed.log
best_loss 11.325397491455078 running bpv: 2.01046
already done with  meta-llama/Llama-2-7b-hf/layer_1/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.q_proj/compressed.log
best_loss 6.516097068786621 running bpv: 2.010536
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.k_proj/compressed.log
best_loss 6.616118431091309 running bpv: 2.01061
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.o_proj/compressed.log
best_loss 0.0628652349114418 running bpv: 2.010682
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_1/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/mlp.down_proj/compressed.log
best_loss 0.21109314262866974 running bpv: 2.010539
already done with  meta-llama/Llama-2-7b-hf/layer_1/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_1/self_attn.v_proj/compressed.log
best_loss 0.6918058395385742 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_1/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_7/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.gate_proj/compressed.log
best_loss 126.69721221923828 running bpv: 2.010545
already done with  meta-llama/Llama-2-7b-hf/layer_7/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_7/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.up_proj/compressed.log
best_loss 97.25100708007812 running bpv: 2.010487
already done with  meta-llama/Llama-2-7b-hf/layer_7/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.q_proj/compressed.log
best_loss 194.13951110839844 running bpv: 2.010549
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.k_proj/compressed.log
best_loss 196.28851318359375 running bpv: 2.01061
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.o_proj/compressed.log
best_loss 1.8761067390441895 running bpv: 2.010669
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_7/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/mlp.down_proj/compressed.log
best_loss 4.567765235900879 running bpv: 2.010551
already done with  meta-llama/Llama-2-7b-hf/layer_7/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_7/self_attn.v_proj/compressed.log
best_loss 58.0523567199707 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_7/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_0/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.gate_proj/compressed.log
best_loss 2.9977715015411377 running bpv: 2.010555
already done with  meta-llama/Llama-2-7b-hf/layer_0/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_0/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.up_proj/compressed.log
best_loss 2.8641300201416016 running bpv: 2.010506
already done with  meta-llama/Llama-2-7b-hf/layer_0/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.q_proj/compressed.log
best_loss 0.36603060364723206 running bpv: 2.010558
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.k_proj/compressed.log
best_loss 0.2853046655654907 running bpv: 2.01061
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.o_proj/compressed.log
best_loss 0.005404493305832148 running bpv: 2.01066
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_0/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/mlp.down_proj/compressed.log
best_loss 0.02371348813176155 running bpv: 2.010559
already done with  meta-llama/Llama-2-7b-hf/layer_0/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_0/self_attn.v_proj/compressed.log
best_loss 0.05582483485341072 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_0/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_31/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.gate_proj/compressed.log
best_loss 585.796630859375 running bpv: 2.010562
already done with  meta-llama/Llama-2-7b-hf/layer_31/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_31/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.up_proj/compressed.log
best_loss 518.9898681640625 running bpv: 2.010519
already done with  meta-llama/Llama-2-7b-hf/layer_31/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.q_proj/compressed.log
best_loss 270.0703430175781 running bpv: 2.010565
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.k_proj/compressed.log
best_loss 292.4201354980469 running bpv: 2.010609
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.o_proj/compressed.log
best_loss 42.25667953491211 running bpv: 2.010653
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_31/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/mlp.down_proj/compressed.log
best_loss 284.68499755859375 running bpv: 2.010565
already done with  meta-llama/Llama-2-7b-hf/layer_31/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_31/self_attn.v_proj/compressed.log
best_loss 200.59046936035156 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_31/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_21/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.gate_proj/compressed.log
best_loss 399.2372131347656 running bpv: 2.010568
already done with  meta-llama/Llama-2-7b-hf/layer_21/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_21/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.up_proj/compressed.log
best_loss 341.3741760253906 running bpv: 2.01053
already done with  meta-llama/Llama-2-7b-hf/layer_21/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.q_proj/compressed.log
best_loss 308.9530334472656 running bpv: 2.01057
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.k_proj/compressed.log
best_loss 317.4786071777344 running bpv: 2.010609
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.o_proj/compressed.log
best_loss 11.116890907287598 running bpv: 2.010648
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_21/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/mlp.down_proj/compressed.log
best_loss 43.497642517089844 running bpv: 2.01057
already done with  meta-llama/Llama-2-7b-hf/layer_21/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_21/self_attn.v_proj/compressed.log
best_loss 194.8931121826172 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_21/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.gate_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_26/mlp.gate_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.gate_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.gate_proj/compressed.log
best_loss 537.7118530273438 running bpv: 2.010572
already done with  meta-llama/Llama-2-7b-hf/layer_26/mlp.gate_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.up_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_26/mlp.up_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.up_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.up_proj/compressed.log
best_loss 462.7989807128906 running bpv: 2.010538
already done with  meta-llama/Llama-2-7b-hf/layer_26/mlp.up_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.q_proj/compressed.log
best_loss 363.3192138671875 running bpv: 2.010574
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.k_proj/compressed.log
best_loss 371.9248962402344 running bpv: 2.010609
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.o_proj/compressed.log
best_loss 21.499774932861328 running bpv: 2.010644
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.o_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.down_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_26/mlp.down_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.down_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/mlp.down_proj/compressed.log
best_loss 64.91046905517578 running bpv: 2.010574
already done with  meta-llama/Llama-2-7b-hf/layer_26/mlp.down_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.v_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.v_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.v_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_26/self_attn.v_proj/compressed.log
best_loss 289.975830078125 running bpv: 2.010608
already done with  meta-llama/Llama-2-7b-hf/layer_26/self_attn.v_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.q_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_14/self_attn.q_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.q_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.q_proj/compressed.log
best_loss 257.5028076171875 running bpv: 2.010641
already done with  meta-llama/Llama-2-7b-hf/layer_14/self_attn.q_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.k_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_14/self_attn.k_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.k_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.k_proj/compressed.log
best_loss 269.93011474609375 running bpv: 2.010674
already done with  meta-llama/Llama-2-7b-hf/layer_14/self_attn.k_proj
path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.o_proj/compressed_args.yaml
yaml_args {'quantizer_kwargs': {'d': 5, 'n_bits': 2, 'cluster_ignore_norms': True, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'seed': 0}
other_args {'alignment_kwargs': {'clip_grad': 0.1, 'discrete_update_every': -1, 'low_bound': 1e-05, 'lr': 0.001, 'lr_multiplier': 0.3333, 'n_iters': 100, 'patience': 100, 'patience_scheduler': 10000, 'verbose': 10}, 'compression_type': 'quantized', 'quantizer_kwargs': {'cluster_ignore_norms': True, 'd': 5, 'n_bits': 2, 'n_iters': 100, 'norm_order': [0, 1], 'zero': [False, False]}, 'seed': 0}
is_same True
already done with  meta-llama/Llama-2-7b-hf/layer_14/self_attn.o_proj
loading from /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.o_proj/compressed.pt
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/gallant-elevator-52/meta-llama/Llama-2-7b-hf/layer_14/self_attn.o_proj/compressed.log
best_loss 9.668866157531738 running bpv: 2.010706
already done with  meta-llama/Llama-2-7b-hf/layer_14/self_attn.o_proj
n_commands 1775
sample command python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_14/mlp.gate_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.gate_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_14/mlp.gate_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.gate_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:5 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.gate_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_14/mlp.up_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.up_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:6 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.up_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_14/mlp.down_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.down_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:2 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.down_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_14/self_attn.v_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/self_attn.v_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:3 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/self_attn.v_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_15/mlp.gate_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.gate_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:4 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.gate_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_15/mlp.up_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.up_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:7 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.up_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_14/self_attn.v_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/self_attn.v_proj/compressed.log
best_loss 97.36930847167969 running bpv: 2.010737
COMMANDS_FINISHED 1 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_15/self_attn.q_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.q_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:3 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.q_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_14/mlp.gate_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.gate_proj/compressed.log
best_loss 188.02056884765625 running bpv: 2.010704
COMMANDS_FINISHED 2 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_15/self_attn.k_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.k_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:5 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.k_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_14/mlp.up_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.up_proj/compressed.log
best_loss 177.93357849121094 running bpv: 2.010671
COMMANDS_FINISHED 3 n_commands 1775
meta-llama/Llama-2-7b-hf/layer_14/mlp.down_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_14/mlp.down_proj/compressed.log
best_loss 14.621866226196289 running bpv: 2.010608
COMMANDS_FINISHED 4 n_commands 1775
meta-llama/Llama-2-7b-hf/layer_15/self_attn.q_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.q_proj/compressed.log
best_loss 240.59933471679688 running bpv: 2.010638
COMMANDS_FINISHED 5 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_15/self_attn.o_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.o_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:6 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.o_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_15/mlp.down_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.down_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:2 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.down_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_15/self_attn.v_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.v_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:3 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.v_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_15/mlp.gate_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.gate_proj/compressed.log
best_loss 207.4346160888672 running bpv: 2.010608
COMMANDS_FINISHED 6 n_commands 1775
meta-llama/Llama-2-7b-hf/layer_15/mlp.up_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.up_proj/compressed.log
best_loss 196.49502563476562 running bpv: 2.01058
COMMANDS_FINISHED 7 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_4/mlp.gate_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.gate_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:4 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.gate_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_4/mlp.up_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.up_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:7 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.up_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_15/self_attn.k_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.k_proj/compressed.log
best_loss 260.69268798828125 running bpv: 2.010609
COMMANDS_FINISHED 8 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_4/self_attn.q_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.q_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:5 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.q_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_15/self_attn.o_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.o_proj/compressed.log
best_loss 10.165802001953125 running bpv: 2.010637
COMMANDS_FINISHED 9 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_4/self_attn.k_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.k_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:6 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.k_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_15/self_attn.v_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/self_attn.v_proj/compressed.log
best_loss 101.61927795410156 running bpv: 2.010666
COMMANDS_FINISHED 10 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_4/self_attn.o_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.o_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:3 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.o_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_4/self_attn.q_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.q_proj/compressed.log
best_loss 107.25277709960938 running bpv: 2.010694
COMMANDS_FINISHED 11 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_4/mlp.down_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.down_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:5 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.down_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_4/self_attn.k_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.k_proj/compressed.log
best_loss 112.94963836669922 running bpv: 2.010721
COMMANDS_FINISHED 12 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_4/self_attn.v_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.v_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:6 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.v_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_15/mlp.down_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_15/mlp.down_proj/compressed.log
best_loss 18.33635139465332 running bpv: 2.010663
COMMANDS_FINISHED 13 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_27/mlp.gate_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.gate_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:2 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.gate_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_4/self_attn.o_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.o_proj/compressed.log
best_loss 0.536568284034729 running bpv: 2.01069
COMMANDS_FINISHED 14 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_27/mlp.up_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.up_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:3 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.up_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_4/mlp.gate_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.gate_proj/compressed.log
best_loss 69.46524047851562 running bpv: 2.010662
COMMANDS_FINISHED 15 n_commands 1775
meta-llama/Llama-2-7b-hf/layer_4/mlp.up_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.up_proj/compressed.log
best_loss 55.66865158081055 running bpv: 2.010635
COMMANDS_FINISHED 16 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_27/self_attn.q_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.q_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:4 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.q_proj/compressed.log 2>&1 &
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_27/self_attn.k_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.k_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:7 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.k_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_4/self_attn.v_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/self_attn.v_proj/compressed.log
best_loss 31.05900764465332 running bpv: 2.010661
COMMANDS_FINISHED 17 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_27/self_attn.o_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.o_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:6 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.o_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_27/self_attn.q_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.q_proj/compressed.log
best_loss 403.30023193359375 running bpv: 2.010687
COMMANDS_FINISHED 18 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_27/mlp.down_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.down_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:4 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.down_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_27/self_attn.k_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.k_proj/compressed.log
best_loss 413.33197021484375 running bpv: 2.010712
COMMANDS_FINISHED 19 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_27/self_attn.v_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.v_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:7 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.v_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_4/mlp.down_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_4/mlp.down_proj/compressed.log
best_loss 1.5733578205108643 running bpv: 2.010659
COMMANDS_FINISHED 20 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_2/mlp.gate_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/mlp.gate_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:5 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/mlp.gate_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_27/self_attn.o_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/self_attn.o_proj/compressed.log
best_loss 19.273479461669922 running bpv: 2.010684
COMMANDS_FINISHED 21 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_2/mlp.up_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/mlp.up_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:6 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/mlp.up_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_27/mlp.gate_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.gate_proj/compressed.log
best_loss 572.5789794921875 running bpv: 2.010658
COMMANDS_FINISHED 22 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_2/self_attn.q_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/self_attn.q_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:2 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/self_attn.q_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_27/mlp.up_proj is done
reading log /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_27/mlp.up_proj/compressed.log
best_loss 496.35308837890625 running bpv: 2.010633
COMMANDS_FINISHED 23 n_commands 1775
running: nohup python -u scripts/1layer_compress/quantize_compress.py --load_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/pajama/128/layer_2/self_attn.k_proj.pt --save_path /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/self_attn.k_proj/compressed.pt --yaml_path /data/lliu/huffman/scripts/1layer_compress/quantizer_args.yaml --device cuda:3 > /data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/compressed/glorious-tree-53/meta-llama/Llama-2-7b-hf/layer_2/self_attn.k_proj/compressed.log 2>&1 &
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-hf/layer_27/self_attn.v_proj is done
reading log                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     