from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS web cell 2 -- arm rope_graph (only the tree planes rotate: DOM position, no sequence position).
# Width / budget are filled by throughput.py; learning_rate by the LR smoke.
# Override per run: --seed=N --out_dir=$TOPOS_RUNS/web2/rope_graph_sN
out_dir = f'{RUNS}/web2/rope_graph_s0'
pack = f'{PACKS}/web_c4html_text_gpt2_b4096_v2'
pos_mode = 'rope_graph'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 10, 4, 512   # head dim 128; 31.5M non-embedding -- throughput.py choice (67.7k tok/s on an A4000)
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 22*2 + 4 token planes = 64 = head_dim/2
rope_token_planes = 4                   # the token-leaf level; rotates only under rope_leaf, angle 0 elsewhere
bias = False
dropout = 0.0
batch_size = 2                          # micro-batch: 4 OOMed on the graph arms at a full 4096-token batch (14.1 GB + 1.5 GB logits)
gradient_accumulation_steps = 16        # 2 x 16 x 4096 = 131k tokens/step, unchanged
learning_rate = 6e-4                  # LR smoke, 1360 steps: lowest final val nats/tok
min_lr = 6.0e-05
max_iters = 27220                       # 4 passes: n_train 595,904 x 1496.3 tok/rec x 4 / 131,072 tok/step = 27210, rounded to 20
lr_decay_iters = 27220
warmup_iters = 544                      # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 1361                    # 5% of max_iters -> 20 milestones, every one kept
eval_records = 1024
log_interval = 20
compile = True
