from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS web cell -- arm rope_graph (only the tree planes rotate: DOM position, no sequence position).
# Width / budget are filled by throughput.py; learning_rate by the LR smoke.
# Override per run: --seed=N --out_dir=$TOPOS_RUNS/web/rope_graph_sN
out_dir = f'{RUNS}/web/rope_graph_s0'
pack = f'{PACKS}/web_c4html_text_gpt2_b4096'
pos_mode = 'rope_graph'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 10, 4, 512   # head dim 128; 31.5M non-embedding -- throughput.py choice (67.7k tok/s on an A4000)
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 24, 2   # 16 + 24*2 = 64 = head_dim/2
bias = False
dropout = 0.0
batch_size = 2                          # micro-batch: 4 OOMed on the graph arms at a full 4096-token batch (14.1 GB + 1.5 GB logits)
gradient_accumulation_steps = 16        # 2 x 16 x 4096 = 131k tokens/step, unchanged
learning_rate = 6e-4                  # LR smoke, 352 steps: lowest final val nats/tok
min_lr = 6.0e-05
max_iters = 7040                        # one pass: 923M train tokens / 131,072 per step (pack meta: 595,904 records x 1,549 tokens)
lr_decay_iters = 7040
warmup_iters = 141                      # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 352                     # 5% of max_iters -> 20 milestones
eval_records = 1024
log_interval = 20
compile = True
