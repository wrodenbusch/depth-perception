from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- Web-L (the long-context web cell, cell 7): sequence + TREE GRID (depth + own sibling
# rank, 4 planes). Same geometry and schedule as every cell-7 arm; the launch command supplies
# the pack (web_text_full_v3_drank), the 537,000-step schedule and the rest, exactly as for
# rope_both_p1 and rope_depth_p1.
#
# WHY. On Web-S the tree grid's first seed beat sequence + depth and the full path; cell 7 is
# the paper's long-context setting, single-seed, so this arm puts the tree grid on the same
# footing as rope_seq / rope_both / rope_depth there.
out_dir = f'{RUNS}/web7/drank_p1'
pack = f'{PACKS}/web_text_full_v3_drank'
pos_mode = 'rope_both'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 10, 4, 512   # head dim 128; 31.5M non-embedding -- throughput.py choice (67.7k tok/s on an A4000)
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 2, 2   # 16 seq + [depth, own rank] x 2 + 44 idle
rope_token_planes = 44                  # IDLE under rope_both; fills the allocation
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
