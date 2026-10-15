# TOPOS REPRODUCTION CELL — arm 'tree', seed 0. NOT YET RUN.
#
# Purpose: verify this repository reproduces the reference implementation's
# numbers before the paper cites them. Every value below is taken from the
# reference run's own summary.json, so a divergence is a port defect and not a
# configuration difference.
#
# Reference (arm 'tree' = pos_mode tree_add_nopos, seed 0):
#     n_params 13,969,540 = wte 8,519,680 + tree tables 561,860 + 4,888,000
#     lr 5e-4 constant, NO WARMUP, weight_decay 0.0, grad_clip 1.0
#     batch 16, block 1024, vocab 32,768, 125,000 steps = 2,000,000 examples
#
# THE PEAK IS AT 600,000 EXAMPLES = STEP 37,500. The reference run continued to
# 2M and got monotonically worse, and the floor's ordering inverts between those
# points -- so this cell runs past the peak to see it turn, and no further.
out_dir = 'out-repro-tree-s0'
dataset = 'packed_windows_apps_kodcode_astcompact'
pos_mode = 'tree'
seed = 0

n_layer = 6
n_head = 10          # head dim 26
n_embd = 260
block_size = 1024
vocab_size = 32768
bias = True
dropout = 0.0

batch_size = 16
gradient_accumulation_steps = 1
learning_rate = 5e-4
decay_lr = False     # constant, no warmup -- the reference used no schedule
weight_decay = 0.0
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95

# 50,000 steps = 800,000 examples: past the 37,500-step peak with margin.
max_iters = 50000
eval_interval = 1250     # every 20,000 examples
eval_iters = 200
log_interval = 50
always_save_checkpoint = True
