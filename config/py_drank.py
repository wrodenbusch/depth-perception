from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- the "tree grid": sequence + TWO tree numbers, the node's DEPTH and its OWN SIBLING RANK.
#
# WHY. On spreadsheets the structural gain came entirely from a grid (row, column) on 6 planes,
# matching the full 44-plane path. The author's analogue for any tree: depth (vertical) and which
# child of its parent the node is (horizontal). Two numbers, 4 planes. Paired against
# sequence only, sequence + depth and the full path at the same seeds.
#
# Data: <pack>_drank from depth_rank_pack.py -- node_path replaced by [depth, own_rank], every
# other array a symlink, so the token stream is byte-identical to the source pack. rope_both
# with rope_levels = 2 rotates level 0 by depth and level 1 by own rank with the same ladder and
# presence handling the full path uses; the other 44 planes are idle.
out_dir = f'{RUNS}/py/drank_s0'
pack = f'{PACKS}/py_ast_423k_drank'
pos_mode = 'rope_both'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 2, 2   # 16 seq + 2 numbers x 2 planes + 44 idle = 64
rope_token_planes = 44                   # IDLE under rope_both; fills the allocation
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 256                   # ~27,000 target tokens/step; one pass = 1,394 steps
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the config record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                  # LR smoke, 560 steps: lowest final val nats/tok
min_lr = 1.2e-04
max_iters = 11200                        # 8 passes over 37.7M train tokens
lr_decay_iters = 11200
warmup_iters = 224                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 560                      # 5% -> 20 milestones, all kept
eval_records = 1024
log_interval = 20
compile = True
