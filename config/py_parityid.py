from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- PYTHON: sequence + PARITY-SPLIT NODE IDs (author's idea, 2026-10-06): a child coincides
# with its parent.
#
# Two ID groups alternate by depth parity: group A carries the nearest even-depth node among
# {self, parent}, group B the nearest odd-depth one. A child therefore shares its parent's ID in
# the parent's group, and every node shares its own ID with its children in the other. Parent <->
# child alignment is exact at any distance -- the long-range link (a statement deep in a long
# function body back to the `def` that owns it) that the sibling-group parent tag could not give,
# since siblings are adjacent in a prefix serialisation and sequence position already relates them.
#
# Data: py_ast_423k_parityid (parity_id_pack.py): node_path = [A, B, depth], everything else a
# symlink, so the token stream is byte-identical. Node IDs per file: p50 79, p99 318, max 602 --
# at the 2-plane wrap (~318), so 3 planes per level (wrap ~5,000). That also gives depth 3 planes
# (it needs 2); total 9 structural planes.
out_dir = f'{RUNS}/py/parityid_s0'
pack = f'{PACKS}/py_ast_423k_parityid'
pos_mode = 'rope_both'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 3, 3   # 16 seq + [A, B, depth] x 3 + 39 idle
rope_token_planes = 39                   # IDLE under rope_both; fills the allocation
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
