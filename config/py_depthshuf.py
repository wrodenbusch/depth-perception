from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- CONTROL: SHUFFLED depth on the same two rotary planes as sequence + depth.
#
# WHY. If the rotary depth marker helps only because two more planes rotate (capacity), it should
# help as much when the depth it carries is wrong. The pack comes from shuffle_depth_pack.py:
# within each record the in-tree nodes' depths are randomly permuted (seed 0), keeping each
# record's multiset of depths and every depth-0 node, and losing the depth-to-node information.
# The token stream is a symlink to the source, so this is paired seed-for-seed with
# sequence only and sequence + depth on the same tokens.
#
# Everything else is identical to config/py_depth.py.
out_dir = f'{RUNS}/py/depthshuf_s0'
pack = f'{PACKS}/py_ast_423k_depthshuf'
pos_mode = 'rope_depth'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # levels parsed, NOT rotated
rope_token_planes = 0                    # refused under rope_depth: rope_seq is the plane-adjacent
                                         # baseline, so the arm takes no token-leaf planes
rope_rel_planes = 2                      # the DEPTH group -- 16 + 2 = 18 of 64 rotated,
                                         # the remaining 46 left unrotated
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 256                   # ~27,000 target tokens/step; one pass = 1,394 steps
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                   # the baseline's smoked rate; same pack, same batch
                                         # shape, nothing about the token stream changed
min_lr = 1.2e-04
max_iters = 11200                        # 8 passes, identical to rope_seq and rope_both
lr_decay_iters = 11200
warmup_iters = 224                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 560                      # 5% -> 20 milestones, all kept
eval_records = 1024
log_interval = 20
compile = True
