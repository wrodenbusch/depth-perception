from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- THE PLANE-MATCHED CONTROL for rope_depth. A PATH of the same 2 planes.
#
# rope_depth rotates 16 sequence + 2 depth = 18 planes and gains whatever it gains against
# rope_both's 16 + 44 = 60. That comparison is ambiguous between two readings:
#   (a) depth is a sufficient encoding of the tree, or
#   (b) most of those 44 path planes were doing nothing, which would be a finding about
#       rope_both rather than about depth.
#
# This arm separates them. It spends EXACTLY the same 2 structural planes as rope_depth, on a
# path instead -- the top-level sibling rank, one level only. 16 + 2 + 46 idle token planes =
# 64, and the token group does not rotate under rope_both, so 18 planes turn, matching
# rope_depth plane for plane. Verified: it distinguishes top-level siblings, identical
# siblings collide, and the 46 filler planes are provably unrotated.
#
# So: rope_depth vs THIS asks whether DEPTH is the right thing to spend 2 planes on. Both vs
# rope_both asks what the other 42 planes buy. Neither question is answerable from the other
# comparison alone.
#
# Expect this to be weak: one level of sibling rank identifies almost nothing -- it says which
# top-level child of the root you descend from and no more. That is the point. If depth beats
# it clearly, the gain is about depth; if they tie, the gain was about having 2 more planes.
out_dir = f'{RUNS}/py/toplevel_s0'
pack = f'{PACKS}/py_ast_423k_lvl1'
pos_mode = 'rope_both'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 1, 2   # 16 seq + 1 level x 2 + 46 idle = 64
rope_token_planes = 46                   # IDLE under rope_both; here only to fill the
                                         # allocation, which is refused if it does not sum to 64
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
