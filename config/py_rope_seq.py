from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS cell 6 -- the PYTHON AST pack. The project's original question, on the corpus the
# banked result came from, and the first cell where the encoding's capacity is not the binding
# constraint: AST depth p50 5 / p99 10, sibling rank p99 7 / max 151 against a collision point
# of 318. 11 programs of 423,111 exceed 22 levels; nothing aliases.
#
# Model is the BANKED PYTHON GEOMETRY, not the web one: 37.7M train tokens against the web cells'
# 31.5M-parameter model would be memorisation. 6 x 256 x 2 heads keeps head_dim at 128, so the
# 64-plane allocation -- and therefore the mechanism under test -- is identical to every other cell.
out_dir = f'{RUNS}/py/rope_seq_s0'
pack = f'{PACKS}/py_ast_423k'
pos_mode = 'rope_seq'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 44 + 4 = 64 = head_dim/2
rope_token_planes = 4
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
