from topos_paths import PACKS, RUNS  # noqa: E402
# TREE FIRST (tttsss): a per-layer plan over the 6 layers. Each dedicated layer gives its
# factor the FULL 64 planes instead of sharing them, so sequence spans the whole head and
# the tree gets 22 levels x 2 + 4 token = 48 with 16 planes left unrotated.
# Blocked, tree first: layers 0-2 tree, layers 3-5 sequence. The order-reversed control for
# py_layer_seq_first -- same budget, same factors, opposite placement.
#
# TOPOS: the layer-mix ablation on the PYTHON AST pack (cell 6 geometry), screened here
# because a Python run is ~1 h against ~5 days for a web run, and every other Python arm
# already has 5 seeds to compare against.
out_dir = f'{RUNS}/py/layer_tree_first_s0'
pack = f'{PACKS}/py_ast_423k'
pos_mode = 'rope_layer_plan'
rope_layer_plan = 'tttsss'          # one character per layer: s=sequence, t=tree, n=none
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 64, 22, 2   # FULL width per factor: 's' layers
                                                                 # span all 64 planes; 't' layers use 48
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
