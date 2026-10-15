from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- FULL-WIDTH RoPE: all 64 planes by the sequence index, as in standard RoPE.
#
# WHY. Every baseline in this project is sequence only on 16 of 64 planes (partial RoPE, the
# GPT-NeoX / Pythia rotary_pct=0.25 allocation), while the structural configurations rotate up to
# 62. A reviewer will ask whether the structured arms win only because they rotate more planes.
# This arm answers it: the same model with every plane rotated by sequence position, at the
# standard spectrum base^(-i/64). Implemented as a per-layer plan that is 's' in every layer
# (sequence at full width); everything else is identical to the setting's sequence-only config.
out_dir = f'{RUNS}/py/full_rope_s0'
pack = f'{PACKS}/py_ast_423k'
pos_mode = 'rope_layer_plan'
rope_layer_plan = 'ssssss'           # sequence at full width in all 6 layers
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 64, 22, 2   # FULL width: every plane by sequence; levels unused (no 't' layer)
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
