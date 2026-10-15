from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS cell 6 -- the PYTHON AST pack. The project's original question, on the corpus the
# banked result came from, and the first cell where the encoding's capacity is not the binding
# constraint: AST depth p50 5 / p99 10, sibling rank p99 7 / max 151 against a collision point
# of 318. 11 programs of 423,111 exceed 22 levels; nothing aliases.
#
# Model is the BANKED PYTHON GEOMETRY, not the web one: 37.7M train tokens against the web cells'
# 31.5M-parameter model would be memorisation. 6 x 256 x 2 heads keeps head_dim at 128, so the
# 64-plane allocation -- and therefore the mechanism under test -- is identical to every other cell.
out_dir = f'{RUNS}/py/ref_slot_s0'
pack = f'{PACKS}/py_ast_423k'
pos_mode = 'rope_ref_slot'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 seq + 22*2 tree + 2 slot = 62 of 64
rope_token_planes = 0                    # the 4 token-leaf planes are worth nothing when sequence rotates (rope_all vs rope_both: +0.0001, t=+0.12, n=5), so they fund the slot group
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 256                   # ~27,000 target tokens/step; one pass = 1,394 steps
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the config record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                  # UNSMOKED: no GPU was free to run the 560-step grid.
                                        # Every other Python arm took 1.2e-3 from that smoke and this
                                        # arm is structurally similar, so it is the defensible default --
                                        # but it is an assumption, not a measurement. If it destabilises,
                                        # the rule is to reject the rate and take the next lower one.
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

# TOPOS: the SYMBOL SLOT group. 16 seq + 22*2 tree + 2 slot = 62 of 64; the remaining 2
# planes stay unrotated (plain content channels), which is what partial-RoPE models do.
# 2 planes is ample: measured slots per record are median 8 / p99 35 / max 75, against a
# slow plane that counts to ~318 before it wraps. 0.00% of records exceed 318.
rope_slot_planes = 2
