from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- PYTHON STATEMENT GRID (author, 2026-10-06): sequence + statement ordinal (level 0), 2 planes per level,
# plus the token's position within its statement on the 4 token-leaf planes (rope_all).
#
# WHY. The grid family carries the whole gain on spreadsheets (row, column) and chat (message,
# position in message): a GLOBAL unit number shared by every token of the unit, plus position
# inside it. Python had no comparable coordinate, and is the one setting where the full tree still
# beats every cheap option. Its natural row is the statement. See py_stmtgrid_pack.py for how
# statements are identified (ASDL field + container parent) and why the coordinates are causal.
#
# Data: py_ast_423k_stmtgrid: every array but the sidecar's tok_node / node_path / node_off / tok_within is a
# symlink to py_ast_423k, so the token stream is byte-identical to rope_seq / depth / rope_both.
# Levels beyond those named are ABSENT (zero rotation). Everything else is py_depth.py unchanged.
out_dir = f'{RUNS}/py/stmtgrid_s0'
pack = f'{PACKS}/py_ast_423k_stmtgrid'
pos_mode = 'rope_all'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2
rope_token_planes = 4                    # position within the statement (rope_all rotates the token-leaf planes)
rope_rel_planes = 0
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 256                   # ~27,000 target tokens/step; one pass = 1,394 steps
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                   # the baseline's smoked rate; same pack, same batch
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
