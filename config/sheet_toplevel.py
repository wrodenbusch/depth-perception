from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- SPREADSHEETS, the depth question on a SECOND corpus.
#
# THIS ARM: THE PLANE-MATCHED CONTROL for sheet_depth, mirroring py_toplevel.
#
# Spends EXACTLY the same two structural planes, on a one-level path (which top-level child of the
# root this cell descends from) instead of on depth. Same 18 rotating planes, same two learned
# scalars. Without it, depth beating the full path is ambiguous between 'depth is the right signal'
# and 'two more planes of anything help'. On Python this control bought -0.0008 (t = -1.24, 3/5
# seeds, not separable) while depth bought -0.0076, which is what made the Python claim about
# depth rather than about scarcity. This asks the same question on a second corpus.
#
# Everything else is byte-identical to config/sheet_v2_both.py, so the comparison against the
# existing both_v2 (n=10) and seq_v2 (n=10) arms is pos_mode and nothing else.
out_dir = f'{RUNS}/sheets/toplevel_v2_s0'
pack = f'{PACKS}/sheet_values_v2_lvl1'   # node_path sliced to 1 level; every other
                                                # array is a symlink to sheet_values_v2, so the
                                                # token stream is byte-identical. The rotary
                                                # module REFUSES a path with more columns than
                                                # it has levels -- correctly -- so a 1-level arm
                                                # needs 1-level data. See slice_levels.py.
pos_mode = 'rope_both'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 1, 2   # 16 + 1 level x 2 + 46 idle = 64
rope_token_planes = 46                   # IDLE under rope_both; fills the allocation
rope_rel_planes = 0
rope_decay = False
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 32
token_budget = 8192
batch_size = 2
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                   # inherited from the v2 pair, unchanged so the arms match
min_lr = 1.2e-04
max_iters = 8240
lr_decay_iters = 8240
warmup_iters = 165
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 412
eval_records = 1024
compile = True
