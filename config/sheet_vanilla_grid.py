from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- CONTROL: spreadsheets with every cell ADDRESS written out in the text, sequence + GRID (sheet, row, column; 6 planes).
#
# WHY. A reviewer's objection: on the values-only render the baseline never sees where a cell
# is, so the tree arms may win just by being given the grid. In this render every cell's
# address is in the token stream, so sequence only CAN read the grid from the text. If
# the grid still helps here, the gain is not merely supplying missing information. Paired
# against vanilla_seq_v2 and both_vanilla_v2 (existing, 10 seeds) on the same token stream.
# Checked 2026-10-03: this pack's node_path levels 0/1/2 are also sheet (0-11) / row (to
# 26,671) / column (to 16,383), as on the values-only pack.
#
# Everything else is identical to config/sheet_grid.py.
out_dir = f'{RUNS}/sheets/grid_vanilla_v2_s0'
pack = f'{PACKS}/sheet_vanilla_v2_lvl3'
                                                # array is a symlink to sheet_values_v2, so the
                                                # token stream is byte-identical. The rotary
                                                # module REFUSES a path with more columns than
                                                # it has levels -- correctly -- so a 1-level arm
                                                # needs 1-level data. See slice_levels.py.
pos_mode = 'rope_both'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 3, 2   # 16 + 3 levels (sheet,row,column) x 2 + 42 idle = 64
rope_token_planes = 42                   # IDLE under rope_both; fills the allocation
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
