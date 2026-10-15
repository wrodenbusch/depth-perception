from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- SPREADSHEETS: sequence + GRID (sheet, row, column), 6 planes.
#
# WHY. The spreadsheet tree's first three path levels are the cell's real coordinates -- sheet
# index, row number, column number -- and levels 4+ are the formula's own nesting. The full
# sequence + tree arm spends 44 planes on all 22 levels. This arm keeps only the grid, the
# cheap analogue of depth for grid-shaped data: if it matches sequence + tree, the practical
# recommendation is "depth for nested documents, row + column for grids".
#
# Data: sheet_values_v2_lvl3, built by slice_levels.py --levels 3: node_path truncated to the
# first three levels, every other array a symlink, so the token stream is byte-identical to
# seq_v2 / both_v2 / depth_v2 and the comparison is the rotated coordinate and nothing else.
out_dir = f'{RUNS}/sheets/grid_v2_s0'
pack = f'{PACKS}/sheet_values_v2_lvl3'   # node_path sliced to 3 levels (sheet, row, column); every other
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
