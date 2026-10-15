from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- SPREADSHEETS: sequence + tree path + DEPTH (`rope_rel`), the best configuration on Python.
#
# WHY. On Python this arm beat sequence + tree on 5 of 5 seeds (-0.0018, p=0.095) and was the best
# configuration measured. It was never run on any other corpus, so it cannot be the paper's headline
# method until it is. This arm is the paired test: identical to the corpus's sequence + tree arm
# except that two extra planes rotate by the token's tree depth (the 4 within-node planes are idle
# under rope_both, so 60 -> 62 rotated planes). Baselines: sequence + tree and sequence + depth,
# both already at n=10 on this pack.
out_dir = f'{RUNS}/sheets/rel_v2_s0'
pack = f'{PACKS}/sheet_values_v2'
pos_mode = 'rope_rel'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 seq + 22*2 tree + 2 depth = 62 of 64
rope_token_planes = 0                    # refused under rope_rel
rope_rel_planes = 2                      # the DEPTH group, on top of the per-level path
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
