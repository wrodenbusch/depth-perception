from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- WEB-S (whole pages, 2,048 context): sequence + tree path + DEPTH (`rope_rel`), the best configuration on Python.
#
# WHY. On Python this arm beat sequence + tree on 5 of 5 seeds (-0.0018, p=0.095) and was the best
# configuration measured. It was never run on any other corpus, so it cannot be the paper's headline
# method until it is. This arm is the paired test: identical to the corpus's sequence + tree arm
# except that two extra planes rotate by the token's tree depth (the 4 within-node planes are idle
# under rope_both, so 60 -> 62 rotated planes). Baselines: sequence + tree and sequence + depth,
# both already at n=10 on this pack.
out_dir = f'{RUNS}/webc/rel_s0'
pack = f'{PACKS}/web_whole_b2048'
pos_mode = 'rope_rel'
seed = 0
block_size = 2048
vocab_size = 50304
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128, so 64 rotary planes; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 seq + 22*2 tree + 2 depth = 62 of 64
rope_token_planes = 0                    # refused under rope_rel
rope_rel_planes = 2                      # the DEPTH group, on top of the per-level path
rope_decay = False
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 32                    # ~32k tokens/step at 996 tokens/record
token_budget = 8192                      # measured best; see the header
batch_size = 2                           # unused under bucket_batching, kept for the record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                   # LR smoke, 600 steps, 3 rates on webc_both:
                                         #   1.2e-3 -> 5.5908  (chosen)
                                         #   6e-4   -> 5.7582
                                         #   2.4e-3 -> 5.8932
min_lr = 1.2e-4
max_iters = 9375                         # ~300M tokens at 32k/step
lr_decay_iters = 9375
warmup_iters = 188                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 469                      # 5% -> 20 milestones
eval_records = 1024
log_interval = 20
compile = False                          # OFF DELIBERATELY. Bucket batching can produce up to
                                         # 128 distinct tensor shapes (8 batch sizes x 16 padded
                                         # lengths) and torch.compile recompiles for each: three
                                         # concurrent runs spent 13 min without finishing step 0.
                                         # Uncompiled is ~0.7 s/step, so a run is ~1.8 h and the
                                         # compile would not amortise. Every arm is treated the
                                         # same, which is what the comparison needs.
