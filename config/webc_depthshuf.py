from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- CONTROL: SHUFFLED depth on the same two rotary planes as sequence + depth.
#
# WHY. If the rotary depth marker helps only because two more planes rotate (capacity), it should
# help as much when the depth it carries is wrong. The pack comes from shuffle_depth_pack.py:
# within each record the in-tree nodes' depths are randomly permuted (seed 0), keeping each
# record's multiset of depths and every depth-0 node, and losing the depth-to-node information.
# The token stream is a symlink to the source, so this is paired seed-for-seed with
# sequence only and sequence + depth on the same tokens.
#
# Everything else is identical to config/webc_depth.py.
out_dir = f'{RUNS}/webc/depthshuf_s0'
pack = f'{PACKS}/web_whole_b2048_depthshuf'
pos_mode = 'rope_depth'
seed = 0
block_size = 2048
vocab_size = 50304
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128, so 64 rotary planes; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # levels parsed, NOT rotated
rope_token_planes = 0                    # refused under rope_depth
rope_rel_planes = 2                      # the DEPTH group -- 16 + 2 = 18 of 64 rotated
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
