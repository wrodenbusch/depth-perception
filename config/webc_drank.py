from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- the "tree grid": sequence + TWO tree numbers, the node's DEPTH and its OWN SIBLING RANK.
#
# WHY. On spreadsheets the structural gain came entirely from a grid (row, column) on 6 planes,
# matching the full 44-plane path. The author's analogue for any tree: depth (vertical) and which
# child of its parent the node is (horizontal). Two numbers, 4 planes. Paired against
# sequence only, sequence + depth and the full path at the same seeds.
#
# Data: <pack>_drank from depth_rank_pack.py -- node_path replaced by [depth, own_rank], every
# other array a symlink, so the token stream is byte-identical to the source pack. rope_both
# with rope_levels = 2 rotates level 0 by depth and level 1 by own rank with the same ladder and
# presence handling the full path uses; the other 44 planes are idle.
out_dir = f'{RUNS}/webc/drank_s0'
pack = f'{PACKS}/web_whole_b2048_drank'
pos_mode = 'rope_both'
seed = 0
block_size = 2048
vocab_size = 50304
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128, so 64 rotary planes; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 2, 2   # 16 seq + 2 numbers x 2 planes + 44 idle = 64
rope_token_planes = 44                   # IDLE under rope_both; fills the allocation
rope_rel_planes = 0
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
