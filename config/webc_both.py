from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- WEB CONFIRMATION CELL. The web result at MANY SEEDS, on a cheap geometry.
#
# WHY THIS CELL EXISTS. Cell 7 measures the web arms at ONE seed each over a 536,000-step pass
# (~180 h per arm), and the answer it is giving -- that plain sequence position BEATS tree
# position on prose -- is the opposite of the spreadsheet and Python cells. A single seed cannot
# distinguish that from seed noise, and seed spread on these corpora (0.004-0.012 nats/token) is
# the same size as the effects. So the whole web story rests on an unreplicated number.
#
# THIS IS NOT COMPARABLE TO CELL 7 AND IS NOT MEANT TO BE. Different pack, different context,
# different model: nats/token must never be quoted across the two. What transfers is the SIGN and
# whether it is stable across seeds -- which is the only thing in doubt.
#
# THE PACK IS WHOLE PAGES ONLY (`web_whole_b2048`, --drop-long). Nothing in it is truncated: a
# page too long for the 2,048 context was left out rather than cut off, at a cost of 26% of
# eligible pages and a bias toward shorter ones (manifest records it). Cell 7s pack crops, and
# discards 11.4% of all text in doing so.
#
# COST, MEASURED not assumed (budget_sweep.py, 2026-09-30): 22,500 real tokens/s on one card at
# token_budget 8192, so ~3.7 h for 300M tokens -- about 48x cheaper per run than a cell-7 arm.
# The saving is from the smaller model and shorter context ONLY. Raising token_budget to 16,384
# measured SLOWER (14,900 tok/s) and 32,768 OOMed, so the "small model + huge batch" plan this
# cell was first sized against is wrong and the inherited 8,192 is already the best setting.
#
# 300M tokens on 4.7M non-embedding parameters is 64 tokens/parameter, well past Chinchilla, and
# matches what the Python cell converged on. The pack holds 896M, so no record repeats.
#
# THIS ARM: sequence position + the full DOM rank path. The treatment under test.
out_dir = f'{RUNS}/webc/both_s0'
pack = f'{PACKS}/web_whole_b2048'
pos_mode = 'rope_both'
seed = 0
block_size = 2048
vocab_size = 50304
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128, so 64 rotary planes; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 44
rope_token_planes = 4                    # idle under rope_both; fills the 64-plane allocation
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
