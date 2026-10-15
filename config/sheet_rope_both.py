"""Sequence + tree. The supplement claim at 1.02 nodes/token.

Sheets cell, arm sheet_rope_both. Sized by analogy to the web cell (10x4x512, head_dim 128), NOT by
throughput.py -- no GPU was free during the build.

RENDER AND COMPARABILITY -- READ BEFORE COMPARING ARMS
------------------------------------------------------
`sheet_rope_seq` trains on the **vanilla** render (cell address + value), which is
SpreadsheetLLM's own baseline serialisation (arXiv:2407.09025). The tree arms train on the
**values** render (no addresses), so the structure reaches them only through rotation.

Measured on 300 dev records: vanilla costs 674 tokens/record against values' 494, i.e. 36.6%
of the sequence arm's stream is address text. Address tokens are highly predictable, so they
DEFLATE nats-per-token for the vanilla arm. **Compare these arms on nats per RECORD, not
nats per token.** nats/token is not comparable across renders.

block_size is 4096, not the Python cell's 1024: full-corpus records average ~3,039 GPT-2
tokens (12,156 text chars), and the small subsets are NOT representative -- they average
2,753 chars. At 1024 most workbooks would be truncated.

learning_rate is UNSMOKED -- no GPU was free to run the 560-step grid.
"""
from topos_paths import PACKS, RUNS  # noqa: E402

out_dir = f'{RUNS}/sheets/sheet_rope_both_s0'
pack = f'{PACKS}/sheet_values'
pos_mode = 'rope_both'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 10, 4, 512   # head dim 128; 4 heads leaves room for a head-split arm
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 44 + 4 = 64 = head_dim/2
rope_token_planes = 4
rope_decay = False                       # rotation tables out of the weight-decay group
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 32
token_budget = 8192
batch_size = 2
gradient_accumulation_steps = 16
learning_rate = 6e-4                  # UNSMOKED: no GPU free. 6e-4 is the rate every
min_lr = 6.0e-05                      # non-'none' web arm took from its own grid.
max_iters = 15200                        # ~8 passes: 248M tokens / 131k tokens per step = 1,893/pass
lr_decay_iters = 15200
warmup_iters = 304                       # 2%
eval_interval = 760                      # 5% -> 20 milestones, all kept
eval_records = 1024
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
log_interval = 20
compile = True
