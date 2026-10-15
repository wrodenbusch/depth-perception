from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- CONTROL: depth as a learned INPUT EMBEDDING, against the rotary depth marker.
#
# WHY. A reviewer's main objection: our structured arms may win because they receive structure
# the sequence-only baseline never sees, not because it arrives THROUGH THE ROTATION. This arm
# gives the model exactly the same depth (model.py computes it as rope_depth does: the count of
# non-ABSENT path levels) but adds it as a learned embedding at the input, on top of plain
# sequence RoPE on the same 16 planes. If it matches sequence + depth, the rotary form adds
# nothing beyond supplying the number; if the rotary depth wins, the mechanism matters.
# Zero-initialised and excluded from weight decay (rope_decay=False), like the rotary step.
#
# Everything else is identical to config/sheet_depth.py.
out_dir = f'{RUNS}/sheets/depthemb_v2_s0'
pack = f'{PACKS}/sheet_values_v2'
pos_mode = 'rope_seq'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # levels parsed for the depth; none rotated
rope_token_planes = 4                    # idle; fills the 64-plane allocation
rope_rel_planes = 0
depth_emb = True                         # THE CONTROL: depth added at the input
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
