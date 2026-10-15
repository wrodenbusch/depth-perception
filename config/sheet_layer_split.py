from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- SPREADSHEETS, rope_layer_split: three layers of sequence, three of tree.
#
# WHY THIS EXISTS. Section 6.3 pre-registered a prediction with two halves: that spreadsheets, at
# 0.56 nodes per token against code's 0.86 and prose's 0.12, would land BETWEEN those corpora on
# (a) the joint margin and (b) the fraction of that margin surviving a layer split. Half (a) came
# out as predicted. Half (b) was never measured, because no layer-split arm was ever run on this
# corpus -- an omission in execution, not a result. This closes it.
#
# THE QUANTITY. Compare this arm's gap to seq_v2 against both_v2's gap to seq_v2; the ratio is the
# fraction surviving the split. Measured elsewhere: 77% on code (where the two signals are nearly
# redundant, so separating them costs little) and 16% on prose (where they are complementary). The
# prediction is that spreadsheets fall between.
#
# SIX LAYERS, so 3 sequence / 3 tree -- the even split the mode requires. The web cell has 10 layers
# and splits 5/5; the Python cell has 6 and splits 3/3, so this matches Python's shape.
#
# A SEQUENCE LAYER SPANS ALL 64 PLANES, not 16. The module REFUSES anything else under a split mode,
# and correctly: a split arm gives each factor a whole layer, so the sequence spectrum covers every
# plane rather than a band. This is why the split arms get MORE rotary resolution than the
# interleaved arm, not less -- which is what makes their losing the interesting result.
out_dir = f'{RUNS}/sheets/layer_split_v2_s0'
pack = f'{PACKS}/sheet_values_v2'
pos_mode = 'rope_layer_split'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; 6 layers -> 3 sequence / 3 tree
rope_seq_planes, rope_levels, rope_planes_per_level = 64, 22, 2   # seq spans ALL 64; tree 22*2=44
rope_token_planes = 4                    # + 4 token = 48 of 64 rotated on a tree layer
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
log_interval = 20
compile = True
