from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- SPREADSHEETS, the depth question on a SECOND corpus.
#
# THIS ARM: sequence position + the cell's tree DEPTH, two planes and nothing else.
#
# WHY IT MATTERS HERE. Depth is the project's cheapest-to-adopt claim -- two planes buying 80% of
# what forty-four planes of full path buy -- and it has been measured on ONE corpus, Python. A
# claim about structural position in general cannot rest on one serialisation.
#
# SPREADSHEETS SATISFY THE PRECONDITION, verified rather than assumed. Depth is lossless only on a
# prefix serialisation whose ancestors are all PRESENT in the stream. Measured on this pack:
# stored in depth-first order for 100% of records, and 'parent = nearest preceding node one level
# shallower' recovers the true parent for 100% of nodes (56.3% of nodes have a descendant in the
# stream, so there is a real tree here to recover). Web fails that test at 0% -- its stream is
# text nodes only -- which is why depth is expected to behave differently there.
#
# Everything else is byte-identical to config/sheet_v2_both.py, so the comparison against the
# existing both_v2 (n=10) and seq_v2 (n=10) arms is pos_mode and nothing else.
out_dir = f'{RUNS}/sheets/depth_v2_s0'
pack = f'{PACKS}/sheet_values_v2'
pos_mode = 'rope_depth'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # levels parsed, NOT rotated
rope_token_planes = 0                    # refused under rope_depth
rope_rel_planes = 2                      # the DEPTH group -- 16 + 2 = 18 of 64 rotated
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
