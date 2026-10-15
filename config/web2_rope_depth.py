from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS web cell 7 -- arm rope_depth: sequence position + the token's DOM DEPTH, two planes.
#
# WHY THIS RUN EXISTS. The cheap confirmation cell (webc, 2,048 context, small model, 10 seeds)
# measured depth as the BEST arm on prose -- better than the full 44-plane path on every seed:
#
#     depth vs seq          -0.0198   7/7 seeds   p=0.0012
#     both  vs seq          -0.0125   8/8 seeds   p=0.0008
#     depth vs both         -0.0085   7/7 seeds   p=0.036
#
# That contradicts what section 6.6 predicted. Web is the one corpus where depth CANNOT
# reconstruct the tree -- its stream is text nodes only, every one a leaf, so 0% of parents are
# recoverable against 100% on Python and spreadsheets -- and the paper argued depth there is
# strictly LESS information than the path. It is, and it still wins. Reconstruction was never the
# reason depth works: depth is a useful fact about where a passage sits in a page even when the
# ancestors are absent.
#
# WHAT THIS RUN ADDS. The confirmation cell cannot be quoted beside the published cell-7 numbers
# -- different pack, context and model. This arm puts depth on the SAME footing as rope_seq,
# rope_both, rope_leaf and rope_layer_split: same pack, same geometry, same 537,000-step schedule,
# same learning rate. It is the only way the claim "two planes beat forty-four" can be stated
# against the numbers the paper already reports.
#
# ALLOCATION. 16 sequence + 2 depth = 18 of 64 planes rotate, against rope_both's 60. The module
# REFUSES rope_token_planes > 0 under this mode, and shares step_rel/_rel_angles with rope_rel
# rather than growing a second depth path, so no new allocation key is introduced.
#
# Override per run: --seed=N --out_dir=...   (cell 7 passes the pack and schedule explicitly)
out_dir = f'{RUNS}/web7/rope_depth_p1'
pack = f'{PACKS}/web_text_full_v3'
pos_mode = 'rope_depth'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 10, 4, 512   # head dim 128; 31.5M non-embedding -- identical to the other arms
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # levels parsed, NOT rotated
rope_token_planes = 0                   # refused under rope_depth
rope_rel_planes = 2                     # the DEPTH group -- 16 + 2 = 18 of 64 rotated
rope_decay = False
bias = False
dropout = 0.0
batch_size = 2
gradient_accumulation_steps = 16
learning_rate = 6e-4                    # inherited UNCHANGED from the other cell-7 arms, which is
                                        # the point: re-smoking it would make this arm's schedule
                                        # differ from the baselines it must be compared against.
min_lr = 6.0e-05
max_iters = 537000
lr_decay_iters = 537000
warmup_iters = 10740                    # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 26850                   # 5% -> 20 milestones, matching the other arms exactly
eval_records = 1024
log_interval = 20
bucket_batching = True
records_per_step = 32
token_budget = 8192
compile = True
