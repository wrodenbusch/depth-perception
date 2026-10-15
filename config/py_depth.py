from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- `rope_depth`: sequence position + the token's tree DEPTH, and NOTHING else.
#
# THE IDEA (author, 2026-09-29). A tree serialised in PREFIX (DFS pre-order) order is fully
# determined by (emission order, depth): the parent of any node is the nearest PRECEDING node one
# level shallower. So depth is a LOSSLESS encoding of the tree even though the tree is never made
# explicit. Verified on this pack's own sidecar -- reconstructing every node's parent from order
# and depth alone recovered 558,271 parents with ZERO errors over 4,000 records, and
# test_depth_and_order_reconstruct_the_tree pins it in the suite rather than in a shell history.
# The Python render is a "compact prefix serialisation", which is why the property holds here.
#
# THE QUESTION. rope_both spends 16 sequence + 22 levels x 2 = 60 planes. This arm spends
# 16 + 2 = 18. If they measure the same, that is a 22x reduction in the structural budget and
# the most practically useful result in the project.
#
# WHY IT MIGHT NOT WORK, which is why it is worth running rather than assuming. Lossless in
# principle is not the same as usable by attention. The full path gives DIRECT coincidence: two
# tokens in one subtree share a path prefix and are aligned on those planes in one step. With
# depth, the relative rotation between two tokens is the depth DIFFERENCE, and "the nearest
# preceding node at depth-1" is not a fixed offset -- attention has to combine a sequence
# rotation with a depth rotation and do something argmax-shaped. It may not recover what the
# path hands it directly.
#
# ==================  WHICH COMPARISON ANSWERS WHICH QUESTION  ==================
# Quote both. A later reader will otherwise take whichever suits them.
#
#   vs `rope_seq` (16 planes, n=20 on this pack) -- THE CLEAN ONE.
#       Plane-adjacent: 18 against 16. Asks "does depth add anything at all?"
#
#   vs `rope_both` (60 planes, n=5 on this pack) -- THE EFFICIENCY ONE, WITH A CONFOUND.
#       18 against 60 is a far larger plane-count gap than anything this project has disclosed
#       before (rope_all vs rope_both was 64 vs 60, measured at +0.0001, t=+0.12, n=5). If
#       rope_depth MATCHES rope_both, the result is AMBIGUOUS between "depth is sufficient" and
#       "most of those 44 planes were doing nothing" -- those are different claims and this arm
#       alone cannot separate them.
# ===============================================================================
#
# Shares `step_rel` and `_rel_angles` with rope_rel rather than growing a second depth path; the
# only difference is that rope_rel keeps the per-level planes ON TOP of depth and this arm drops
# them. It therefore reuses `rope_rel_planes` and adds NO new allocation key -- no fifth chance
# at the plumbing trap that killed two launches (3ca636a) and burned five seeds (d132e3c). A
# missing value cannot fail silently either: GraphRoPE refuses rel_planes <= 0 for this mode.
#
# No presence term and no absent class, exactly as rope_rel argues: every token has a depth, and
# a token outside the tree has depth 0 -- a real value distinct from a depth-1 child of the root,
# not a sentinel.
#
# BOTH BASELINES ALREADY EXIST on this pack, so only this arm needs running.
out_dir = f'{RUNS}/py/depth_s0'
pack = f'{PACKS}/py_ast_423k'
pos_mode = 'rope_depth'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # levels parsed, NOT rotated
rope_token_planes = 0                    # refused under rope_depth: rope_seq is the plane-adjacent
                                         # baseline, so the arm takes no token-leaf planes
rope_rel_planes = 2                      # the DEPTH group -- 16 + 2 = 18 of 64 rotated,
                                         # the remaining 46 left unrotated
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 256                   # ~27,000 target tokens/step; one pass = 1,394 steps
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                   # the baseline's smoked rate; same pack, same batch
                                         # shape, nothing about the token stream changed
min_lr = 1.2e-04
max_iters = 11200                        # 8 passes, identical to rope_seq and rope_both
lr_decay_iters = 11200
warmup_iters = 224                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 560                      # 5% -> 20 milestones, all kept
eval_records = 1024
log_interval = 20
compile = True
