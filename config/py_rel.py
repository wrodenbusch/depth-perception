from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS R11a -- `rope_rel`, DIRECTED RELATIONS BY CONSTANT OFFSET, on the PYTHON AST pack.
#
# The arm rotates a dedicated plane group by the token's tree DEPTH, so the angle difference
# across a parent/child hop is EXACTLY one step -- signed, so child-to-parent is minus one
# step, and a grandparent is two. Direction, relation type and composition all fall out of
# the group structure; coincidence (rope_ref_slot) can express none of them, because sharing
# an angle is symmetric by construction.
#
# WHY IT IS ALSO A FIX, NOT ONLY AN ADDITION. Measured 2026-09-28 on rope_ref_slot: an absent
# slot contributes exactly zero, so all non-symbol tokens are as coincident with each other as
# two mentions of one symbol are, and at init slot 0 is indistinguishable from absent. 57% of
# tokens carry no slot and the slot group has no shallow level to fall back on, so the majority
# of the corpus sits in one meaningless class on that axis. Every token has a DEPTH, so this
# axis has no absent class at all.
#
# PAIRED BASELINE IS rope_both. This arm deliberately carries no slot group and no token-leaf
# group, so seq + tree is the only difference and the comparison isolates the relational axis.
# Note the asymmetry a reviewer will find: rope_both rotates 60 planes and leaves its 4 token
# planes idle, while this arm rotates 62. Two extra rotating planes are a confound, though a
# small one -- four extra were worth nothing (rope_all vs rope_both: +0.0001, t=+0.12, n=5).
#
# PRIOR ART TO CITE: RotatE (Sun, Deng, Nie, Tang, ICLR 2019, arXiv 1902.10197) defines each
# relation as a rotation from source to target and proves symmetry/antisymmetry, inversion and
# composition -- as a knowledge-graph link-prediction embedding, with no use as a positional
# encoding inside a transformer. The claim here is the placement and the partitioning.
#
# Model is the BANKED PYTHON GEOMETRY, unchanged from every other arm on this pack, so head_dim
# stays 128 and the 64-plane allocation is comparable cell to cell.
out_dir = f'{RUNS}/py/rel_s0'
pack = f'{PACKS}/py_ast_423k'
pos_mode = 'rope_rel'
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 seq + 22*2 tree + 2 rel = 62 of 64
rope_token_planes = 0                    # refused under rope_rel: the arm isolates the relational axis
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 256                   # ~27,000 target tokens/step; one pass = 1,394 steps
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the config record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                  # UNSMOKED: no GPU was free to run the 560-step grid.
                                        # Every other Python arm took 1.2e-3 from that smoke and this
                                        # arm is structurally similar, so it is the defensible default --
                                        # but it is an assumption, not a measurement. If it destabilises,
                                        # the rule is to reject the rate and take the next lower one.
min_lr = 1.2e-04
max_iters = 11200                        # 8 passes over 37.7M train tokens
lr_decay_iters = 11200
warmup_iters = 224                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 560                      # 5% -> 20 milestones, all kept
eval_records = 1024
log_interval = 20
compile = True

# TOPOS: the RELATIONAL (depth) group. 16 seq + 22*2 tree + 2 rel = 62 of 64; the remaining 2
# planes stay unrotated (plain content channels), which is what partial-RoPE models do.
# 2 planes is ample: the fast plane resolves adjacent depths -- which is the point, a hop is a
# step of one -- and the slow plane counts to ~318 depths, far past the deepest tree observed
# (Python max 57, sheets max 60). No presence term on this group, by design.
rope_rel_planes = 2
