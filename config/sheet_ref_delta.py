from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS R11b -- `rope_ref_delta`: a reference points at its definition by a CONSTANT VECTOR.
#
# THE IDEA, and whose it is. The author proposed it: stop encoding "are we the same" and encode
# "what relation holds between us", so the DIFFERENCE between two related tokens is a fixed
# vector per relation type. docs/design-reference-rotation.md specifies it at the R11 section
# and `rope_rel` built only half -- the PARENT/CHILD half, as `delta * depth`. This is the
# reference half, which was never built.
#
# WHAT IT ADDS TO THE ARM THAT ALREADY WINS. rope_ref_anchor makes a reference's angle EQUAL
# its target's, so their difference is ZERO. But zero is also the difference between two tokens
# in the SAME cell -- so no attention head can distinguish "the cell I point at" from "my own
# neighbours": they are the same geometric state. R7b fixed the MAGNITUDE of R7's swamping
# problem (no more doubling) and left the CONFUSABILITY. A learned constant separates them:
#
#     angle(reference) = angle_from(target's coordinate) + delta_ref
#     angle(target)    = unchanged
#
# so angle(reference) - angle(target) == delta_ref exactly. That is RotatE inside attention, and
# it recovers two properties coincidence cannot express at all, both free from the group
# structure: DIRECTION (use->def is +delta, def->use is -delta; coincidence is symmetric by
# construction) and COMPOSITION (a two-hop chain is 2*delta, so nothing has to represent
# transitivity).
#
# ONE DEGREE OF FREEDOM, AND IT STARTS AT ZERO. delta_ref initialises to EXACTLY zero, so at
# step 0 this arm is bit-identical to rope_ref_anchor -- asserted by a test, not assumed. It is
# therefore strictly more expressive and can only differentiate. It adds no plane group and no
# allocation key, so it rotates precisely the planes rope_both rotates and there is no fourth
# chance at the train.py plumbing trap.
#
# PAIRED BASELINE IS ref_anchor, which isolates delta: the two arms differ in nothing but the
# constant, on the same pack, same schedule, same seeds. rope_both is the SECONDARY comparison,
# answering "is the whole reference mechanism worth anything" rather than "is the offset worth
# anything on top of coincidence". Quote them separately; ref_anchor already measures -0.0101
# against rope_both (t=-2.37, 4/5 seeds, n=5, going to n=10).
#
# THE DIAGNOSTIC IS THE RESULT. Because delta starts at zero, whether training moves it says
# directly whether the model wanted "referent" to be a different state from "co-located":
#
#     python -c "import rope_pos,json; print(json.dumps(rope_pos.read_ref_delta(
#         '$TOPOS_RUNS/sheets/ref_delta_s0/ckpt.pt'), indent=1))"
#
# delta ~ 0 means coincidence was already the right encoding and the offset bought nothing;
# delta moving of order BASE_STEP (~0.316 rad, one sibling's worth) means it was not. This is
# the third arm diagnosed this way -- theta_slot went 0 -> 1.23 rad on rope_ref_slot, revealing
# a presence flag, and step_ref moved 3-6x on rope_ref_path, excluding that failure.
#
# THE CONFOUND A REVIEWER WILL FIND, stated rather than hidden: this arm holds ONE MORE
# PARAMETER than ref_anchor (6 scalars, delta_ref of shape (3, 2)) against ~4.7M non-embedding.
# It is not a plane-count confound -- the rotating planes are identical -- but it is not
# literally nothing either. A strictly-more-expressive arm that starts bit-identical can also
# only improve on the training objective, so the comparison is fair only because both are
# scored on HELD-OUT data at a FIXED step with no checkpoint selection.
pack = f'{PACKS}/sheet_values_v2'
out_dir = f'{RUNS}/sheets/ref_delta_s0'
pos_mode = 'rope_ref_delta'
seed = 0
block_size = 4096                        # MUST match the pack: v2 keeps whole workbooks up to
                                         # 4,096 tokens (longest observed 3,962 in values,
                                         # 4,096 in vanilla). 1024 here would fail on the
                                         # first over-long record.
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 22*2 + 4 token = 64
rope_token_planes = 4                    # idle here exactly as under rope_both -- NOT spent,
                                         # which is what makes the plane counts match
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0

# NO NEW ALLOCATION KEY. rope_ref_anchor adds no plane group, so unlike rope_slot_planes /
# rope_rel_planes / rope_ref_planes there is nothing extra to plumb through train.py's three
# hand-maintained lists -- the trap that has broken four launches in this project. The arm is
# still covered by test_every_mode_survives_a_round_trip_through_gptconfig.

# BATCHING. Must match whatever the sheet_both arm on THIS pack uses, so the two see the same
# records in the same order at a seed. 32 records x the v2 pack's 1,050-token mean is ~33,600
# target tokens per step -- MEASURED 39,970 on a real step, since bucket_batching draws whole
# records and their lengths vary. That is well above the ~26,500 the 1,024-window pack gave.
bucket_batching = True
records_per_step = 32
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the record
gradient_accumulation_steps = 16

# SCHEDULE RECOMPUTED FOR THIS PACK. Dropping over-long workbooks at a 4,096 window leaves
# n_train = 32,997 records, not the 51,440 the 1,024-window pack had, so one pass is 1,031.2
# steps and the old 12,880 would have been 12.5 passes rather than 8.
learning_rate = 1.2e-3                   # ASSUMPTION, no longer a measurement -- see below
min_lr = 1.2e-04
max_iters = 8240                         # 8 passes: n_train 32,997 / 32 = 1,031.2 steps/pass
lr_decay_iters = 8240
warmup_iters = 165                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 412                      # 5% -> exactly 20 milestones, every one kept

# THE LEARNING RATE IS NOW UNSMOKED FOR THIS GEOMETRY, and that is a change from what an
# earlier draft of this file claimed. 1.2e-3 was smoked on the 1,024-window pack at ~26,500
# target tokens per step (6e-4 -> 2.2881, 1.2e-3 -> 2.1694, 2.4e-3 -> 2.3653; an interior
# minimum -- docs/sheets-lr-smoke.md). The window is now 4,096 and the step carries ~40,000
# tokens, so that smoke no longer transfers: a larger batch generally wants a somewhat HIGHER
# rate, which means 1.2e-3 is more likely too low than too high. It is the defensible default
# and nothing more. Re-smoke on this pack before treating any result here as converged.
eval_records = 1024
log_interval = 20
compile = True

# COVERAGE, measured on the finished pack rather than estimated.
#
# Dropping instead of truncating removed the problem the previous draft had to disclose here:
# no reference loses its target to the window any more. 98.68% of references carry a usable
# pointer; the residue is references to a defined name, which has no grid location and is a
# permanent property of the corpus rather than a window artifact.
#
# What to state in any writeup instead: the surviving population is SELECTED. 23,055 workbooks
# (34.9%) were dropped for exceeding 4,096 tokens under the longer of the two renders, and the
# survivors are markedly smaller -- 25.7 rows per workbook against 182.8 for the dropped ones,
# a 7.1x difference. So results speak to small and medium workbooks, not to the corpus as a
# whole, and a null here would not license a claim about large spreadsheets.
