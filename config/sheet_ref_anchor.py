from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS sheets cell -- `rope_ref_anchor` (R7b), A REFERENCE TAKES ITS TARGET'S COORDINATE.
#
# THE ARM ROTATES EXACTLY THE PLANES rope_both ROTATES -- 16 sequence + 22x2 tree, with the 4
# token-leaf planes idle in both -- so a `rope_both` run on the SAME pack is the paired baseline
# with NO plane-count confound. Its predecessor `rope_ref_path` had to disclose 64 against 60.
# Geometry, batching and schedule here must be copied to that baseline arm verbatim; the only
# key that may differ between the two is `pos_mode`.
#
# WHAT IT DOES. A token carrying a reference has its own SHEET, ROW and COLUMN path levels
# overwritten by its target's; everything else is rope_both untouched. A reference and the cell
# it names then agree on exactly the planes, and by exactly the amount, that two tokens in one
# cell agree on. Levels below the column are left alone -- they are the token's position WITHIN
# the formula, which a reference still has. It gives up its own address, which for a formula
# operand is the right trade: what matters about `A50` is what it names.
#
# WHY IT REPLACES rope_ref_path. At n=2 that arm is a WASH, not a loss: +0.0075 nats/token on
# seed 0 but -0.0043 on seed 1, paired mean +0.0016, t=+0.27. So the motivation is NOT that the
# mechanism is harmful -- it is that the old measurement was too noisy and too compromised to
# conclude anything from. Its learned parameters do rule out the rope_ref_slot failure
# (step_ref moved 3-6x on the slow planes, so the coordinate was genuinely used). Two design
# faults made it unable to answer the question, and this arm removes both:
#
#   1. NO SHEET IN THE COORDINATE. It carried (row, col) only, so a reference crossing to
#      another sheet coincided with whatever cell held those coordinates on its OWN sheet.
#      Measured over 1,500 workbooks: 17.25% of references are cross-sheet and 25.5% of
#      workbooks hold several sheets. Those references did not fail quietly; they asserted a
#      wrong match. This arm carries (sheet, row, col), which is why the pack was rebuilt.
#   2. THE SIDE EFFECT WAS TWICE THE SIGNAL. A token's own cell coordinate already lives in the
#      tree planes, so for the 63% of tokens with no reference the separate pointer group
#      DUPLICATED it. Two tokens in one cell then agreed on 8 planes (4 tree + 4 pointer) while
#      a reference and its target agreed on 4 and actively DISagreed on the tree planes. The
#      mechanism was half the strength of an artifact of its own construction. Folding the
#      pointer into the tree levels makes the two symmetric.
#
# THE PACK IS sheet_values_v2, and it is NOT comparable to v1. Rebuilt 2026-09-28 with a
# 4,096-token window, over-long workbooks DROPPED rather than truncated, four corrected builder
# defects (byte-based offsets, two ABSENT-sentinel collisions, no int16 clamping) and the new
# tok_ref_sheet.npy axis. Every one of those changes the token stream or node_path, so the
# sheet_both / sheet_seq runs already finished on v1 are NOT a valid baseline for this arm.
# A sheet_both run on THIS pack is required before the comparison means anything.
#
# Measured on the finished pack: 41,245 records survive of 66,015 (62.5%), 43.3M tokens,
# structural density 0.562 nodes per token, and reference coverage is COMPLETE -- 2,991,629 of
# 3,031,723 references have their target in the same record, the remaining 40,094 being
# references to a DEFINED NAME, which denotes no cell at all. Zero references lost to the
# window. sheet_vanilla_v2 holds the SAME 41,245 records in the SAME order (record_ids.json is
# byte-identical), which is what makes the two packs comparable to each other.
pack = f'{PACKS}/sheet_values_v2'
out_dir = f'{RUNS}/sheets/ref_anchor_s0'
pos_mode = 'rope_ref_anchor'
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
