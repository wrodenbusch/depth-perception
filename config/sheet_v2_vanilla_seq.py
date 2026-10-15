from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS sheets cell v2 -- `rope_seq` on the pack that SPELLS EVERY ADDRESS OUT AS TEXT.
#
# The tokens-versus-rotation arm. SCORING CAVEAT, unchanged and still not small: this pack has
# a different token stream (57.8M against 43.3M), and the extra tokens are the easy, highly
# predictable ones, so nats-per-TOKEN is NOT comparable to the other arms. Both packs hold the
# SAME 41,245 records in the same order -- record_ids.json is byte-identical -- so nats per
# WORKBOOK is well defined and is this arm's only valid yardstick. chart_ablation.py refuses a
# mixed-pack axis outright; do not override it for this arm.
pack = f'{PACKS}/sheet_vanilla_v2'
out_dir = f'{RUNS}/sheets/vanilla_seq_v2_s0'
pos_mode = 'rope_seq'
seed = 0
block_size = 4096                        # MUST match the pack: v2 keeps whole workbooks up to
                                         # 4,096 tokens (longest observed 3,962 in values,
                                         # 4,096 in vanilla). 1024 here would fail on the
                                         # first over-long record.
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 22*2 + 4 token = 64
rope_token_planes = 4                    # idle under this mode; declared so the allocation
                                         # matches every other v2 arm exactly
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
