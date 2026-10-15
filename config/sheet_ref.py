from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS sheets cell -- R7 `rope_ref_path`, THE GRID POINTER: rotate by what a cell REFERS TO.
#
# WHAT IT ADDS OVER `sheet_both`. Every arm before this one rotates by where a token IS. This
# one adds a group that rotates by the (row, column) of the cell a reference NAMES. A rotary
# score depends on the DIFFERENCE of two angles, so a reference and its target become angularly
# coincident -- "the cell this formula points at" and "that cell" sit at the same angle, exactly
# as two tokens at the same sequence position do.
#
# Spreadsheets are the easy case for this, and the reason to test it here first: THE NAME IS THE
# LOCATION. `A1` means row 1, column A, readable from the reference text itself whether or not
# that cell has been emitted yet. So there is no interning, no slot table, no first-mention
# bookkeeping and no dropped edges -- just the referent's actual coordinate, and it costs 4
# planes rather than a 22-level path. See docs/design-reference-rotation.md, "Spreadsheets are
# the exception, and the easy case".
#
# ONE GROUP, WRITTEN BY BOTH SIDES -- the property the arm lives or dies on. Rotary position
# only ever compares THE SAME PLANE INDEX between query and key, so a reference coordinate in a
# group of its own would have the query rotating planes the target never touches, and the two
# would never interact: it would encode nothing while looking entirely reasonable. Instead there
# is a single group and EVERY token writes a (row, column) into it -- its target's if it carries
# a reference, its own enclosing cell's otherwise, read from path levels 1 and 2. Both sides run
# the same arithmetic through the same step/theta, so the coincidence is bit-exact by
# construction rather than by arrangement.
#
# MEASURED on this pack, 4.0M tokens over 4,862 records:
#   17.07%  carry a reference           -> write the TARGET's coordinate
#   63.39%  carry no reference          -> write their OWN cell's coordinate
#   19.35%  have no coordinate at all   -> contribute EXACTLY zero, as an absent tree level does
# That last figure is the point of comparison with `rope_ref_slot`, where 57% of tokens had no
# slot, all mutually coincident, and the model spent the group's theta on a crude presence flag
# instead of on identity (theta_slot 0 -> 1.23 rad while step_slot barely moved). Here four
# fifths of tokens carry a real coordinate, so there is far less for a presence flag to buy.
#
# theta_ref exists and starts at exactly zero, as every other group's does, but note what it can
# and cannot do: it is added with present=1 for every token that HAS a coordinate, so among
# those it is a constant and cancels in every query-key difference. It can only separate in-cell
# tokens from tokens with no cell.
#
# THE CONFOUND A REVIEWER WILL FIND, stated rather than hidden. Against its paired baseline
# `sheet_both` this arm rotates 64 planes to that arm's 60: `sheet_both` allocates 4 token-leaf
# planes that sit at angle 0 because only `rope_leaf` rotates them, and this arm spends those 4
# on the pointer. So 4 extra ROTATING planes are a real difference beyond the pointer itself.
# Probably small -- four extra rotating planes were worth nothing when measured directly
# (`rope_all` vs `rope_both`: +0.0001, t=+0.12, n=5) -- but not zero.
#
# KNOWN IMPURITY, measured. The trailing defined-names table hangs off a `Names` node that is
# also a depth-1 child of the root, so a defined name sits at the path level a ROW occupies and
# its target at the level a COLUMN occupies. Those tokens therefore report a spurious cell
# coordinate rather than none. Upper bound 0.469% of rendered characters before truncation cuts
# most of it away (the section is last, and 61% of workbooks are cut). Fixing it properly needs
# a marker from build_sheet_pack.py; not worth a 2.5 h rebuild of both packs for 0.5%.
#
# PAIRED BASELINE IS `sheet_both` (config/sheet_both.py), same pack, same seed, same records in
# the same order. The only difference is that 4 planes carry the grid pointer instead of sitting
# idle, so the comparison isolates the pointer axis.
#
# PRIOR ART to cite and distinguish: RotatE (Sun et al., ICLR 2019, arXiv 1902.10197) makes each
# relation a rotation from source to target, but as a knowledge-graph link-prediction embedding,
# not a positional encoding in a transformer. Position Coupling (Cho et al., NeurIPS 2024,
# Appendix B.4) has a rotary variant with exact cancellation for shared ids -- but it is NOT
# partitioned: coupled ids REPLACE position in every head of every layer. The claim here is the
# partitioning: a pointer coordinate on its own plane group, coexisting with ordinary sequence
# position and structural rank on disjoint groups inside every head.
out_dir = f'{RUNS}/sheets/ref_s0'
pack = f'{PACKS}/sheet_values'
pos_mode = 'rope_ref_path'
seed = 0
block_size = 1024
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 seq + 22*2 tree + 4 ref = 64
rope_token_planes = 0                    # refused under rope_ref_path: the arm isolates the pointer
rope_ref_planes = 4                      # 2 planes for the ROW axis, 2 for the COLUMN axis
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0

# BATCHING, identical to the other sheets arms so the packs are compared over identical
# workbooks in an identical order. See config/sheet_seq.py for why records_per_step is 32.
bucket_batching = True
records_per_step = 32
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the config record
gradient_accumulation_steps = 16         # likewise: records_per_step overrides it

learning_rate = 1.2e-3                   # SMOKED on this pack, 644 steps: 6e-4 -> 2.2881,
                                         # 1.2e-3 -> 2.1694, 2.4e-3 -> 2.3653 val nats/tok. An
                                         # interior minimum, so the grid needs no extending.
                                         # docs/sheets-lr-smoke.md
min_lr = 1.2e-04
max_iters = 12880                        # 8 passes: n_train 51,440 / 32 records = 1,607.5 steps/pass
lr_decay_iters = 12880
warmup_iters = 258                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 644                      # 5% -> 20 milestones, every one kept
eval_records = 1024
log_interval = 20
compile = True
