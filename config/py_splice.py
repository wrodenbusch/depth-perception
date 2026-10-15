from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS R7d -- `py_splice`: the CORRECTED Python anchor. Prefix only, path length preserved.
#
# WHY R7c LOST, AND WHY THIS IS NOT THE SAME ARM. py_anchor replaced a use's WHOLE path with its
# binder's, and measured +0.0025 against rope_both (t=+3.30, worse on 4 of 5 seeds) -- the
# opposite sign to the spreadsheet arm it was meant to mirror. That was my error, not the data's.
# Binders sit far shallower than uses (median depth 3 against 5), so whole-path replacement
# TRUNCATED 91.7% of substituted paths, losing 3.13 levels on average. Those levels became
# ABSENT, which contributes exactly zero rotation -- so the arm threw away the use's own position
# AND recreated the absent-class degeneracy on the deep planes, the very failure that sank
# rope_ref_slot.
#
# The sheets arm never does this: its targets are cells at a FIXED depth, it overwrites levels
# 0-2 only, and path length is preserved exactly. This arm restores that shape -- the binder's
# path in the PREFIX (the referent's identity), the use's own levels from that depth onward (its
# own role). Measured: 0% truncated, 100% of path lengths preserved, against 91.7% truncated
# before.
#
# The pointers were never the problem: use and binder share an identity hash 89% of the time,
# ABOVE the 70.43% the pack documents, so node_ref links the right nodes. Only the geometry was
# wrong.
#
# THE POINT. This is the Python version of the spreadsheet arm that works. There a reference
# overwrites its own sheet/row/column path levels with its target's; here a variable use
# overwrites its whole tree path with its resolved binder's. A use and its definition then land
# on the same rotary angle, on the same planes, by exactly the amount two tokens in one AST node
# land on. Nothing is added to the geometry.
#
# WHY THERE IS NO NEW pos_mode. The loader builds a token's path by gathering
# node_path[tok_node[t]], so the entire mechanism is a SUBSTITUTION OF tok_node -- done at pack
# build time by build_py_anchor_sidecar.py. So this arm is literally `rope_both`, unchanged,
# reading a sidecar whose tok_node points at the binder. No new mode, no new plane group, no
# model edit, and therefore no possibility of a rotation bug distinguishing it from its baseline.
#
# THE BASELINE IS ALREADY RUN. `rope_both_s0..s4` on $TOPOS_PACKS/py_ast_423k, complete at
# n=5, same model, same tokens, same schedule, same seeds. The two packs differ in ONE array and
# that difference IS the mechanism, which makes this the cleanest comparison in the project.
# tokens.npy / tok_off.npy / meta.json are symlinks to the baseline pack, so the token stream is
# not merely equivalent, it is the same bytes.
#
# WHY THE DESIGN DOC RULED THIS OUT, AND WHY THAT IS VOID. docs/design-reference-rotation.md
# rejected a path anchor for Python and chose a 2-plane symbol SLOT, because a full 22-level path
# looked like it needed a separate ~42-plane group. OVERWRITING needs zero extra planes, so the
# objection does not apply -- and the slot arm is the one that failed: on ref_slot_s0 the model
# drove theta_slot from exactly 0 to 1.23 rad (a presence flag) while step_slot, which
# discriminates WHICH symbol, barely moved, with 57% of tokens carrying no slot and mutually
# coincident.
#
# COVERAGE AND CAUSALITY, measured. 9,712,560 of 47,108,791 tokens are substituted (20.6%),
# matching the corpus's own 9,707,041 backward reference edges. The substitution reads `node_ref`
# -- "the pyscope-resolved canonical binder of a use, -1 when the target is emitted later" -- so
# the corpus itself drops the 139,028 forward edges and every surviving one points backward. Over
# 4,000 records, 100% of referents were resolved strictly before the referring token. A token
# whose node has no binder keeps its own path, so unreferenced tokens are geometrically identical
# to rope_both and this axis has no absent class at all.
#
# NOT built from tok_slot, which I tried first and which would have been wrong: it is a CARRY
# signal ("the slot of the most recently RESOLVED symbol"), so 27% of tokens carry a slot they are
# not an occurrence of, and substituting on it would have moved a quarter of all tokens to
# whichever variable was mentioned most recently.
out_dir = f'{RUNS}/py/splice_s0'
pack = f'{PACKS}/py_ast_423k_splice'
pos_mode = 'rope_both'                   # unchanged; the mechanism is in the sidecar
seed = 0
block_size = 1024
vocab_size = 32768
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 44 + 4 = 64 = head_dim/2
rope_token_planes = 4
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 256                   # ~27,000 target tokens/step; one pass = 1,394 steps
token_budget = 8192
batch_size = 2                           # unused under bucket_batching, kept for the config record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                   # the baseline's smoked rate, and the same batch shape;
                                         # nothing about the token stream changed, so it transfers
min_lr = 1.2e-04
max_iters = 11200                        # 8 passes, identical to the rope_both baseline
lr_decay_iters = 11200
warmup_iters = 224                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 560                      # 5% -> 20 milestones, all kept
eval_records = 1024
log_interval = 20
compile = True
