from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS sheets cell -- `rope_seq` on the pack that SPELLS EVERY ADDRESS OUT AS TEXT.
#
# The tokens-versus-rotation arm. This reads `sheet_vanilla`, where each cell is rendered
# '<address>:<content>' -- SpreadsheetLLM's own baseline serialisation (Dong et al., EMNLP
# 2024, arXiv:2407.09025) -- so the model is TOLD where each cell is, in words, and needs no
# rotation to know it. Against the tree arm on `sheet_values` it asks whether rotating a
# coordinate beats writing it down.
#
# SCORING CAVEAT, and it is not small. This pack has a DIFFERENT token stream from
# `sheet_values`: 56.9M tokens against 52.8M, because the addresses are extra words, and they
# are highly predictable ones. So nats-per-TOKEN is NOT comparable across the two packs -- an
# arm can look better simply by having more easy tokens to average over. The comparison that
# means anything here is nats per WORKBOOK: how many nats each representation spends to encode
# the same spreadsheet. Both packs hold the same 64,300 records in the same order with the same
# split, so that comparison is well defined; it is just a different yardstick from the rest of
# the project, and any writeup must say which one it is quoting.
#
# WHY A SPREADSHEET CELL. The project's claim is about structural position, and the two cells
# built so far sit at opposite ends of structural density: Python AST at 0.86 nodes per token,
# web DOM at 0.12. Spreadsheets land at 0.56 -- between them, and with a kind of structure
# neither has: a 2-D grid whose cells NAME each other. If the effect tracks density, this cell
# should fall between the two; if it tracks something else, this is where that shows.
#
# THE PACK. `sheet_values` renders each cell's CONTENT only -- the literal value, or the full
# infix formula including operators, function names and reference addresses -- with the cell's
# OWN address withheld, because that is exactly what the tree rotation supplies. The paired
# `sheet_vanilla` pack spells the address out as text instead (see config/sheet_vanilla_seq.py).
# Reference addresses INSIDE formulas are spelled in both packs, so no arm has a pointer the
# others cannot see.
#
# Built by build_sheet_pack.py and validated against the corpus's OWN decoder: 51,541 cells
# over 120 workbooks render byte-for-byte identically to `record_to_decoded` + `render_text`
# in docs/sheet_graph_corpus_ref.py.reference. That check exists because the first version of
# the renderer read cell text from `payloads`, which lives only on `Lit` nodes, so every
# formula rendered as the EMPTY STRING and nothing noticed.
#
# BLOCK SIZE 1024, measured rather than assumed. At 1024, 61% of workbooks are cut and only
# 10.5% of the corpus's references reach a token; at 4096 that becomes 30% and 29.8%. But
# 17.0% of TOKENS carry a reference coordinate at 1024 -- denser than any other pack -- so the
# 4096 gain is corpus utilisation, not training signal, and it would cost a 15-24 h run per arm
# against 1024's ~3.5 h. The cut is recorded in the pack manifest under `truncation`.
#
# GEOMETRY is the Python cell's, unchanged, so head_dim stays 128 and the 64-plane allocation
# is comparable cell to cell. vocab_size is NOT set here: train.py reads 50304 from the pack's
# meta.json (GPT-2, pretrained), and stating it twice is how a config drifts from its data.
out_dir = f'{RUNS}/sheets/vanilla_seq_s0'
pack = f'{PACKS}/sheet_vanilla'
pos_mode = 'rope_seq'
seed = 0
block_size = 1024
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # 16 + 22*2 + 4 token = 64
rope_token_planes = 4                    # the token-leaf level; rotates only under rope_leaf
rope_decay = False                       # the corrected setting; see model.py
bias = False
dropout = 0.0

# BATCHING. Records are whole workbooks and their lengths vary enormously, so micro-batches are
# formed by sorting each step's records by length (topos_data.step_batches) -- the gradient is
# what an unpadded run would produce, and arms at one seed see identical records in identical
# order. `records_per_step` is what sets the batch: 32 x the pack's 821-token mean is ~26,300
# target tokens per step, matching the Python cell's ~27,000 rather than inheriting its record
# count, whose records are 105 tokens each.
bucket_batching = True
records_per_step = 32                    # same RECORDS per step as the other arms, so the
                                         # two packs are compared over identical workbooks
token_budget = 8192                      # caps len(group) x padded_len; up to 8 full-length records
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
