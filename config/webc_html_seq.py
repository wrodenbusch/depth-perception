from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS -- Web-S WITH THE TAGS IN THE TOKEN STREAM ("structure as tokens" control): sequence only.
#
# WHY. A referee's objection: the Web-S text pack keeps only visible text, so its sequence-only
# baseline never sees the page structure, and the structured arms' gain may be "don't delete the
# markup" rather than anything about rotary position. This cell trains on web_tags_b8192
# (build_web_tags_pack.py): the SAME pages, the SAME split and byte-identical visible text as
# web_whole_b2048, with every element written as <tag>...</tag> (no attributes). A sequence-only
# model here can read every open, close and nesting level.
#
# THE COMPARISON THAT MATTERS is on VISIBLE TEXT ONLY, page by page (eval_visible_text.py):
# nats/token over the whole stream is not comparable across a tags pack and a text pack -- ~70%
# of this pack's tokens are markup. The visible-text tokens are identical in both packs (checked
# page by page at build time), so loss on them is directly comparable: this arm against the
# text pack's sequence + depth asks whether READING the tags is as good as being TOLD the depth.
#
# GEOMETRY: identical to Web-S except block_size 8192, the smallest context that keeps whole
# pages (99.6% of Web-S pages fit at 8,192; 32.8% at 2,048, 77.8% at 4,096 -- tags roughly treble
# a page). Never cropped: a page that does not fit is dropped and recorded in meta.json.
# records_per_step and max_iters are unchanged, so each step sees the same 32 pages and a run sees
# the same visible text as a Web-S run; the markup makes each step ~3x the tokens.
#
# LR 1.2e-3 is Web-S's smoked rate, NOT re-swept on this pack (no card was free for a sweep): an
# assumption, stated as one.
out_dir = f'{RUNS}/webc_html/seq_s0'
pack = f'{PACKS}/web_tags_b8192'
pos_mode = 'rope_seq'
seed = 0
block_size = 8192
vocab_size = 50304
n_layer, n_head, n_embd = 6, 2, 256      # head dim 128, so 64 rotary planes; ~4.7M non-embedding
rope_seq_planes, rope_levels, rope_planes_per_level = 16, 22, 2   # levels parsed, NOT rotated
rope_token_planes = 4
rope_rel_planes = 0
rope_decay = False
bias = False
dropout = 0.0
bucket_batching = True
records_per_step = 32                    # same pages per step as Web-S; ~100k tokens/step with markup
token_budget = 8192                      # Web-S's measured best; a max-length page is its own group
batch_size = 2                           # unused under bucket_batching, kept for the record
gradient_accumulation_steps = 16
learning_rate = 1.2e-3                   # Web-S's smoked rate, not re-swept here (see header)
min_lr = 1.2e-4
max_iters = 9375                         # identical to Web-S: same pages, same visible text
lr_decay_iters = 9375
warmup_iters = 188                       # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 469                      # 5% -> 20 milestones
eval_records = 1024
log_interval = 20
compile = False                          # as Web-S: bucket batching produces many shapes and
                                         # torch.compile recompiles per shape
