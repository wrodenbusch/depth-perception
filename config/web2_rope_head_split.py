from topos_paths import PACKS, RUNS  # noqa: E402
# TOPOS web cell 2 -- arm rope_head_split.
# Half the HEADS rotate by SEQUENCE across all 64 planes; half rotate by the TREE
# (22 levels x 2 + 4 token = 48 planes, the remaining 16 unrotated). No head shares
# its planes between the two factors, which is the allocation MRoPE-I recommends
# ("dedicate a head per axis") over the narrow contiguous bands rope_both uses.
# Width / budget are filled by throughput.py; learning_rate copied from the other rope arms.
# Override per run: --seed=N --out_dir=$TOPOS_RUNS/<cell>/rope_head_split_sN
out_dir = f'{RUNS}/web2/rope_head_split_s0'
pack = f'{PACKS}/web_c4html_text_gpt2_b4096_v2'
pos_mode = 'rope_head_split'
seed = 0
block_size = 4096
n_layer, n_head, n_embd = 10, 4, 512   # head dim 128; n_head must be EVEN -> heads 0,1 sequence / 2,3 tree
rope_seq_planes, rope_levels, rope_planes_per_level = 64, 22, 2   # seq spans ALL 64 planes; tree 22*2=44
rope_token_planes = 4                   # + 4 token planes = 48 of 64 rotated, 16 left as content channels
bias = False
dropout = 0.0
batch_size = 2                          # micro-batch: 4 OOMed on the graph arms at a full 4096-token batch
gradient_accumulation_steps = 16        # 2 x 16 x 4096 = 131k tokens/step, unchanged
learning_rate = 6e-4                  # the rate every non-'none' rope arm uses
min_lr = 6.0e-05
max_iters = 27220                       # cell-2 budget; cell 7 overrides with --max_iters=537000
lr_decay_iters = 27220
warmup_iters = 544                      # 2%
weight_decay = 0.1
grad_clip = 1.0
beta1, beta2 = 0.9, 0.95
eval_interval = 1361                    # 5% of max_iters -> 20 milestones, every one kept
eval_records = 1024
log_interval = 20
compile = True
