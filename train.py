"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
from topos_paths import PACKS  # noqa: E402

import os
import time
import math
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from rope_pos import (TREE_MODES, SLOT_MODES, REF_COORD_MODES, REF_ANCHOR_MODES,
                      PLAN_MODE, plan_needs_path)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
pos_mode = 'learned' # TOPOS: 'learned' | 'none' | 'tree' | 'rope_seq' | 'rope_graph' | 'rope_both' -- see architecture.md
# TOPOS: the pack to read (tokens.npy / tok_off.npy / meta.json, plus a sidecar
# when the arm needs one). Defaults to the Python experiment's pack.
pack = f'{PACKS}/packedwindows_wholefile_apps_kodcode_astcompact_ident32k_b1024'
seed = 0 # TOPOS: seeds the model init AND the batch order, so every arm at seed s sees identical batches
eval_records = 1024 # TOPOS: the first eval_records val records, fixed, scored at every eval
# TOPOS: rope plane allocation (rope arms only); must sum to head_dim/2
rope_seq_planes = 16
rope_levels = 24
rope_planes_per_level = 2
rope_token_planes = 0          # cell 2: planes on the within-node token index (rope_leaf); 0 = none
rope_decay = True              # cell 3: False takes rope.step/rope.theta out of the weight-decay group (see model.py)
rope_layer_plan = ''           # rope_layer_plan arm: one char per layer from ('s','t','n'); see rope_pos.py
rope_slot_planes = 0           # rope_ref_slot arm: planes on the SYMBOL SLOT (first-mention coreference); 0 = none
rope_rel_planes = 0            # RELATIONAL DEPTH planes (parent/child = one constant step). Read by
                               # rope_rel (R11a, depth ON TOP of the path) and rope_depth (depth
                               # INSTEAD of it); 0 = none
rope_ref_planes = 0            # rope_ref_path arm (R7): planes on the GRID POINTER, the referent's (row, col); 0 = none
depth_emb = False              # control: depth as a learned INPUT embedding on top of rope_seq (see model.py)
# TOPOS cell 4: length-bucketed micro-batches. Records per optimizer step stays fixed
# (batch_size * gradient_accumulation_steps by default); how they are grouped does not.
bucket_batching = False
records_per_step = 0           # 0 -> batch_size * gradient_accumulation_steps
token_budget = 8192            # caps len(group) * padded_len, so peak memory matches a fixed micro-batch of 2 at block 4096
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
lr_restart_at = 0 # TOPOS: warm restart -- the schedule (warmup + cosine, lr_decay_iters long) is measured from this iteration; used to extend a finished run in place
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# TOPOS (2026-09-24): the modern spelling of the line above -- it also covers the
# float32 ops torch routes outside the legacy tf32 flag.
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Refuse the MATH fallback for attention. With bucket_batching the (B, T) shapes
# vary far more than before, and a shape the fast kernels decline is served by the
# materialise-the-whole-matrix path -- correct, but quadratic in memory and far
# slower, and it would show up only as an unexplained slowdown mid-run.
if device_type == "cuda" and hasattr(torch.nn.attention, "sdpa_kernel"):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _sdpa = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                         SDPBackend.EFFICIENT_ATTENTION])
    _sdpa.__enter__()
    print("sdpa: flash / cudnn / mem-efficient only (math fallback refused)")

# TOPOS data loader. Whole packed windows, with tree coordinates when the arm
# needs them. NOT a random offset into a flat stream: see topos_data.py for why
# that would straddle window boundaries AND mix train with val.
from topos_data import PackedWindows
# TOPOS: which sidecar this arm needs. Derived from rope_pos.TREE_MODES rather than a
# hardcoded list, because a hardcoded list drifts: the split arms were added without it and
# crashed with "needs a DOM path" on their first launch (caught 2026-09-26 by an LR smoke).
if pos_mode == 'tree':
    _sidecar = 'tree'
elif pos_mode in TREE_MODES:
    _sidecar = 'path'
elif pos_mode == PLAN_MODE:
    _sidecar = 'path' if plan_needs_path(rope_layer_plan) else None
else:
    _sidecar = None
if depth_emb:                       # the input-embedding depth control reads the same path sidecar
    _sidecar = 'path'
# Derived, not hardcoded, for the same reason as _sidecar: rope_ref_slot additionally needs
# tok_slot.npy, and the loader REFUSES without it rather than silently degrading to rope_both.
_need_slot = pos_mode in SLOT_MODES
# Same again for the reference arms, which need tok_ref_row/tok_ref_col: the loader REFUSES
# without them rather than silently degrading to rope_both. rope_ref_anchor additionally
# needs tok_ref_sheet -- without it a cross-sheet reference points at the wrong cell.
_need_ref = pos_mode in REF_COORD_MODES
_need_ref_sheet = pos_mode in REF_ANCHOR_MODES
_pw = PackedWindows(pack, need_tree=False, sidecar=_sidecar, need_slot=_need_slot,
                    need_ref=_need_ref, need_ref_sheet=_need_ref_sheet)
print(f"topos: {_pw}  arm={pos_mode}")
# The batch order is a function of the seed alone, so arms at one seed see the
# same records in the same order.
_gen = torch.Generator().manual_seed(seed + seed_offset)
def get_batch(split):
    return _pw.get_batch(split, batch_size, torch.device(device), generator=_gen)

if records_per_step <= 0:
    records_per_step = batch_size * gradient_accumulation_steps

def _materialise(groups):
    """Turn record-ordinal groups into (x, y, pos, n_target_tokens) micro-batches."""
    out = []
    for grp in groups:
        xb, yb, pb = _pw.get_batch_records(grp, torch.device(device))
        out.append((xb, yb, pb, int((yb != -1).sum())))
    return out

# nanoGPT's original flat-stream loader, kept for reference and unused here.
data_dir = os.path.join('data', dataset)
def _get_batch_flatstream(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  pos_mode=pos_mode, rope_seq_planes=rope_seq_planes, rope_levels=rope_levels,
                  rope_planes_per_level=rope_planes_per_level, rope_decay=rope_decay, rope_token_planes=rope_token_planes,
                  rope_layer_plan=rope_layer_plan, rope_slot_planes=rope_slot_planes,
                  rope_rel_planes=rope_rel_planes, rope_ref_planes=rope_ref_planes,
                  depth_emb=depth_emb, bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size or _pw.vocab_size or 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'pos_mode', 'rope_seq_planes', 'rope_levels', 'rope_planes_per_level', 'rope_decay', 'rope_token_planes',
              'rope_layer_plan', 'rope_slot_planes', 'rope_rel_planes', 'rope_ref_planes', 'depth_emb']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    if model_args['pos_mode'] != pos_mode:
        raise ValueError(f"checkpoint in {out_dir} is arm {model_args['pos_mode']!r}, not {pos_mode!r}")
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Advance the batch generator to where the checkpoint left off, so a resumed
    # run sees the same record sequence an unbroken run would have. Draws the
    # exact call get_batch makes, once per micro-step already taken.
    _lo, _hi = _pw.splits['train']
    for _ in range(iter_num * gradient_accumulation_steps):
        torch.randint(_lo, _hi, (batch_size,), generator=_gen)
    print(f"resumed at iter {iter_num}; batch generator advanced {iter_num * gradient_accumulation_steps} draws")
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# TOPOS: a FIXED validation set, not random batches. The first eval_records val
# records in fixed batches, identical for every arm and every eval, so a milestone
# curve is comparable across arms and across steps.
@torch.no_grad()
def evaluate_fixed_val():
    """Nats per token (token-weighted over non-pad targets) and nats per record."""
    model.eval()
    lo, hi = _pw.splits['val']
    hi = min(hi, lo + eval_records)
    tot, n_tok, n_rec = 0.0, 0, 0
    for start in range(lo, hi, batch_size):
        ix = list(range(start, min(start + batch_size, hi)))
        X, Y, POS = _pw.get_batch_records(ix, torch.device(device))
        with ctx:
            logits, loss = model(X, Y, pos=POS)
        nt = int((Y != -1).sum())
        tot += float(loss) * nt
        n_tok += nt
        n_rec += len(ix)
    model.train()
    return tot / n_tok, tot / n_rec

def _log(rec):
    with open(os.path.join(out_dir, 'train.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')

def eval_and_checkpoint(step, lr):
    """Score the fixed val set, log it, write a model-only milestone and the resumable latest."""
    global best_val_loss
    v_tok, v_page = evaluate_fixed_val()
    print(f"step {step}: val nats/tok {v_tok:.4f} nats/page {v_page:.2f}")
    _log(dict(step=step, val_nats_tok=v_tok, val_nats_page=v_page, lr=lr, elapsed_s=time.time() - _t_start))
    if wandb_log:
        wandb.log({"iter": step, "val/nats_tok": v_tok, "val/nats_page": v_page, "lr": lr, "mfu": running_mfu*100})
    if step > 0:
        torch.save(dict(model=raw_model.state_dict(), model_args=model_args, iter_num=step, val_nats_tok=v_tok),
                   os.path.join(out_dir, f'ckpt_{step}.pt'))
        best_val_loss = min(best_val_loss, v_tok)
        torch.save(dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(), model_args=model_args,
                        iter_num=step, best_val_loss=best_val_loss, config=config),
                   os.path.join(out_dir, 'ckpt.pt'))
        print(f"saving checkpoint to {out_dir}")

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    it = it - lr_restart_at   # TOPOS: a warm restart re-runs warmup + cosine from lr_restart_at
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, POS = get_batch('train') # fetch the very first batch
GROUPS, STEP_TOKENS, MICRO = [], 1, []
if bucket_batching:
    GROUPS, STEP_TOKENS = _pw.step_batches('train', records_per_step, token_budget, generator=_gen)
    MICRO = _materialise(GROUPS)
    print(f"bucket_batching: {records_per_step} records/step, token_budget {token_budget}, "
          f"first step -> {len(GROUPS)} micro-batches, {STEP_TOKENS:,} target tokens")
t0 = time.time()
_t_start = t0
_last_eval_step = -1
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        eval_and_checkpoint(iter_num, lr)
        _last_eval_step = iter_num
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with gradient accumulation over the step's micro-batches.
    #
    # TOPOS (2026-09-24): with bucket_batching the micro-batches are formed by SORTING
    # the step's records by length, so each one is padded to roughly its own length
    # instead of to the longest page in a random pair. Measured on this pack a random
    # micro-batch of 2 is 72% useful and a bucketed one is 98%.
    #
    # Each micro-batch's loss is then weighted by ITS SHARE OF THE STEP'S TARGET TOKENS,
    # not by 1/n_micro_batches. That makes the accumulated gradient exactly the gradient
    # of the mean loss over every target token in the step -- independent of how the
    # records happened to be grouped, which is what lets the grouping change for speed
    # without changing the result. (Fixed-size accumulation weights each micro-batch
    # equally, so it silently up-weights tokens that land in short batches.)
    step_loss = 0.0
    if bucket_batching:
        n_micro = len(GROUPS)
        for micro_step, (Xg, Yg, POSg, n_tok) in enumerate(MICRO):
            if ddp:
                model.require_backward_grad_sync = (micro_step == n_micro - 1)
            with ctx:
                logits, loss = model(Xg, Yg, pos=POSg)
                loss = loss * (n_tok / STEP_TOKENS)
            step_loss += loss.item()          # already token-weighted: this IS the step's mean loss
            scaler.scale(loss).backward()
        GROUPS, STEP_TOKENS = _pw.step_batches('train', records_per_step, token_budget, generator=_gen)
        MICRO = _materialise(GROUPS)
    else:
      for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, pos=POS)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, POS = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = step_loss if bucket_batching else loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        _log(dict(step=iter_num, loss=lossf, lr=lr, dt_ms=dt*1000))
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# The last step is always a milestone, even when max_iters is not a multiple of eval_interval.
if master_process and _last_eval_step != max_iters and not eval_only:
    eval_and_checkpoint(max_iters, lr)

if ddp:
    destroy_process_group()
