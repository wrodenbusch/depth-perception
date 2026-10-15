"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from tree_pos import TreeAddEmbedding
from rope_pos import GraphRoPE, apply_rotary, ROPE_MODES

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, rot=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if rot is not None:  # TOPOS: rotary position on q and k (the rope arms)
            cos, sin = rot
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, rot=None):
        x = x + self.attn(self.ln_1(x), rot)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # TOPOS: the ONLY thing that differs between the arms of this project.
    #   'learned'     stock nanoGPT -- one learned row per sequence index (CONTROL
    #                 of the Python experiment)
    #   'none'        no positional input at all; ordering comes from the causal
    #                 mask alone. A real floor, not a broken arm -- see tree_pos.py
    #   'tree'        additive learned TREE coordinates over NODE ordinals, replacing
    #                 the sequence table entirely (TREATMENT of the Python experiment)
    #   'rope_seq'    rotary position on the SEQUENCE planes only   (web cell control)
    #   'rope_graph'  rotary position on the TREE planes only       (web cell treatment)
    #   'rope_both'   both ranges rotate                            (web cell interaction)
    #   'rope_leaf'   tree planes + the token-leaf planes rotate     (web cell 2: every token a leaf)
    #   'rope_all'    sequence AND tree AND token-leaf planes rotate (the true superset arm)
    #   'rope_head_split'   half the HEADS rotate by sequence at FULL width, half by the
    #                 tree at full width -- no band-sharing inside a head
    #   'rope_layer_split'  EVEN layers rotate by sequence at full width, ODD layers by
    #                 the tree; the sequence-vs-tree analogue of Llama 4's iRoPE
    # The three rope arms and 'none' form a 2x2 over two position sources, each
    # owning a fixed range of rotary planes in every arm -- see rope_pos.py. The two
    # SPLIT arms instead give a whole head or a whole layer to one factor at full width,
    # which is the allocation the multi-axis RoPE literature recommends over narrow bands.
    # Only 'learned' creates the wpe table; only 'tree' creates the tree tables;
    # only the rope arms create the rotary module. Nothing else can hold
    # positional parameters or receive gradients by accident.
    pos_mode: str = 'learned'
    # Rope plane allocation. Must sum to head_dim/2 under a rope mode, or the
    # model refuses to build (a partial allocation is not silently accepted).
    rope_seq_planes: int = 16
    rope_levels: int = 24
    rope_planes_per_level: int = 2
    rope_token_planes: int = 0      # cell 2: the token-leaf level (rope_leaf); 0 = no such planes
    # TOPOS: under 'rope_layer_plan', one character per layer from ('s','t','n') -- sequence
    # at full width, tree at full width, or no rotation. Length must equal n_layer. See
    # rope_pos.py; 'ststst' reproduces rope_layer_split on 6 layers.
    rope_layer_plan: str = ''
    # TOPOS: under 'rope_ref_slot', how many planes carry the SYMBOL SLOT -- the rank, in
    # first-mention order, of the symbol a token refers to. All mentions of one symbol share
    # a slot, so they become angularly coincident in these planes. 2 is enough: measured
    # slots per record are median 8 / max 75 against a slow plane that counts to ~318. Take
    # them from rope_token_planes, which is worth nothing when sequence rotates.
    rope_slot_planes: int = 0
    # TOPOS: under 'rope_rel' (R11a), how many planes carry the DIRECTED RELATION -- the
    # token's tree DEPTH, so a parent and child differ by exactly one step and a grandparent
    # by two. Unlike the slot group there is no presence term: every token has a depth, so
    # there is no absent class. Additional to the per-level rank planes, never a replacement
    # (depth alone would flatten every node at one depth). See rope_pos.REL_MODE.
    rope_rel_planes: int = 0
    # TOPOS: under 'rope_ref_path' (R7), how many planes carry the GRID POINTER -- the
    # referent's actual (row, column), split evenly between the two axes. ONE group, written by
    # BOTH sides: a token holding a reference writes its TARGET's coordinate, every other token
    # writes its own enclosing cell's, so the two land on the same angle. A private group for
    # references would encode nothing, because rotary position only compares matching plane
    # indices. See rope_pos.REF_PATH_MODE.
    rope_ref_planes: int = 0
    # TOPOS: whether the ROTATION parameters (rope.step, rope.theta) are weight-decayed.
    # True reproduces the cell-2 runs of 2026-09-21/22, which inherited nanoGPT's "every
    # parameter with >= 2 dims decays" rule -- a rule written for matmul weights. Those
    # tables are (levels, planes_per_level), so they landed in the decay group and were
    # pulled toward zero, i.e. toward NO ROTATION, at ~0.41 of their initial size over
    # four passes and ~0.07 over twelve. That headwind applies only to the arms that HAVE
    # rotation parameters -- the graph arms -- while the sequence control (whose angles
    # come from a buffer) is untouched, so it biases the comparison one way and grows with
    # schedule length. False is the corrected setting. Left as a flag, not a silent fix,
    # so the banked runs stay reproducible.
    rope_decay: bool = True
    # TOPOS: DEPTH AS AN INPUT EMBEDDING -- the control for the rotary depth marker. A learned
    # table indexed by the token's tree depth (0 = outside the tree, up to rope_levels), ADDED to
    # the token embedding, with sequence position on the usual rope_seq planes. The depth is
    # computed exactly as rope_depth computes it (count of non-ABSENT levels in the path), so
    # the two arms see the same information and differ only in where it enters the model.
    # Only valid with pos_mode='rope_seq'. Zero-initialised, so at step 0 the arm IS rope_seq.
    depth_emb: bool = False

POS_MODES = ('learned', 'none', 'tree') + ROPE_MODES

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        if config.pos_mode not in POS_MODES:
            raise ValueError(f"pos_mode {config.pos_mode!r} is not one of {POS_MODES}")
        self.config = config
        self.pos_mode = config.pos_mode

        parts = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_mode == 'learned':
            parts['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(parts)
        # The tree tables live outside `transformer` so that `wpe`-shaped code
        # paths (crop_block_size, get_num_params) stay honest about what exists.
        self.tree_pos = TreeAddEmbedding(config.n_embd) if config.pos_mode == 'tree' else None
        # The rotary module, only under a rope arm. Its init draws no random
        # numbers, so the block stack below is bit-identical across the four
        # web arms at a seed.
        self.rope = None
        if config.pos_mode in ROPE_MODES:
            self.rope = GraphRoPE(config.n_embd // config.n_head, config.pos_mode,
                                  config.rope_seq_planes, config.rope_levels,
                                  config.rope_planes_per_level,
                                  token_planes=config.rope_token_planes,
                                  n_head=config.n_head,
                                  layer_plan=config.rope_layer_plan,
                                  n_layer=config.n_layer,
                                  slot_planes=config.rope_slot_planes,
                                  rel_planes=config.rope_rel_planes,
                                  ref_planes=config.rope_ref_planes)
        # rope_layer_split is the only arm whose rotation differs BY LAYER, so it is the
        # only one that needs the block loop to select. A static Python bool, so compile
        # specialises it away.
        self.rope_layer_split = config.pos_mode == 'rope_layer_split'
        # rope_layer_plan: a static tuple of indices into (seq_pair, tree_pair, None), one per
        # layer. Plain Python ints decided at init, so compile specialises each layer away.
        self.rope_plan = None
        if config.pos_mode == 'rope_layer_plan':
            self.rope_plan = tuple({'s': 0, 't': 1, 'n': 2}[c]
                                   for c in str(config.rope_layer_plan).strip().lower())
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # The depth table is created AFTER every random init and zero-filled, so it draws no
        # random numbers: the rest of the network is bit-identical to rope_seq at a seed (the
        # same guarantee GraphRoPE gives the rotary arms), and at step 0 the arm IS rope_seq.
        self.depth_emb = None
        if config.depth_emb:
            if config.pos_mode != 'rope_seq':
                raise ValueError("depth_emb is the input-embedding control for rope_depth and is "
                                 f"only defined on top of pos_mode='rope_seq', not {config.pos_mode!r}")
            self.depth_emb = nn.Embedding(config.rope_levels + 1, config.n_embd)
            torch.nn.init.zeros_(self.depth_emb.weight)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_mode == 'learned':
            # Only the sequence table is subtracted. The TREE tables are NOT:
            # they are the mechanism under test, not incidental position
            # bookkeeping, and hiding them would understate the treatment arm's
            # real parameter cost in every comparison.
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pos=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # ---- TOPOS: the entire experiment is this branch --------------------
        rot = None
        if self.pos_mode == 'learned':
            seq = torch.arange(0, t, dtype=torch.long, device=device)
            x = tok_emb + self.transformer.wpe(seq)
        elif self.pos_mode == 'tree':
            x = tok_emb + self.tree_pos(pos)
        else:  # 'none' and the rope arms: nothing is added to the embedding
            x = tok_emb
            if self.depth_emb is not None:
                path = pos.get('path') if pos is not None else None
                if path is None:
                    raise ValueError("depth_emb needs the DOM path (pos['path'] is None); the "
                                     "depth is the count of non-ABSENT levels in it")
                depth = (path[..., : self.config.rope_levels] >= 0).sum(dim=-1)
                x = x + self.depth_emb(depth)
            if self.rope is not None:
                # rope_seq ignores the path; every tree-bearing arm -- rope_graph,
                # rope_both, rope_leaf, rope_all and both split arms -- refuses without it
                path = pos.get('path') if pos is not None else None
                # 'slot' is present only under rope_ref_slot; every other arm gets None and
                # GraphRoPE.forward ignores it, so this is additive for the existing arms.
                slot = pos.get('slot') if pos is not None else None
                # 'ref' is present only under rope_ref_path -- the (row, col) of the cell each
                # token REFERS to. Same additive shape as 'slot': every other arm gets None.
                ref = pos.get('ref') if pos is not None else None
                rot = self.rope(t, path, device, slot, ref)
        x = self.transformer.drop(x)
        if self.rope_plan is not None and rot is not None:
            # rot is (seq_pair, tree_pair); either may be None if the plan never names it.
            # An 'n' layer gets rot=None and skips apply_rotary entirely.
            lut = (rot[0], rot[1], None)
            for i, block in enumerate(self.transformer.h):
                x = block(x, lut[self.rope_plan[i]])
        elif self.rope_layer_split and rot is not None:
            # rot is ((cos,sin) for SEQUENCE, (cos,sin) for TREE): even layers take the
            # first, odd layers the second.
            for i, block in enumerate(self.transformer.h):
                x = block(x, rot[i % 2])
        else:
            for block in self.transformer.h:
                x = block(x, rot)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.pos_mode == 'learned':
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # TOPOS: the rotation tables are (levels, planes) and so look like matmul weights to
        # the rule above, but they are FREQUENCIES, not weights: decaying them pulls the arm
        # toward no rotation, and only the arms that have them. See GPTConfig.rope_decay.
        rope_ids = set()
        if self.rope is not None and not self.config.rope_decay:
            rope_ids = {id(p) for p in self.rope.parameters()}
            print(f"rope_decay=False: {len(rope_ids)} rotation tensors excluded from weight decay")
        # The depth-embedding control gets the same treatment as the rotary depth step it is
        # compared with: decay would pull a zero-initialised table back toward "no depth", the
        # headwind rope_decay=False removed from the rotary arms.
        if self.depth_emb is not None and not self.config.rope_decay:
            rope_ids.add(id(self.depth_emb.weight))
            print("rope_decay=False: depth_emb table excluded from weight decay")
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in rope_ids]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or id(p) in rope_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
