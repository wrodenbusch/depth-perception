"""Additive tree coordinates for a token-level decoder.

WHAT THIS IS
------------
nanoGPT gives every token a *sequence* position: one learned row per index into
``block_size``, added to the token embedding. This module is the alternative the
project exists to test — one learned row per *tree* coordinate, added at exactly
the same point. Nothing else about the model changes.

Three tables, summed:

  ordinal   the DFS ordinal of the token's own AST node, and — from the same
            table — the ordinal of that node's parent
  depth     how deep the node sits
  sibling   its rank among its parent's children

THE ARM THAT CARRIES THE CLAIM
------------------------------
``ordinal`` is a NODE ordinal, not a sequence position. That is the whole point:
the positional parameter budget is then a function of the *tree* (``MAX_NODES``
rows) and not of ``block_size``. A model whose positions are tree-shaped has no
context-length-sized position table to outgrow, which is the property that makes
this worth measuring rather than just worth having.

Tokens belonging to one node share an ordinal, so ordering *within* a node comes
only from the causal mask. That is deliberate and it is not nothing — decoder-only
models are known to carry usable positional information in the mask alone
(Haviv et al. 2022; Kazemnejad et al. 2023), which is also why ``pos_mode='none'``
is a real floor rather than a broken control.

THREE DECISIONS THAT ARE EASY TO GET WRONG
------------------------------------------
1. **Causal coordinates only.** ``forward`` reads the node an incremental parser
   would know *at* the token, never the node the finished tree assigns. On a
   canonical prefix-AST render the two coincide (the sidecar records
   ``tokens_lookahead_dependent: 0``), so here the choice is free; on any other
   render it is the difference between an honest arm and one that reads a
   quarter of its positions out of the future.

2. **Outside the tree means ZERO, not row 0.** Framing tokens get exactly zero
   from all three terms. Row 0 is a real depth and a real sibling rank; handing
   it to a token that has neither would teach the model that framing looks like
   a shallow first child.

3. **The ordinal table is shared** between a token's own node and its parent —
   one table queried twice, not two tables. A parent ordinal and an own ordinal
   live in the same space and mean the same thing.

Initialisation is nanoGPT's embedding init (normal, std 0.02) drawn under a
private seed, so that the transformer stack's own initialisation is bit-identical
across positional arms at a given seed. Without that, an arm comparison is partly
a comparison of two different random models.
"""

import torch
import torch.nn as nn

#: Rows in the node-ordinal table. Sized by the TREE, not by ``block_size`` —
#: this is the length-independence property under test.
MAX_NODES = 2048

#: Depth rows. One more than the deepest tree the corpus builder admits, so the
#: clamp below is a guard rather than a routine truncation.
MAX_DEPTH = 49

#: Sibling-rank rows.
MAX_SIB = 64

#: Private init seed, so the block stack is identical across arms at one seed.
_INIT_SEED = 1515


class TreeAddEmbedding(nn.Module):
    """``(B, T, n_embd)`` to add to the token embedding, in place of ``wpe``."""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.n_embd = int(n_embd)
        state = torch.random.get_rng_state()
        torch.manual_seed(_INIT_SEED)
        # The SHARED ordinal table: the token's own node and its parent.
        self.ord_emb = nn.Embedding(MAX_NODES, self.n_embd)
        self.depth_emb = nn.Embedding(MAX_DEPTH, self.n_embd)
        self.sib_emb = nn.Embedding(MAX_SIB, self.n_embd)
        for e in (self.ord_emb, self.depth_emb, self.sib_emb):
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        torch.random.set_rng_state(state)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, pos: dict) -> torch.Tensor:
        """``pos`` carries, for one batch of windows:

        ``causal``  ``(B, T)`` int64 — the node each token belongs to as an
                    incremental parser would know it; ``-1`` outside the tree.
        ``parent``  ``(B, N)`` int64 — parent ordinal per node, ``-1`` at a root.
        ``depth``   ``(B, N)`` int64 — depth per node.
        ``sib``     ``(B, N)`` int64 — sibling rank per node.
        """
        for key in ("causal", "parent", "depth", "sib"):
            if pos.get(key) is None:
                have = sorted(k for k, v in pos.items() if v is not None)
                raise ValueError(f"tree positions need pos[{key!r}]; this batch carries {have}")

        ids = pos["causal"]                       # (B, T)
        parent = pos["parent"]                    # (B, N)
        inside = (ids >= 0).unsqueeze(-1)         # (B, T, 1)
        safe = ids.clamp(min=0)                   # gather-safe copy

        # -- own ordinal, and the parent's, from the same table ----------------
        par_of_tok = torch.gather(parent, 1, safe)            # (B, T), -1 at a root
        has_parent = (par_of_tok >= 0) & (ids >= 0)
        own = safe.clamp(max=MAX_NODES - 1)
        par = par_of_tok.clamp(min=0, max=MAX_NODES - 1)

        x = self.ord_emb(own) * inside.to(self.ord_emb.weight.dtype)
        x = x + self.ord_emb(par) * has_parent.unsqueeze(-1).to(x.dtype)

        # -- depth and sibling rank, clamped, zeroed outside the tree ----------
        depth = torch.gather(pos["depth"], 1, safe).clamp(0, MAX_DEPTH - 1)
        sib = torch.gather(pos["sib"].to(torch.long), 1, safe).clamp(0, MAX_SIB - 1)
        m = inside.to(x.dtype)
        return x + self.depth_emb(depth) * m + self.sib_emb(sib) * m
