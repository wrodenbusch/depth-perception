"""Record-indexed loader for packed program windows, with tree coordinates.

WHY THIS REPLACES nanoGPT's `get_batch` RATHER THAN FEEDING IT
---------------------------------------------------------------
nanoGPT's loader treats a dataset as **one flat token stream** and samples a
random offset into it every batch. That is exactly right for OpenWebText and
exactly wrong here, in two ways that would both be silent:

1. **It would straddle window boundaries.** A record of this pack IS a window of
   ~8.5 whole programs, and the train/val/test split is *per record*. Sampling a
   random offset into a concatenated stream would hand the model spans that begin
   in one window and end in the next — and, where the splits are adjacent, spans
   that begin in train and end in val. The pack's own `split_rule` says a window
   never mixes splits; a flat-stream sampler would reintroduce exactly that.
2. **It would destroy the coordinate alignment.** Tree coordinates are indexed by
   record: token *i of record r* has a causal node whose depth and sibling rank
   live at `node_off[r] + node_id`. An arbitrary offset into a flat stream has no
   record, so it has no coordinates.

So there is no `train.bin`. Converting to one would throw away the record identity
this experiment is built on. The pack is read where it lives.

WHAT A WINDOW IS, AND THE PROPERTY BEING TESTED
------------------------------------------------
From the sidecar manifest, quoted because it is the whole design:

    A RECORD OF THIS PACK IS A WINDOW HOLDING MANY PROGRAMS. Each program stays
    its own tree: the window is a FOREST of one root per program and tree
    coordinates restart at every program boundary, so nothing in the coordinate
    channel says which program a token is in. That is the thing being tested,
    not an oversight.

Consequently the coordinate channel carries no window-level position at all. A
model in the `tree` arm knows where a token sits in *its program's* tree and
nothing about where that program sits in the window — while the `learned` arm
knows the absolute offset and nothing about the tree.

NO LOOKAHEAD
------------
`tok_causal` is the node an incremental parser knows at the token. The manifest
records `0 of 46,262,569 tokens lookahead-dependent` inside a program, and across
programs the causal position is the EOT count — a prefix fact. So the `tree` arm
reads no positional information out of the future. On a render without that
property this loader would be feeding a leak.
"""
from topos_paths import PACKS  # noqa: E402

import json
import os

import numpy as np
import torch

#: Default location of the pack and its tree sidecar, in the project-neutral
#: registry. Override with TOPOS_PACK.
DEFAULT_PACK = os.environ.get(
    "TOPOS_PACK",
    f"{PACKS}/packedwindows_wholefile_apps_kodcode_423k_astcompact_ident32k_b1024",
)


SIDECARS = (None, "tree", "path")


class PackedWindows:
    """Reads one pack + one sidecar, record by record.

    ``sidecar``:
      ``'tree'``  the ``_treepos_v3`` node arrays (the Python experiment's ``tree`` arm)
      ``'path'``  the ``_dompath`` per-token DOM rank path (the web cell's rope arms)
      ``None``    tokens only (``learned``, ``none``, ``rope_seq``)
    ``need_tree=True`` with no ``sidecar`` means ``'tree'``, for the older callers.

    Two optional extras ride along with the ``'path'`` sidecar, each requested explicitly and
    each REFUSED if absent rather than silently skipped -- an arm that trains without its
    pointer coordinate degrades to ``rope_both`` on a loss curve that looks entirely normal:
      ``need_slot``  ``tok_slot.npy``, the symbol slot (``rope_ref_slot``)
      ``need_ref``   the grid pointer, served as ONE array: ``(B, t, 3)`` of
                     ``(sheet, row, col)`` when the pack has ``tok_ref_sheet.npy``, else
                     ``(B, t, 2)`` of ``(row, col)`` for a pack built before 2026-09-28
      ``need_ref_sheet``  additionally REFUSE a pack with no sheet axis. ``rope_ref_anchor``
                     needs it: without the sheet, 17.25% of references (the cross-sheet
                     ones) coincide with the wrong cell on the reference's own sheet
    """

    def __init__(self, pack_dir: str = DEFAULT_PACK, need_tree: bool = True,
                 sidecar: str | None = None, need_slot: bool = False,
                 need_ref: bool = False, need_ref_sheet: bool = False):
        self.pack_dir = pack_dir
        self.side_dir = pack_dir + "_treepos_v3"
        with open(os.path.join(pack_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.block_size = int(self.meta["block_size"])
        self.vocab_size = int(self.meta["vocab_size"])

        self.tokens = np.load(os.path.join(pack_dir, "tokens.npy"), mmap_mode="r")
        self.tok_off = np.load(os.path.join(pack_dir, "tok_off.npy"), mmap_mode="r")
        self.n_records = len(self.tok_off) - 1

        if sidecar not in SIDECARS:
            raise ValueError(f"sidecar {sidecar!r} is not one of {SIDECARS}")
        self.sidecar = sidecar if sidecar is not None else ("tree" if need_tree else None)
        self.need_tree = self.sidecar == "tree"
        if self.sidecar == "path":
            s = pack_dir + "_dompath"
            if not os.path.isdir(s):
                raise FileNotFoundError(
                    f"rope_graph / rope_both need the DOM-path sidecar at {s}, which is "
                    "absent. Refusing rather than training an arm with no coordinates."
                )
            self.tok_node = np.load(os.path.join(s, "tok_node.npy"), mmap_mode="r")
            self.node_path = np.load(os.path.join(s, "node_path.npy"), mmap_mode="r")
            self.node_off = np.load(os.path.join(s, "node_off.npy"), mmap_mode="r")
            self.levels = int(self.node_path.shape[1])
            # Sidecar v2 carries the within-node token index; it becomes one extra
            # trailing column of the path (the token-leaf level, read by rope_leaf).
            self.has_within = os.path.exists(os.path.join(s, "tok_within.npy"))
            self.tok_within = np.load(os.path.join(s, "tok_within.npy"), mmap_mode="r") if self.has_within else None
            self.path_cols = self.levels + (1 if self.has_within else 0)
            # SYMBOL SLOT (2026-09-28), for rope_ref_slot. A SEPARATE key from `path`, not a
            # trailing column: adding a column would change path_cols and every existing arm
            # checks that width, so an additive column would break rope_leaf and friends.
            self.has_slot = os.path.exists(os.path.join(s, "tok_slot.npy"))
            self.tok_slot = np.load(os.path.join(s, "tok_slot.npy"), mmap_mode="r") if self.has_slot else None
            if len(self.tok_node) != len(self.tokens):
                raise ValueError(
                    f"sidecar has {len(self.tok_node)} token rows but the pack has "
                    f"{len(self.tokens)}; these are not the same pack"
                )
            if need_slot and not self.has_slot:
                raise FileNotFoundError(
                    f"rope_ref_slot needs tok_slot.npy in {s}, which is absent. Rebuild the "
                    "sidecar with `convert_treepos.py --slots`. Refusing rather than training "
                    "the reference arm with no reference coordinate -- it would silently become "
                    "rope_both and the loss curve would look entirely normal."
                )
            if self.has_slot and len(self.tok_slot) != len(self.tokens):
                raise ValueError(
                    f"tok_slot has {len(self.tok_slot)} rows but the pack has "
                    f"{len(self.tokens)}; these are not the same pack"
                )
            # GRID POINTER (R7, 2026-09-28), for the reference arms. Like the slot, a SEPARATE
            # key and not a trailing path column: every existing arm checks path_cols, so
            # widening the path would break rope_leaf and friends.
            #
            # TWO WIDTHS, because the SHEET axis was added later (2026-09-28) and a v1 pack does
            # not have it. A pack with tok_ref_sheet.npy is served as (B, t, 3) of
            # (sheet, row, col); one without, as (B, t, 2) of (row, col). rope_ref_path reads
            # either -- it only wants the last two -- while rope_ref_anchor REQUIRES three,
            # because a pointer with no sheet aliases onto the wrong cell for 17.25% of
            # references, which is one of the two faults it exists to fix.
            self.has_ref = all(
                os.path.exists(os.path.join(s, f)) for f in ("tok_ref_row.npy", "tok_ref_col.npy")
            )
            self.has_ref_sheet = self.has_ref and os.path.exists(
                os.path.join(s, "tok_ref_sheet.npy")
            )
            self.tok_ref_row = (
                np.load(os.path.join(s, "tok_ref_row.npy"), mmap_mode="r") if self.has_ref else None
            )
            self.tok_ref_col = (
                np.load(os.path.join(s, "tok_ref_col.npy"), mmap_mode="r") if self.has_ref else None
            )
            self.tok_ref_sheet = (
                np.load(os.path.join(s, "tok_ref_sheet.npy"), mmap_mode="r")
                if self.has_ref_sheet else None
            )
            if need_ref and not self.has_ref:
                raise FileNotFoundError(
                    f"a reference arm needs tok_ref_row.npy and tok_ref_col.npy in {s}, which "
                    "are absent. Only build_sheet_pack.py emits them. Refusing rather than "
                    "training the pointer arm with no pointer -- it would silently become "
                    "rope_both on a loss curve that looks entirely normal."
                )
            if need_ref_sheet and not self.has_ref_sheet:
                raise FileNotFoundError(
                    f"rope_ref_anchor needs tok_ref_sheet.npy in {s}, which is absent -- this "
                    "looks like a pack built before 2026-09-28. Without the sheet axis a "
                    "cross-sheet reference coincides with whatever cell holds its (row, col) on "
                    "the reference's OWN sheet, which is 17.25% of references pointing "
                    "confidently at the wrong cell. Refusing rather than measuring that. "
                    "Rebuild with build_sheet_pack.py."
                )
            if self.has_ref:
                arrays = [("tok_ref_row", self.tok_ref_row), ("tok_ref_col", self.tok_ref_col)]
                if self.has_ref_sheet:
                    arrays.append(("tok_ref_sheet", self.tok_ref_sheet))
                for nm, arr in arrays:
                    if len(arr) != len(self.tokens):
                        raise ValueError(
                            f"{nm} has {len(arr)} rows but the pack has {len(self.tokens)}; "
                            "these are not the same pack"
                        )
        self.need_slot = need_slot
        self.need_ref = need_ref
        self.need_ref_sheet = need_ref_sheet
        if self.need_tree:
            if not os.path.isdir(self.side_dir):
                raise FileNotFoundError(
                    f"pos_mode='tree' needs the tree sidecar at {self.side_dir}, which is "
                    "absent. Refusing rather than training an arm with no coordinates."
                )
            s = self.side_dir
            self.tok_causal = np.load(os.path.join(s, "tok_causal.npy"), mmap_mode="r")
            self.node_parent = np.load(os.path.join(s, "node_parent.npy"), mmap_mode="r")
            self.node_depth = np.load(os.path.join(s, "node_depth.npy"), mmap_mode="r")
            self.node_sib = np.load(os.path.join(s, "node_sib.npy"), mmap_mode="r")
            self.node_off = np.load(os.path.join(s, "node_off.npy"), mmap_mode="r")
            if len(self.tok_causal) != len(self.tokens):
                raise ValueError(
                    f"sidecar has {len(self.tok_causal)} token rows but the pack has "
                    f"{len(self.tokens)}; these are not the same pack"
                )

        # Splits are ranges of RECORD ordinals: the pack places each source
        # split's windows at contiguous output ordinals, and a window never
        # mixes splits (checked at build time by the builder, not argued here).
        n_tr, n_va = int(self.meta["n_train"]), int(self.meta["n_val"])
        self.splits = {
            "train": (0, n_tr),
            "val": (n_tr, n_tr + n_va),
            "test": (n_tr + n_va, self.n_records),
        }

    def __repr__(self) -> str:
        return (
            f"PackedWindows({os.path.basename(self.pack_dir)}, {self.n_records} windows, "
            f"block {self.block_size}, vocab {self.vocab_size}, sidecar={self.sidecar})"
        )

    def record(self, r: int):
        """One record's tokens and, if a sidecar is loaded, its coordinates."""
        a, b = int(self.tok_off[r]), int(self.tok_off[r + 1])
        toks = np.asarray(self.tokens[a:b], dtype=np.int64)
        if self.sidecar == "path":
            # Per-token DOM rank path: tok_node is record-local, -1 outside the
            # tree (framing); a token outside the tree gets an all-ABSENT row.
            tn = np.asarray(self.tok_node[a:b], dtype=np.int64)
            na, nb = int(self.node_off[r]), int(self.node_off[r + 1])
            npth = np.asarray(self.node_path[na:nb], dtype=np.int64)
            path = np.full((b - a, self.path_cols), -1, dtype=np.int64)
            inside = tn >= 0
            path[inside, : self.levels] = npth[tn[inside]]
            if self.has_within:
                path[:, self.levels] = np.asarray(self.tok_within[a:b], dtype=np.int64)
            out = dict(path=path)
            if self.need_slot:
                out["slot"] = np.asarray(self.tok_slot[a:b], dtype=np.int64)
            if self.need_ref:
                axes = []
                if self.has_ref_sheet:      # (sheet, row, col); sheet first, as in the path
                    axes.append(np.asarray(self.tok_ref_sheet[a:b], dtype=np.int64))
                axes.append(np.asarray(self.tok_ref_row[a:b], dtype=np.int64))
                axes.append(np.asarray(self.tok_ref_col[a:b], dtype=np.int64))
                out["ref"] = np.stack(axes, axis=-1)
            return toks, out
        if not self.need_tree:
            return toks, None
        na, nb = int(self.node_off[r]), int(self.node_off[r + 1])
        # Node ids are ALREADY RECORD-LOCAL and must NOT be rebased.
        #
        # The sidecar manifest says node ids are "shifted by the window's
        # running node base", which describes how a window is BUILT from its
        # member programs -- not how the arrays are stored. Stored, they are
        # local: record 0 holds 793 nodes and its ids run 0..792.
        #
        # Subtracting node_off here (which I did, first pass) drives every id
        # negative, every token reads as outside the tree, the tree arm adds
        # exactly zero everywhere, and it silently becomes the `none` arm. The
        # loss still looks correct at init -- both arms sit at ln(vocab) -- so
        # nothing fails. The check that catches it is `causal.max() ==
        # n_nodes - 1`, asserted in test_topos_data.py.
        causal = np.asarray(self.tok_causal[a:b], dtype=np.int64)
        parent = np.asarray(self.node_parent[na:nb], dtype=np.int64)
        return toks, dict(
            causal=causal,
            parent=parent,
            depth=np.asarray(self.node_depth[na:nb], dtype=np.int64),
            sib=np.asarray(self.node_sib[na:nb], dtype=np.int64),
        )

    def get_batch(self, split: str, batch_size: int, device, generator=None):
        """``(x, y, pos)`` — whole records, never a random offset into a stream.

        Targets are the record shifted by one, so the final token of a window has
        no target and is dropped. Every arm sees the identical windows in the
        identical order for a given seed, which is what makes arm-versus-arm
        comparison meaningful.
        """
        lo, hi = self.splits[split]
        ix = torch.randint(lo, hi, (batch_size,), generator=generator).tolist()
        return self.get_batch_records(ix, device)

    #: Micro-batch shapes the length bucketer is allowed to emit. Restricting the
    #: batch dimension and rounding the time dimension to a multiple of ``PAD_TO``
    #: keeps the number of distinct shapes small, so ``torch.compile`` does not
    #: recompile on every step.
    #: Candidate group sizes, tried largest first. Raised from a cap of 16 on 2026-09-30:
    #: measured on the Python pack, a cap of 16 left the most common micro-batch at
    #: 16 x 128 = 2,048 positions against a token_budget of 8,192, so the CAP and not the
    #: budget was binding and the GPU ran 16.5 tiny micro-batches per step. A cap of 64 takes
    #: that to 6.7 -- 2.5x fewer kernel launches, each 2.5x larger -- with padding efficiency
    #: unchanged at 58.8%, because widening a group does not widen its length range once the
    #: step is already length-sorted.
    #:
    #: SAFE because ``token_budget`` still bounds ``len(group) * padded_len``, so peak
    #: activation memory is unchanged: a bigger ``b`` is only ever chosen when ``t`` is small.
    #: And grouping is gradient-neutral by design (see ``step_batches``) -- verified
    #: empirically rather than trusted: same drawn records under caps 16 and 64 give gradients
    #: agreeing to **1.9e-07 relative**, i.e. float summation order, so runs before and after
    #: this change remain comparable and the existing 20-seed baselines stand.
    #:
    #: Do NOT also raise ``token_budget`` to chase fewer micro-batches: measured, a cap of 128
    #: with a 32k budget reaches 3.0 micro-batches per step but padding efficiency FALLS to
    #: 46.4%, so total work rises. Padding is a separate problem, owned by ``PAD_TO``.
    BUCKET_SIZES = (64, 48, 32, 16, 8, 4, 2, 1)
    PAD_TO = 128

    def step_batches(self, split: str, n_records: int, token_budget: int, generator=None):
        """Record ordinals for one optimizer step, grouped by length.

        WHY. A record is a whole web page and page lengths vary enormously, so a
        micro-batch padded to its longest member wastes a lot of compute: measured
        on this pack, a random micro-batch of 2 is only **72% useful** and one of 4
        is 55%. Sorting the step's records by length before grouping takes that to
        98% and 94%. Attention is quadratic in the padded length, so the saving is
        larger than the token ratio alone suggests.

        The step draws exactly the same ``n_records`` ordinals it would have drawn
        before, from the same generator -- only the grouping inside the step
        changes. With the token-weighted accumulation in ``train.py`` the gradient
        is therefore what an unpadded run would produce, and arms at one seed still
        see identical records in identical order.

        ``token_budget`` caps ``len(group) * padded_length``, so peak activation
        memory is bounded by the same worst case as a fixed micro-batch of
        ``token_budget / block_size``.

        Returns ``(groups, total_target_tokens)``.
        """
        lo, hi = self.splits[split]
        ix = torch.randint(lo, hi, (n_records,), generator=generator).tolist()
        lens = [int(self.tok_off[r + 1]) - int(self.tok_off[r]) - 1 for r in ix]  # targets per record
        order = sorted(range(len(ix)), key=lambda i: lens[i])
        groups, i = [], 0
        while i < len(order):
            best = None
            for b in self.BUCKET_SIZES:
                if b > len(order) - i:
                    continue
                t = max(lens[order[j]] for j in range(i, i + b))
                t = min(self.block_size, ((t + self.PAD_TO - 1) // self.PAD_TO) * self.PAD_TO)
                if b * t <= token_budget:
                    best = b
                    break
            if best is None:
                best = 1                      # one long record always goes alone
            groups.append([ix[order[j]] for j in range(i, i + best)])
            i += best
        return groups, sum(lens)

    def get_batch_records(self, ix, device):
        """``(x, y, pos)`` for an explicit list of record ordinals (fixed evaluation sets)."""
        # WINDOWS ARE VARIABLE LENGTH -- only ~0.9% are exactly block_size
        # (median 967, min 149 over the first 20k records). So a batch must be
        # padded, and the padding must not be learned:
        #   x is padded with EOT, which the model has seen as framing;
        #   y is padded with -1, which nanoGPT's cross_entropy takes as
        #     ignore_index (model.py: `ignore_index=-1`).
        # Padding y with 0 instead would silently train the model to predict EOT
        # on ~6% of all positions and would never show up as an error.
        eot = int(self.meta.get("eot", 0))
        recs = [self.record(r) for r in ix]
        t_max = max(len(toks) for toks, _ in recs) - 1
        b = len(recs)

        x = torch.full((b, t_max), eot, dtype=torch.long)
        y = torch.full((b, t_max), -1, dtype=torch.long)
        per = []
        for i, (toks, pos) in enumerate(recs):
            t = len(toks) - 1
            x[i, :t] = torch.from_numpy(toks[:t])
            y[i, :t] = torch.from_numpy(toks[1 : t + 1])
            if pos is not None:
                per.append((pos, t))

        pos_batch = None
        if per and self.sidecar == "path":
            path = torch.full((b, t_max, self.path_cols), -1, dtype=torch.long)
            for i, (p, t) in enumerate(per):
                path[i, :t] = torch.from_numpy(p["path"][:t])
            pos_batch = dict(path=path)
            if self.need_slot:
                # Padding is ABSENT, exactly as for the path: a pad position carries no
                # symbol, so its slot planes must contribute zero rather than slot 0.
                slot = torch.full((b, t_max), -1, dtype=torch.long)
                for i, (p, t) in enumerate(per):
                    slot[i, :t] = torch.from_numpy(p["slot"][:t])
                pos_batch["slot"] = slot
            if self.need_ref:
                # Padding is ABSENT on both axes, exactly as for the path. A pad position has
                # no reference AND no cell, so its pointer planes must contribute zero rather
                # than pointing at row 0, column A.
                n_axes = 3 if self.has_ref_sheet else 2
                refb = torch.full((b, t_max, n_axes), -1, dtype=torch.long)
                for i, (p, t) in enumerate(per):
                    refb[i, :t] = torch.from_numpy(p["ref"][:t])
                pos_batch["ref"] = refb
        elif per:
            n_max = max(len(p["parent"]) for p, _ in per)
            causal = torch.full((b, t_max), -1, dtype=torch.long)
            parent = torch.full((b, n_max), -1, dtype=torch.long)
            depth = torch.zeros((b, n_max), dtype=torch.long)
            sib = torch.zeros((b, n_max), dtype=torch.long)
            for i, (p, t) in enumerate(per):  # noqa: B020 - p reused deliberately
                causal[i, :t] = torch.from_numpy(p["causal"][:t])
                n = len(p["parent"])
                parent[i, :n] = torch.from_numpy(p["parent"])
                depth[i, :n] = torch.from_numpy(p["depth"])
                sib[i, :n] = torch.from_numpy(p["sib"])
            pos_batch = dict(causal=causal, parent=parent, depth=depth, sib=sib)

        if device.type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            if pos_batch is not None:
                pos_batch = {k: v.pin_memory().to(device, non_blocking=True) for k, v in pos_batch.items()}
        else:
            x, y = x.to(device), y.to(device)
            if pos_batch is not None:
                pos_batch = {k: v.to(device) for k, v in pos_batch.items()}
        return x, y, pos_batch
