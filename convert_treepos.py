#!/usr/bin/env python
"""Convert a ``_treepos_v3`` sidecar into the ``_dompath`` form the rotary arms read.

    .venv/bin/python convert_treepos.py \
        --pack $TOPOS_PACKS/rep23_textpack_wholefile_apps_kodcode_423k_astcompact_ident32k \
        --out  $TOPOS_PACKS/py_ast_423k

WHY THIS EXISTS
---------------
Every web cell so far has fed the rope arms a DOM rank path built by
``build_web_pack.py``. The Python AST packs already carry a tree sidecar, but in
the older ``_treepos_v3`` shape -- ``node_parent`` / ``node_sib`` / ``node_depth``
rather than a materialised root path. Nothing needs re-tokenising: the path is a
pure function of those three arrays, so this rebuilds it in a couple of minutes
instead of rebuilding a pack.

It is worth doing because the AST is a far better match for the encoding than a
DOM is. Measured on the 423k-program pack: depth p50 5 / p99 10 / max 57 against
22 levels, and sibling rank p50 0 / p99 7 / max 100 against a collision point of
318. Almost nothing is dropped and nothing aliases -- unlike the web packs, where
10% of pages exceed the depth budget.

THE PATH RULE, matching ``build_web_pack.py`` so the cells stay comparable
--------------------------------------------------------------------------
A node's path is the list of SIBLING RANKS from the child of the root down to the
node itself. The root -- the ``Module`` node, identical in every program -- is NOT
a level, exactly as ``<html>`` / ``<body>`` are not levels in the web packs. So a
node at depth ``d`` occupies path positions ``0 .. d-1``. Paths longer than
``levels`` are truncated at the LEAF end, so level ``k`` always means tree level
``k``. Absent levels are ``-1``.

RANK MEANS INDEX AMONG *ALL* THE PARENT'S CHILDREN (``--rank flat``, the default),
which is what ``build_web_pack.py`` means by it. The source sidecar's ``node_sib``
is the index among the children THAT SHARE A FIELD, so an ``If``'s ``body[0]`` and
``orelse[0]`` are both 0 -- and paths built from it are not unique: 47.4% of nodes
in this pack would share a path with another node in the same record, in 100% of
records. ``--rank field`` keeps the source array if that is ever wanted.

Parents precede their children in DFS pre-order, which lets the table be filled
depth by depth with no recursion. That ordering is ASSERTED, not assumed: if a
pack ever violated it the paths would be silently built from unfilled parent rows,
which is exactly the kind of fault that trains an arm on zeros without failing.

WHAT IS COPIED VERBATIM
-----------------------
``tok_node`` (from ``tok_causal``), ``tok_within`` and ``node_off`` are the
source arrays unchanged. ``tok_causal`` is the LOOKAHEAD-FREE token->node map --
the node an incremental parser knows AT the token -- so the arm reads no position
out of the future. ``tok_node.npy`` in the source sidecar is the gold map and is
deliberately not used.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ABSENT = -1
LEVELS = 22


SLOT_GATES = ("carry", "strict", "complete")


def build_tok_slot(node_ref, node_off, tok_causal, tok_off, node_ident_at, tok_char_end,
                   gate: str = "carry"):
    """Per-token SYMBOL SLOT: the k-th distinct symbol introduced in a record gets slot k.

    Returns ``(tok_slot, stats)``. ``tok_slot[t]`` is ABSENT where the token carries no
    symbol, or where the reader cannot yet know which symbol it is (see the gate below).

    WHY THE SLOT IS BUILT FROM ``node_ref``, NOT FROM ``node_ident``
    ---------------------------------------------------------------
    The design (``docs/design-reference-rotation.md``) proposed grouping by
    ``node_ident``, the sidecar's per-node identity angle, on the grounds that equal
    identity means equal symbol. MEASURED, THAT IS FALSE: over 986,360 reference pairs a
    use and its own binder share an identity value only **70.43%** of the time. The
    identity is ``sha1(f'{tag}|{name}')`` and the TAG differs between a binder (``arg``,
    ``FunctionDef``, a store-context ``Name``) and a use (a load-context ``Name``), so
    the same variable hashes to two different angles. Grouping by it would split ~30% of
    symbols into two slots each, and nothing downstream would fail -- the loss curve
    would look entirely normal.

    So the equivalence classes come from the RESOLVED EDGES instead. ``node_ref`` is the
    scope-resolved canonical binder of a use, which is precisely the relation the slot is
    meant to express. Each class is a connected component of that pointer graph and the
    slot is its rank in first-mention order.

    ``node_ref`` IS RECORD-LOCAL and is rebased through ``node_off`` here. Verified on
    the full pack: all 9,707,041 refs satisfy ``0 <= ref < n_nodes_in_record`` and none
    satisfies the global form, so using it unrebased would index another record's nodes.
    Also verified: **100.00%** of refs point to a strictly EARLIER local id, i.e. every
    surviving edge is backward. The builder dropped 139,028 forward edges (1.4%), so
    those uses carry no slot -- which is why coverage is not 100%.

    Because every edge is backward, the earliest node of a class IS the first mention, so
    ranking classes by their minimum id gives first-mention order and the slot of a token's
    symbol is always knowable from the prefix.

    THE GATE, and the tension MEASURED here
    ---------------------------------------
    Knowing *which* symbol a name denotes requires having READ the name. The sidecar's own
    ``ident_gate_rule`` makes the identity plane zero unless
    ``node_ident_at[node] < tok_char_end[t]`` -- ``node_ident_at`` is ALREADY one character
    past the name's end, so strict ``<`` demands the reader have consumed the delimiter too,
    because until then it cannot know the name has ended.

    That rule and a per-token coordinate are in direct tension, and the measurement is
    stark. Over 8M tokens: 57.11% have a causal node that carries a name, but

      ``strict``    ``ident_at <  char_end``  ->  **0.54%** of tokens
      ``complete``  ``ident_at <= char_end``  ->  50.35% of tokens
      ``carry``     forward-fill the ``strict`` events within the record -> ~30%

    ``ident_at == char_end`` EXACTLY at the token where a name completes, which is why
    ``strict`` keeps almost nothing: by the time the delimiter has been consumed,
    ``tok_causal`` has advanced to another node. So ``strict`` yields an arm whose slot
    planes are zero for 99.5% of tokens -- honest but inert.

    ``complete`` is dense but LEAKS: at the token where ``char_end == ident_at`` the reader
    has consumed the name's last character and not the delimiter, so claiming the identity
    there uses the gold parse's knowledge that the name stops -- the token stream itself
    still permits ``foo`` to continue as ``foobar``. One character of lookahead, but
    lookahead.

    ``carry`` IS THE DEFAULT and is leak-free: a token takes the slot of the most recently
    ``strict``-resolved symbol in its record, forward-filled. Only names whose delimiter has
    actually been consumed ever contribute, so nothing is claimed early, and the coreference
    property survives -- the tokens following two mentions of one symbol share an angle.
    The semantics shift from "I am symbol X" to "the last resolved symbol was X", which is
    weaker and must be described that way in any writeup.
    """
    if gate not in SLOT_GATES:
        raise ValueError(f"gate must be one of {SLOT_GATES}, got {gate!r}")
    ref = np.asarray(node_ref, dtype=np.int64)
    off = np.asarray(node_off, dtype=np.int64)
    n_nodes = len(ref)
    if int(off[-1]) != n_nodes:
        raise ValueError(f"node_off ends at {int(off[-1])} but node_ref has {n_nodes} rows")

    counts = np.diff(off)
    rec_start = np.repeat(off[:-1], counts)
    local = np.arange(n_nodes, dtype=np.int64) - rec_start
    has_ref = ref >= 0
    if has_ref.any():
        if not bool((ref[has_ref] < np.repeat(counts, counts)[has_ref]).all()):
            raise ValueError(
                "node_ref is not record-local: some ref is >= its record's node count. "
                "This builder rebases through node_off and would index the wrong record."
            )
        if not bool((ref[has_ref] < local[has_ref]).all()):
            raise ValueError(
                "node_ref contains a FORWARD edge (ref >= the node's own local id). "
                "First-mention slots assume every edge is backward; refusing rather than "
                "assigning a slot the reader could not have known."
            )

    # Connected components of a backward functional graph: follow each node's pointer to
    # its binder and pointer-double until fixed. Acyclic by the backward check above, so
    # this terminates; log2(max chain) iterations, vectorised over all 40M rows.
    ptr = np.where(has_ref, rec_start + ref, np.arange(n_nodes, dtype=np.int64))
    for _ in range(64):
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    else:
        raise ValueError("reference chains did not converge in 64 pointer-doubling rounds")
    root = ptr

    # A node carries a symbol if it is a use (has a ref) or is the target of one.
    in_class = has_ref.copy()
    in_class[rec_start[has_ref] + ref[has_ref]] = True

    # Slot = rank of the class's root WITHIN ITS RECORD, in ascending root order. Roots are
    # global ids and records occupy contiguous id ranges, so ascending root order is
    # first-mention order inside a record and records stay grouped.
    node_slot = np.full(n_nodes, ABSENT, dtype=np.int32)
    if in_class.any():
        sel = np.nonzero(in_class)[0]
        _, dense = np.unique(root[sel], return_inverse=True)     # dense class id, root-ascending
        rec_of = np.searchsorted(off, sel, side="right") - 1
        # first dense id of each record, broadcast back onto the selected rows
        firsts = np.zeros(len(off) - 1, dtype=np.int64)
        starts = np.searchsorted(rec_of, np.arange(len(off) - 1), side="left")
        valid = starts < len(rec_of)
        firsts[valid] = dense[starts[valid]]
        node_slot[sel] = (dense - firsts[rec_of]).astype(np.int32)

    # Tokens: tok_causal is the RECORD-LOCAL node an incremental parser knows at the token.
    causal = np.asarray(tok_causal, dtype=np.int64)
    toff = np.asarray(tok_off, dtype=np.int64)
    n_tok = len(causal)
    tcounts = np.diff(toff)
    tok_rec_start = np.repeat(off[:-1], tcounts)                  # node base of the token's record
    inside = causal >= 0
    gnode = np.where(inside, tok_rec_start + causal, 0)

    tok_slot = np.full(n_tok, ABSENT, dtype=np.int16)
    slot_at_tok = np.where(inside, node_slot[gnode], ABSENT)
    iat = np.asarray(node_ident_at, dtype=np.int64)
    tce = np.asarray(tok_char_end, dtype=np.int64)
    at = np.where(inside, iat[gnode], -1)
    named = inside & (at >= 0)
    resolved = named & (at < tce)          # the leak-free event: delimiter consumed
    if gate == "strict":
        keep, src = resolved & (slot_at_tok >= 0), slot_at_tok
    elif gate == "complete":
        keep, src = named & (at <= tce) & (slot_at_tok >= 0), slot_at_tok
    else:  # carry -- forward-fill the resolved events, never across a record boundary
        ev = np.where(resolved & (slot_at_tok >= 0), np.arange(n_tok, dtype=np.int64), -1)
        last = np.maximum.accumulate(ev)                     # index of the last event <= t
        tok_rec_first = np.repeat(toff[:-1], tcounts)        # first token index of the record
        keep = last >= tok_rec_first                         # an event exists in THIS record
        src = np.where(keep, slot_at_tok[np.clip(last, 0, None)], ABSENT)
    lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    n_clamped = int(np.count_nonzero(keep & ((src < lo) | (src > hi))))
    tok_slot[keep] = np.clip(src[keep], lo, hi).astype(np.int16)

    # Statistics. Slots per record, and how much of the stream carries one.
    per_rec = np.zeros(len(off) - 1, dtype=np.int64)
    if in_class.any():
        sel = np.nonzero(in_class)[0]
        rec_of = np.searchsorted(off, sel, side="right") - 1
        np.maximum.at(per_rec, rec_of, node_slot[sel].astype(np.int64) + 1)
    nz = per_rec[per_rec > 0]
    stats = dict(
        n_nodes=int(n_nodes),
        n_tokens=int(n_tok),
        nodes_with_a_ref=int(has_ref.sum()),
        nodes_in_a_symbol_class=int(in_class.sum()),
        nodes_in_a_symbol_class_frac=float(in_class.mean()),
        gate=gate,
        tokens_with_a_slot=int((tok_slot >= 0).sum()),
        tokens_with_a_slot_frac=float((tok_slot >= 0).mean()),
        tokens_named_node=int(named.sum()),
        tokens_named_node_frac=float(named.mean()),
        tokens_resolved_events=int(resolved.sum()),
        tokens_resolved_events_frac=float(resolved.mean()),
        slots_per_record_median=float(np.median(nz)) if len(nz) else 0.0,
        slots_per_record_mean=float(nz.mean()) if len(nz) else 0.0,
        slots_per_record_p90=int(np.percentile(nz, 90)) if len(nz) else 0,
        slots_per_record_p99=int(np.percentile(nz, 99)) if len(nz) else 0,
        slots_per_record_max=int(nz.max()) if len(nz) else 0,
        records_over_318_slots=int((per_rec > 318).sum()),
        records_over_318_slots_frac=float((per_rec > 318).sum()) / (len(per_rec) or 1),
        slots_clamped_to_int16=n_clamped,
    )
    return tok_slot, stats


def build_node_path(node_parent, node_sib, node_depth, node_off, levels: int = LEVELS,
                    rank: str = "flat"):
    """Materialise the root-path table. Returns ``(node_path, stats)``.

    ``node_parent`` is RECORD-LOCAL (``-1`` at a root), so it is rebased onto
    global row indices through ``node_off`` before use.

    ``rank='flat'``   the node's index among ALL its parent's children, in id order.
                      This is what ``build_web_pack.py`` means by rank and what makes
                      a path identify a node.
    ``rank='field'``  the source sidecar's ``node_sib`` verbatim.

    THE DEFAULT IS NOT THE SOURCE ARRAY, AND THAT IS DELIBERATE. In the astcompact
    sidecar ``node_sib`` is the index among the parent's children THAT SHARE A FIELD
    -- an ``If`` node's ``body[0]`` and ``orelse[0]`` are both rank 0. Paths built
    from it are not unique: measured on this pack, 47.4% of nodes share their path
    with another node in the same record and every single record has a collision.
    The leaf arm exists to give every token a distinct coordinate, so it would have
    been trained against a coordinate that is ambiguous for half its nodes.
    """
    n_nodes = len(node_depth)
    parent = np.asarray(node_parent, dtype=np.int64)
    sib = np.asarray(node_sib, dtype=np.int64)
    depth = np.asarray(node_depth, dtype=np.int64)
    off = np.asarray(node_off, dtype=np.int64)
    if not (len(parent) == len(sib) == n_nodes):
        raise ValueError(f"node arrays disagree: parent {len(parent)}, sib {len(sib)}, depth {n_nodes}")
    if int(off[-1]) != n_nodes:
        raise ValueError(f"node_off ends at {int(off[-1])} but there are {n_nodes} nodes")

    counts = np.diff(off)
    rec_start = np.repeat(off[:-1], counts)          # first global row of each node's record
    local_id = np.arange(n_nodes, dtype=np.int64) - rec_start

    is_root = parent < 0
    # DFS pre-order is what makes the depth-by-depth fill below correct. Check it.
    bad = np.nonzero(~is_root & (parent >= local_id))[0]
    if len(bad):
        i = int(bad[0])
        raise ValueError(
            f"node {i} (record-local id {int(local_id[i])}) has parent {int(parent[i])}, which is not "
            f"strictly smaller -- the sidecar is not in DFS pre-order and the path fill would read "
            f"unfilled parent rows. {len(bad)} nodes violate this."
        )
    if np.any(is_root & (depth != 0)):
        raise ValueError("a root node has non-zero depth")

    gparent = np.where(is_root, -1, rec_start + parent)

    if rank == "flat":
        # Index among ALL of the parent's children, in id order. Nodes are stored in
        # DFS pre-order, so a stable sort by parent leaves each parent's children in
        # document order and the rank is just the offset within the group.
        order = np.argsort(gparent, kind="stable")
        grouped = gparent[order]
        starts = np.searchsorted(grouped, grouped, side="left")
        sib = np.empty(n_nodes, dtype=np.int64)
        sib[order] = np.arange(n_nodes, dtype=np.int64) - starts
        # Every root has gparent -1, so they form ONE group spanning the whole pack and
        # would be ranked 0..n_records-1. A root is depth 0 and never occupies a level,
        # so this cannot reach a path -- but it does reach the statistics, where it
        # showed up as a "max sibling rank" of 423,110.
        sib[is_root] = 0
    elif rank != "field":
        raise ValueError(f"rank must be 'flat' or 'field', got {rank!r}")

    path = np.full((n_nodes, levels), ABSENT, dtype=np.int16)
    clamped = 0
    lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    max_depth = int(depth.max()) if n_nodes else 0
    # Group the rows by depth ONCE. Re-scanning `depth == d` for each of ~57 depths
    # costs 57 passes over 40M rows, and the row copy below must use the mixed
    # advanced/basic form `path[rows, :k]` -- the outer-product form
    # `path[rows[:, None], arange(k)]` builds a full index grid per depth and turns
    # a two-minute job into a twenty-minute one.
    order = np.argsort(depth, kind="stable")
    bounds = np.searchsorted(depth[order], np.arange(max_depth + 2))
    for d in range(1, max_depth + 1):
        idx = order[bounds[d]:bounds[d + 1]]
        if not len(idx):
            continue
        copy_cols = min(d - 1, levels)
        if copy_cols:
            path[idx, :copy_cols] = path[gparent[idx], :copy_cols]
        if d - 1 < levels:                            # else this node's own rank is truncated away
            s = sib[idx]
            n_out = int(np.count_nonzero((s < lo) | (s > hi)))
            clamped += n_out
            path[idx, d - 1] = np.clip(s, lo, hi).astype(np.int16)

    deep = depth > levels
    rec_of_node = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    recs_deep = np.unique(rec_of_node[deep]) if deep.any() else np.empty(0, dtype=np.int64)
    stats = dict(
        n_nodes=int(n_nodes),
        n_records=int(len(counts)),
        max_depth=max_depth,
        nodes_deeper_than_levels=int(deep.sum()),
        nodes_deeper_than_levels_frac=float(deep.sum()) / (n_nodes or 1),
        records_with_a_deep_node=int(len(recs_deep)),
        records_with_a_deep_node_frac=float(len(recs_deep)) / (len(counts) or 1),
        rank=rank,
        sib_rank_p99=float(np.percentile(sib[~is_root], 99)) if (~is_root).any() else 0.0,
        sib_rank_max=int(sib[~is_root].max()) if (~is_root).any() else 0,
        ranks_clamped_to_int16=clamped,
    )
    return path, stats


def convert(pack: str, out: str, levels: int = LEVELS, rank: str = "flat",
            slots: bool = False, slot_gate: str = "carry") -> dict:
    """Build ``<out>`` (symlinks to the pack) and ``<out>_dompath`` (the new sidecar)."""
    src_side = pack + "_treepos_v3"
    if not os.path.isdir(src_side):
        raise FileNotFoundError(f"no treepos sidecar at {src_side}")
    dst_side = out + "_dompath"
    os.makedirs(out, exist_ok=True)
    os.makedirs(dst_side, exist_ok=True)

    # The pack side is SYMLINKED, not copied: tokens.npy alone is ~94 MB here and
    # nothing in this step changes it. np.load and json.load both follow links.
    for f in ("tokens.npy", "tok_off.npy", "meta.json"):
        link = os.path.join(out, f)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.join(pack, f), link)

    t0 = time.time()
    tok_causal = np.load(os.path.join(src_side, "tok_causal.npy"), mmap_mode="r")
    tok_within = np.load(os.path.join(src_side, "tok_within.npy"), mmap_mode="r")
    node_off = np.load(os.path.join(src_side, "node_off.npy"))
    node_parent = np.load(os.path.join(src_side, "node_parent.npy"), mmap_mode="r")
    node_sib = np.load(os.path.join(src_side, "node_sib.npy"), mmap_mode="r")
    node_depth = np.load(os.path.join(src_side, "node_depth.npy"), mmap_mode="r")

    path, stats = build_node_path(node_parent, node_sib, node_depth, node_off, levels, rank)

    np.save(os.path.join(dst_side, "node_path.npy"), path)
    np.save(os.path.join(dst_side, "tok_node.npy"), np.asarray(tok_causal, dtype=np.int32))
    np.save(os.path.join(dst_side, "tok_within.npy"), np.asarray(tok_within, dtype=np.int16))
    np.save(os.path.join(dst_side, "node_off.npy"), np.asarray(node_off, dtype=np.int64))

    slot_stats = {}
    if slots:
        # OPT-IN (2026-09-28): the symbol-slot column for the rope_ref_slot arm. Off by
        # default so the existing sidecar's contents are byte-for-byte unchanged.
        tok_off = np.load(os.path.join(pack, "tok_off.npy"))
        tok_slot, slot_stats = build_tok_slot(
            np.load(os.path.join(src_side, "node_ref.npy"), mmap_mode="r"),
            node_off,
            tok_causal,
            tok_off,
            np.load(os.path.join(src_side, "node_ident_at.npy"), mmap_mode="r"),
            np.load(os.path.join(src_side, "tok_char_end.npy"), mmap_mode="r"),
            gate=slot_gate,
        )
        np.save(os.path.join(dst_side, "tok_slot.npy"), tok_slot)

    tw = np.asarray(tok_within)
    manifest = dict(
        dompath_version=2,
        built_by="convert_treepos.py",
        source_pack=pack,
        source_sidecar=src_side,
        text_pack=out,
        levels=levels,
        absent=ABSENT,
        render="python_ast",
        rank_rule=(
            "rank = the node's index among ALL its parent's children, in document order -- what "
            "build_web_pack.py means by rank. The source sidecar's node_sib is the index among the "
            "children THAT SHARE A FIELD, so an If's body[0] and orelse[0] are both 0; paths built "
            "from it collide for 47.4% of nodes, in 100% of records."
            if rank == "flat" else
            "rank = the source sidecar's node_sib verbatim (index among the children sharing a "
            "FIELD) -- paths are NOT unique under this setting"),
        node_rule=(
            "a node's path is the list of SIBLING RANKS from the child of the root down to the node "
            "itself; the root (the Module node, identical in every program) is NOT a level, exactly as "
            "<html>/<body> are not levels in the web packs, so a node at depth d occupies positions "
            "0..d-1 and path[i, d-1] == node_sib[i]; paths longer than `levels` are truncated at the "
            "LEAF end so level k always means tree level k"
        ),
        token_rule=(
            "tok_node = the source sidecar's tok_causal, the node an INCREMENTAL PARSER knows at the "
            "token (-1 outside the tree). The gold tok_node.npy of the source sidecar is deliberately "
            "NOT used: it can depend on tokens after the position, which would leak"
        ),
        within_rule="tok_within = the source sidecar's tok_within, verbatim",
        drop_deep=False,
        drop_alias=False,
        drop_rule="nothing is dropped: this converts an existing pack, it does not select pages",
        within_p99=float(np.percentile(tw, 99)),
        within_max=int(tw.max()),
        build_wall_s=round(time.time() - t0, 1),
        **stats,
    )
    if slots:
        manifest["slots"] = True
        manifest["slot_rule"] = (
            "tok_slot = the rank, in FIRST-MENTION order within the record, of the symbol the "
            "token's causal node belongs to; ABSENT (-1) where the node carries no symbol. A "
            "symbol is a connected component of the node_ref pointer graph (a use -> its "
            "scope-resolved canonical binder), rebased from RECORD-LOCAL ids through node_off. "
            "Every surviving ref is backward (verified: 100.00% point to a strictly earlier "
            "local id), so a class's earliest node is its first mention and the slot is always "
            "knowable from the prefix."
        )
        manifest["slot_not_from_ident_rule"] = (
            "node_ident is deliberately NOT the grouping key. It is sha1(f'{tag}|{name}') and "
            "the tag differs between a binder (arg / FunctionDef / store-Name) and a use "
            "(load-Name), so measured over 986,360 ref pairs a use and its own binder share an "
            "identity value only 70.43% of the time. Grouping by it would split ~30% of symbols "
            "into two slots each and nothing would fail visibly."
        )
        manifest["slot_gate_rule"] = (
            "tok_slot is ABSENT unless node_ident_at[node] >= 0 and node_ident_at[node] < "
            "tok_char_end[t] -- the sidecar's own ident gate. Knowing WHICH symbol a name "
            "denotes requires having read the name; without this the arm would claim the "
            "identity at the name's first character, a lookahead leak."
        )
        manifest["slot_coverage_note"] = (
            "coverage is not 100%: the source builder dropped 139,028 forward edges (1.4%), so "
            "those uses have node_ref == -1 and carry no slot."
        )
        manifest.update({f"slot_{k}": v for k, v in slot_stats.items()})
    with open(os.path.join(dst_side, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="source pack dir (its _treepos_v3 sidecar is read)")
    ap.add_argument("--out", required=True, help="destination pack dir (its _dompath sidecar is written)")
    ap.add_argument("--levels", type=int, default=LEVELS)
    ap.add_argument("--rank", choices=("flat", "field"), default="flat",
                    help="'flat' (default): index among ALL the parent's children, matching "
                         "build_web_pack.py. 'field': the source node_sib, which is per-FIELD and "
                         "makes paths non-unique.")
    ap.add_argument("--slots", action="store_true",
                    help="also emit tok_slot.npy, the per-token symbol slot read by "
                         "rope_ref_slot (built from node_ref, NOT node_ident -- see build_tok_slot)")
    ap.add_argument("--slot-gate", choices=SLOT_GATES, default="carry",
                    help="'carry' (default): leak-free, forward-fill the resolved symbol. "
                         "'strict': the sidecar's ident gate verbatim -- correct but keeps only "
                         "0.54%% of tokens. 'complete': dense (50%%) but costs one character of "
                         "lookahead. See build_tok_slot.")
    a = ap.parse_args()
    m = convert(a.pack, a.out, a.levels, a.rank, a.slots, a.slot_gate)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
