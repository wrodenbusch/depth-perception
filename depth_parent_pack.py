#!/usr/bin/env python
"""Derive a pack whose tree coordinate is TWO numbers: the node's depth and a SIBLING-GROUP tag.

    .venv/bin/python depth_parent_pack.py --pack $TOPOS_PACKS/py_ast_423k

THE IDEA (author, 2026-10-05): "depth plus parent. Parent will not actually have an identifier other
than the number associated with the children ... all the children of 20 get marked as 20.
Essentially tagging them all as a sibling group." Each node carries ``[depth, parent_tag]``. The
parent tag numbers the node's PARENT, so every child of one parent shares it: a token can tell that
another token is in its own sibling group. Unlike sibling rank (``depth_rank_pack.py``), which
restarts under every parent and so says nothing across groups, the tag is unique within the
document -- the tree's analogue of a spreadsheet row shared by all the cells in it.

DEFINITION. Within a record, a node at depth ``d`` has parent path ``path[:d-1]`` (the root's
children share the empty path). ``parent_tag`` = the order in which that parent path FIRST appears
among the record's nodes, taken in node-table order: the first parent seen is 0, the next new one
1, and so on. Identity comes from the path itself, so it works when the parent emits no token and
has no row of its own (web text nodes, spreadsheet cells).

CAUSALITY. Node tables are stored in prefix (pre-order) order, which is the order the tokens
reach them, and the tag only counts parents seen at or before a node. So a token's tag depends on
nothing later in the record -- valid for a causal model. ``--check`` verifies prefix order on a
sample (every node's parent path, if it is a row, appears earlier).

WRAP. The tag is rotated like any level: ``theta + tag * step`` on a fast and a slow plane. The
slow plane starts at 0.316/16 rad and wraps after ~318 steps; ``main`` reports the distribution of
distinct parents per record so a corpus whose tags exceed that is visible rather than aliased.
``--per-depth`` switches to the fallback in which the tag counts parents only among those at the
same depth (smaller numbers; depth already separates the levels).

EVERYTHING EXCEPT ``node_path`` IS A SYMLINK, so the token stream is byte-identical to the source.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def parent_tags(P: np.ndarray, rec: np.ndarray, per_depth: bool = False):
    """``(depth, tag)`` for a block of whole records.

    ``P`` is ``(n, L)`` node paths, ``rec`` the record id of each row (non-decreasing). The parent
    key is (record, [depth if per_depth], parent path); its tag is the rank, within its record
    (and depth), of the key's first row.
    """
    n, L = P.shape
    present = P >= 0
    depth = present.sum(1).astype(np.int64)
    par = np.where(np.arange(L)[None, :] < (depth - 1)[:, None], P, -1).astype(np.int16)
    grp = rec.astype(np.int64) * 64 + (depth if per_depth else 0)
    key = np.concatenate([grp.view(np.int16).reshape(n, 4), par], axis=1)
    key = np.ascontiguousarray(key).view(np.dtype((np.void, key.dtype.itemsize * key.shape[1]))).ravel()
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    # rank each key's first row within its group (record, or record x depth)
    # order keys by (group, first row): with --per-depth the groups interleave in row order, so
    # sorting by first row alone would not leave each group's keys contiguous
    order = np.lexsort((first, grp[first]))
    g_of_key = grp[first[order]]
    start = np.r_[0, np.flatnonzero(g_of_key[1:] != g_of_key[:-1]) + 1]
    rank_sorted = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    rank = np.empty(len(order), np.int64)
    rank[order] = rank_sorted
    tag = rank[inv.ravel()]
    tag = np.where(depth > 0, tag, -1)
    return depth, tag


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", default=None, help="default: <pack>_dparent (or _dparentd with --per-depth)")
    ap.add_argument("--per-depth", action="store_true", help="tag counts parents at the same depth only")
    ap.add_argument("--stats-only", action="store_true", help="report tag statistics, write nothing")
    ap.add_argument("--records-per-chunk", type=int, default=20_000)
    a = ap.parse_args()

    src, side = a.pack.rstrip("/"), a.pack.rstrip("/") + "_dompath"
    P = np.load(os.path.join(side, "node_path.npy"), mmap_mode="r")
    off = np.asarray(np.load(os.path.join(side, "node_off.npy")), dtype=np.int64)
    nrec = len(off) - 1
    out = a.out or f"{src}_dparent{'d' if a.per_depth else ''}"
    out_side = out + "_dompath"

    D = None
    if not a.stats_only:
        os.makedirs(out, exist_ok=True)
        os.makedirs(out_side, exist_ok=True)
        for f in sorted(os.listdir(src)):
            link(os.path.join(src, f), os.path.join(out, f))
        D = np.lib.format.open_memmap(os.path.join(out_side, "node_path.npy"), mode="w+",
                                      dtype=P.dtype, shape=(len(P), 2))
        if off[-1] < len(P):                       # rows past the last record, if any: ABSENT
            D[off[-1]:] = -1

    per_rec_max = np.full(nrec, -1, np.int64)
    prefix_bad = prefix_checked = 0
    for r0 in range(0, nrec, a.records_per_chunk):
        r1 = min(nrec, r0 + a.records_per_chunk)
        a0, a1 = int(off[r0]), int(off[r1])
        if a1 == a0:
            continue
        blk = np.asarray(P[a0:a1])
        rec = np.repeat(np.arange(r0, r1), np.diff(off[r0:r1 + 1]))
        depth, tag = parent_tags(blk, rec, a.per_depth)
        if D is not None:
            D[a0:a1, 0] = np.where(depth > 0, depth, -1)
            D[a0:a1, 1] = tag
        np.maximum.at(per_rec_max, rec, tag)
        # prefix order on the first record of the chunk: a parent row, when present, precedes its children
        rr = slice(0, int(off[r0 + 1] - off[r0]))
        rows = {tuple(x[x >= 0]): i for i, x in reversed(list(enumerate(blk[rr])))}
        for i, x in enumerate(blk[rr]):
            p = tuple(x[x >= 0])[:-1]
            if p and p in rows:
                prefix_checked += 1
                prefix_bad += rows[p] > i
    if D is not None:
        D.flush()

    n_par = per_rec_max + 1
    n_par = n_par[n_par > 0]
    q = np.percentile(n_par, [50, 90, 99]) if len(n_par) else [0, 0, 0]
    stats = dict(pack=src, per_depth=a.per_depth, records=int(nrec),
                 distinct_parents_p50=float(q[0]), p90=float(q[1]), p99=float(q[2]),
                 max=int(n_par.max()) if len(n_par) else 0,
                 frac_records_over_318=float((n_par > 318).mean()) if len(n_par) else 0.0,
                 prefix_order_checked=prefix_checked, prefix_order_violations=prefix_bad)
    print(json.dumps(stats))
    if D is None:
        return

    for f in sorted(os.listdir(side)):
        if f not in ("node_path.npy", "manifest.json"):
            link(os.path.join(side, f), os.path.join(out_side, f))
    man = {}
    if os.path.exists(os.path.join(side, "manifest.json")):
        man = json.load(open(os.path.join(side, "manifest.json")))
    man.update(derived_from=side, levels=2, depth_parent=True, tag_stats=stats,
               rule=("node_path replaced by [depth, parent_tag]: parent_tag = order of first "
                     "appearance of the node's parent path within the record"
                     + (", counted among parents at the same depth" if a.per_depth else "")
                     + "; ABSENT outside the tree; every other array a symlink to the source"))
    json.dump(man, open(os.path.join(out_side, "manifest.json"), "w"), indent=1)

    # verify on sampled records against a direct per-node recomputation
    rng = np.random.default_rng(0)
    for r in rng.choice(nrec, size=min(300, nrec), replace=False):
        a0, a1 = int(off[r]), int(off[r + 1])
        seen: dict = {}
        for i in range(a0, a1):
            x = np.asarray(P[i]); d = int((x >= 0).sum())
            if d == 0:
                assert tuple(D[i]) == (-1, -1), (r, i)
                continue
            k = ((d,) if a.per_depth else ()) + tuple(x[:d - 1])
            if a.per_depth:
                seen.setdefault(d, {})
                t = seen[d].setdefault(k, len(seen[d]))
            else:
                t = seen.setdefault(k, len(seen))
            assert (int(D[i, 0]), int(D[i, 1])) == (d, t), (r, i, tuple(D[i]), (d, t))
    print(f"  verified on 300 records; pack -> {out}")


if __name__ == "__main__":
    main()
