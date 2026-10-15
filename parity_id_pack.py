#!/usr/bin/env python
"""Derive a pack whose tree coordinate is PARITY-SPLIT NODE IDs: a child coincides with its parent.

    .venv/bin/python parity_id_pack.py --pack $TOPOS_PACKS/py_ast_423k

THE IDEA (author, 2026-10-06). Put children close to their parent by alternating two plane groups.
Every node has an ordinal ``o`` (its index in the record's prefix-ordered node table, + 1, so the
virtual root is 0) and a parent ordinal ``p``. Group A carries the nearest EVEN-depth node among
{self, parent}; group B the nearest ODD-depth one:

    A = o if depth is even else p          B = o if depth is odd else p

So a child equals its parent in exactly one group (the one of the parent's parity), and every node
equals its own children in the other. Parent <-> child alignment is exact at any distance, which is
what the sibling-group parent tag could not give: siblings are adjacent in a prefix serialisation,
so sequence position already related them, whereas a statement deep in a long function body can be
hundreds of tokens from the ``def`` that is its parent.

OUTPUT. ``node_path`` = ``[A, B, depth]``, rotated by ``rope_both`` with ``rope_levels = 3``, the
same per-level ladder and presence handling the full path uses. Nodes outside the tree (depth 0)
are ABSENT on all three. Everything except ``node_path`` is a symlink to the source, so the token
stream is byte-identical.

CAUSAL. A node's ordinal is fixed when it opens (prefix order: the table is in document order and
a parent row precedes its children -- verified here), and its parent opened earlier, so A and B
use nothing a causal model has not already seen.
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


def _keys(rec: np.ndarray, paths: np.ndarray) -> np.ndarray:
    k = np.concatenate([rec.astype(np.int64).view(np.int16).reshape(len(rec), 4),
                        paths.astype(np.int16)], axis=1)
    k = np.ascontiguousarray(k)
    return k.view(np.dtype((np.void, k.dtype.itemsize * k.shape[1]))).ravel()


def parity_ids(P: np.ndarray, rec: np.ndarray, local: np.ndarray):
    """``(A, B, depth)`` for a block of whole records; ``local`` = row index within its record."""
    n, L = P.shape
    present = P >= 0
    depth = present.sum(1).astype(np.int64)
    own = _keys(rec, P)
    par_path = np.where(np.arange(L)[None, :] < (depth - 1)[:, None], P, -1)
    par = _keys(rec, par_path)
    order = np.argsort(own, kind="stable")
    sown = own[order]
    pos = np.searchsorted(sown, par)
    pos = np.clip(pos, 0, n - 1)
    found = sown[pos] == par
    par_row = order[pos]                                   # first row with that path (stable sort)
    o = local + 1
    p = np.where(depth <= 1, 0, local[par_row] + 1)
    missing = int(((depth > 1) & ~found).sum())
    after = int(((depth > 1) & found & (par_row > np.arange(n))).sum())
    even = depth % 2 == 0
    A = np.where(even, o, p)
    B = np.where(even, p, o)
    out = np.stack([A, B, depth], 1)
    out[depth == 0] = -1
    return out, missing, after


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", default=None, help="default: <pack>_parityid")
    ap.add_argument("--records-per-chunk", type=int, default=20_000)
    a = ap.parse_args()

    src, side = a.pack.rstrip("/"), a.pack.rstrip("/") + "_dompath"
    out = a.out or f"{src}_parityid"
    out_side = out + "_dompath"
    os.makedirs(out, exist_ok=True)
    os.makedirs(out_side, exist_ok=True)
    for f in sorted(os.listdir(src)):
        link(os.path.join(src, f), os.path.join(out, f))

    P = np.load(os.path.join(side, "node_path.npy"), mmap_mode="r")
    off = np.load(os.path.join(side, "node_off.npy"))
    nrec = len(off) - 1
    D = np.lib.format.open_memmap(os.path.join(out_side, "node_path.npy"), mode="w+",
                                  dtype=np.int16, shape=(len(P), 3))
    if off[-1] < len(P):
        D[off[-1]:] = -1
    miss = after = 0
    over = 0
    maxid = np.zeros(nrec, np.int64)
    for r0 in range(0, nrec, a.records_per_chunk):
        r1 = min(nrec, r0 + a.records_per_chunk)
        a0, a1 = int(off[r0]), int(off[r1])
        if a1 == a0:
            continue
        blk = np.asarray(P[a0:a1])
        sizes = np.diff(off[r0:r1 + 1])
        rec = np.repeat(np.arange(r0, r1), sizes)
        local = np.arange(a1 - a0) - np.repeat(off[r0:r1] - a0, sizes)
        res, m, af = parity_ids(blk, rec, local)
        if res[:, :2].max() > np.iinfo(np.int16).max:
            raise SystemExit("node ordinal exceeds int16")
        D[a0:a1] = res.astype(np.int16)
        miss += m
        after += af
        np.maximum.at(maxid, rec, np.maximum(res[:, 0], res[:, 1]))
    D.flush()
    over = int((maxid + 1 > 318).sum())

    for f in sorted(os.listdir(side)):
        if f not in ("node_path.npy", "manifest.json"):
            link(os.path.join(side, f), os.path.join(out_side, f))
    man = json.load(open(os.path.join(side, "manifest.json"))) if os.path.exists(
        os.path.join(side, "manifest.json")) else {}
    ids = maxid[maxid >= 0] + 1
    stats = dict(records=nrec, ids_per_record_p50=float(np.percentile(ids, 50)),
                 p99=float(np.percentile(ids, 99)), max=int(ids.max()),
                 records_over_318=over, parents_missing=miss, parents_after_child=after)
    man.update(derived_from=side, levels=3, parity_id=True, stats=stats,
               rule="node_path replaced by [A, B, depth] (author's idea, 2026-10-06): o = node's index "
                    "in its record's prefix-ordered node table + 1 (virtual root 0), p = parent's o; "
                    "A = o if depth even else p, B = o if depth odd else p, so a child equals its "
                    "parent in exactly one group; ABSENT outside the tree; every other array a "
                    "symlink to the source")
    json.dump(man, open(os.path.join(out_side, "manifest.json"), "w"), indent=1)

    # verify on a sample of records with a direct per-node recomputation
    rng = np.random.default_rng(0)
    bad = checked = 0
    for r in rng.choice(nrec, min(300, nrec), replace=False):
        a0, a1 = int(off[r]), int(off[r + 1])
        Pr = np.asarray(P[a0:a1])
        Dr = np.asarray(D[a0:a1])
        first = {}
        for i, x in enumerate(Pr):
            first.setdefault(tuple(x[x >= 0]), i)
        for i, x in enumerate(Pr):
            k = tuple(x[x >= 0])
            d = len(k)
            if d == 0:
                bad += int((Dr[i] != -1).any())
                continue
            o = i + 1
            p = 0 if d == 1 else first[k[:-1]] + 1
            exp = (o, p, d) if d % 2 == 0 else (p, o, d)
            checked += 1
            bad += int(tuple(int(v) for v in Dr[i]) != exp)
            if d > 1:                                  # child equals its parent in the parent's group
                pi = first[k[:-1]]
                g = 0 if (d - 1) % 2 == 0 else 1
                bad += int(Dr[i, g] != Dr[pi, g])
    print(f"{src}: node_path {P.shape} -> {D.shape}")
    print(f"  ids/record p50 {stats['ids_per_record_p50']:.0f} p99 {stats['p99']:.0f} max {stats['max']}; "
          f"records over the 2-plane wrap (318): {over} of {nrec}")
    print(f"  parents missing {miss}, parent after child {after}; sample check {checked:,} nodes, {bad} errors")
    if bad or miss or after:
        raise SystemExit("verification failed")
    print(f"  pack -> {out}")


if __name__ == "__main__":
    main()
