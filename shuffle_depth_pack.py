#!/usr/bin/env python
"""Derive a pack whose node depths are randomly permuted within each record: the shuffled-depth control.

    .venv/bin/python shuffle_depth_pack.py --pack $TOPOS_PACKS/py_ast_423k --seed 0

WHY. ``rope_depth`` gives two rotary planes a coordinate. If it helps only because two more
planes rotate -- capacity, not information -- then the same planes rotated by a depth that has
nothing to do with the tree should help as much. This pack keeps every record's MULTISET of node
depths but assigns them to the wrong nodes, so the arm keeps the planes, the value range and the
per-record depth distribution and loses only the information.

WHAT IS PERMUTED. Within each record, the depths of the nodes that are IN the tree (depth >= 1)
are shuffled among themselves. Nodes outside the tree (depth 0, every level ABSENT) keep depth 0:
"outside the tree" marks framing and separator tokens the baseline can already identify from
the token itself, so moving it would add noise of a different kind rather than remove tree
information. One property survives by construction: all tokens of one node still share a value,
because the permutation acts on nodes, not tokens.

ENCODING. ``rope_depth`` reads depth as the count of non-ABSENT levels in ``node_path``, so a
node given depth d' gets a row of d' zeros followed by ABSENT. The sibling ranks are gone too,
which is fine: the control is only run under ``rope_depth``, which never reads them.

EVERYTHING EXCEPT ``node_path`` IS A SYMLINK, as in ``slice_levels.py``: the token stream is
byte-identical to the source, so nats/token are directly comparable with every arm on it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ABSENT = -1


def link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def shuffled_depths(depth: np.ndarray, node_off: np.ndarray, seed: int) -> np.ndarray:
    """Per record, permute the in-tree (depth >= 1) depths; depth-0 nodes stay 0."""
    rng = np.random.default_rng(seed)
    out = depth.copy()
    for r in range(len(node_off) - 1):
        a, b = int(node_off[r]), int(node_off[r + 1])
        d = depth[a:b]
        idx = np.flatnonzero(d > 0)
        if len(idx) > 1:
            out[a + idx] = d[idx][rng.permutation(len(idx))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--seed", type=int, default=0, help="permutation seed (fixed across runs)")
    ap.add_argument("--out", default=None, help="default: <pack>_depthshuf")
    a = ap.parse_args()

    src, side = a.pack.rstrip("/"), a.pack.rstrip("/") + "_dompath"
    out = a.out or f"{src}_depthshuf"
    out_side = out + "_dompath"
    os.makedirs(out, exist_ok=True)
    os.makedirs(out_side, exist_ok=True)
    for f in sorted(os.listdir(src)):
        link(os.path.join(src, f), os.path.join(out, f))

    P = np.load(os.path.join(side, "node_path.npy"), mmap_mode="r")
    node_off = np.load(os.path.join(side, "node_off.npy"))
    if int(node_off[-1]) != len(P):
        raise SystemExit(f"node_off ends at {node_off[-1]} but node_path has {len(P)} rows")
    L = P.shape[1]
    step = 20_000_000
    depth = np.empty(len(P), dtype=np.int16)
    for i in range(0, len(P), step):
        depth[i:i + step] = (np.asarray(P[i:i + step]) >= 0).sum(1)
    new = shuffled_depths(depth, node_off, a.seed)

    D = np.lib.format.open_memmap(os.path.join(out_side, "node_path.npy"), mode="w+",
                                  dtype=P.dtype, shape=P.shape)
    cols = np.arange(L)[None, :]
    for i in range(0, len(P), step):
        nd = new[i:i + step, None]
        D[i:i + step] = np.where(cols < nd, 0, ABSENT).astype(P.dtype)
    D.flush()
    for f in sorted(os.listdir(side)):
        if f not in ("node_path.npy", "manifest.json"):
            link(os.path.join(side, f), os.path.join(out_side, f))
    man = {}
    if os.path.exists(os.path.join(side, "manifest.json")):
        man = json.load(open(os.path.join(side, "manifest.json")))
    man.update(derived_from=side, depth_shuffled=True, shuffle_seed=a.seed,
               rule="per record, depths of in-tree nodes (depth>=1) randomly permuted; depth-0 "
                    "nodes unchanged; node_path rows encode only the depth (d zeros then ABSENT); "
                    "every other array a symlink to the source")
    json.dump(man, open(os.path.join(out_side, "manifest.json"), "w"), indent=1)

    # prove it: depth from the new rows equals the shuffled depth; each record keeps its
    # multiset of depths; depth-0 nodes are untouched; and the shuffle actually moved values
    rng = np.random.default_rng(1)
    rec = rng.choice(len(node_off) - 1, size=min(2000, len(node_off) - 1), replace=False)
    moved = total = 0
    for r in rec:
        a_, b_ = int(node_off[r]), int(node_off[r + 1])
        got = (np.asarray(D[a_:b_]) >= 0).sum(1)
        assert np.array_equal(got, new[a_:b_]), r
        assert np.array_equal(np.sort(got), np.sort(depth[a_:b_])), r
        assert np.array_equal(got == 0, depth[a_:b_] == 0), r
        inside = depth[a_:b_] > 0
        moved += int((got[inside] != depth[a_:b_][inside]).sum())
        total += int(inside.sum())
    print(f"{src}: {len(P):,} nodes, {len(node_off) - 1:,} records; verified on {len(rec):,} "
          f"records -- multiset kept, depth-0 kept, {100 * moved / max(total, 1):.1f}% of in-tree "
          f"nodes moved to a different depth")
    print(f"  pack -> {out}")


if __name__ == "__main__":
    main()
