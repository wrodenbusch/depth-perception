#!/usr/bin/env python
"""Derive a pack whose tree coordinate is TWO numbers: the node's depth and its own sibling rank.

    .venv/bin/python depth_rank_pack.py --pack $TOPOS_PACKS/py_ast_423k

WHY. On spreadsheets the whole structural gain came from a grid -- row and column, two numbers --
not from the full 22-level path. The analogue for any tree is a "tree grid": how deep the node is
(the vertical coordinate) and which child of its parent it is (the horizontal one). This tests
whether those two numbers carry what the full path carries on code and web text.

HOW, WITHOUT NEW MODEL CODE. The derived ``node_path`` has two columns, ``[depth, own_rank]``,
and the run uses ``rope_both`` with ``rope_levels = 2``. Level 0's planes are then rotated by
``theta + depth * step`` and level 1's by ``theta + own_rank * step`` -- the same per-level
rotation, ladder and presence handling the full path uses, applied to these two numbers. A node
outside the tree (depth 0) gets ``[-1, -1]``, ABSENT on both, exactly as before.

EVERYTHING EXCEPT ``node_path`` IS A SYMLINK, as in ``slice_levels.py``, so the token stream is
byte-identical to the source and the comparison is the rotated coordinate and nothing else.
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


def depth_rank(P: np.ndarray) -> np.ndarray:
    """``(n, 2)`` int: ``[depth, rank at the deepest present level]``; ``[-1, -1]`` outside the tree."""
    present = P >= 0
    depth = present.sum(1)
    # the path is a contiguous prefix (a dropped level makes the node read as shallower), so the
    # node's own rank sits at column depth-1
    if not np.all(present == (np.arange(P.shape[1])[None, :] < depth[:, None])):
        raise SystemExit("node_path is not a contiguous prefix; depth-1 is not the node's own level")
    out = np.full((len(P), 2), -1, dtype=P.dtype)
    inside = depth > 0
    out[inside, 0] = depth[inside]
    out[inside, 1] = P[inside, depth[inside] - 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", default=None, help="default: <pack>_drank")
    a = ap.parse_args()

    src, side = a.pack.rstrip("/"), a.pack.rstrip("/") + "_dompath"
    out = a.out or f"{src}_drank"
    out_side = out + "_dompath"
    os.makedirs(out, exist_ok=True)
    os.makedirs(out_side, exist_ok=True)
    for f in sorted(os.listdir(src)):
        link(os.path.join(src, f), os.path.join(out, f))

    # Chunked through a memmap: the long-context web pack's node_path is 135 GB, more than this
    # machine can hold, while the derived two-column array is a tenth of that.
    P = np.load(os.path.join(side, "node_path.npy"), mmap_mode="r")
    D = np.lib.format.open_memmap(os.path.join(out_side, "node_path.npy"), mode="w+",
                                  dtype=P.dtype, shape=(len(P), 2))
    step = 20_000_000
    for i in range(0, len(P), step):
        D[i:i + step] = depth_rank(np.asarray(P[i:i + step]))
    D.flush()
    for f in sorted(os.listdir(side)):
        if f not in ("node_path.npy", "manifest.json"):
            link(os.path.join(side, f), os.path.join(out_side, f))
    man = {}
    if os.path.exists(os.path.join(side, "manifest.json")):
        man = json.load(open(os.path.join(side, "manifest.json")))
    man.update(derived_from=side, levels=2, depth_rank=True,
               rule="node_path replaced by [depth, own sibling rank]; ABSENT outside the tree; "
                    "every other array a symlink to the source")
    json.dump(man, open(os.path.join(out_side, "manifest.json"), "w"), indent=1)

    # prove it on a sample: depth matches the source's present count, rank its deepest entry
    idx = np.random.default_rng(0).choice(len(P), size=min(200_000, len(P)), replace=False)
    for i in idx:
        row = np.asarray(P[i])
        d = int((row >= 0).sum())
        assert (D[i, 0], D[i, 1]) == ((d, row[d - 1]) if d else (-1, -1)), i
    print(f"{src}: node_path {P.shape} -> {D.shape}; depth max {D[:, 0].max()}, "
          f"own-rank p99 {np.percentile(D[D[:, 1] >= 0, 1], 99):.0f}; verified on {len(idx):,} rows")
    print(f"  pack -> {out}")


if __name__ == "__main__":
    main()
