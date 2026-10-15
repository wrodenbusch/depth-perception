#!/usr/bin/env python
"""Derive a pack whose ``node_path`` keeps only the first N tree levels. The plane-matched control.

    .venv/bin/python slice_levels.py --pack $TOPOS_PACKS/sheet_values_v2 --levels 1

WHY A DERIVED PACK RATHER THAN A CONFIG FLAG. The rotary module refuses a path with more columns
than it has levels, and that refusal is correct -- silently ignoring columns is how an arm ends up
rotating something other than what its config says. So an arm that spends only N levels needs data
with only N levels. ``py_ast_423k_lvl1`` was built by hand for the Python control and the steps
were never written down; the sheets control then failed the same way, which is why this is a script.

EVERYTHING EXCEPT ``node_path`` IS A SYMLINK, deliberately. The token stream, record offsets, node
offsets and ``tok_node`` are byte-identical to the source, so the derived arm differs from its
baseline in the rotated coordinate and in nothing else. Symlinks make that true by construction
rather than by a copy that could drift, and they also keep ``chart_ablation.py``'s mixed-pack guard
satisfiable: it can verify the streams are the same bytes.

The slice keeps the SHALLOWEST levels (the prefix), so level k still means tree level k and a
one-level path is "which top-level child of the root this node descends from".
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


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True, help="source pack dir (its _dompath sidecar is read)")
    ap.add_argument("--levels", type=int, required=True)
    ap.add_argument("--out", default=None, help="default: <pack>_lvl<N>")
    a = ap.parse_args()

    src, side = a.pack.rstrip("/"), a.pack.rstrip("/") + "_dompath"
    if not os.path.isdir(side):
        raise SystemExit(f"no sidecar at {side}")
    out = a.out or f"{src}_lvl{a.levels}"
    out_side = out + "_dompath"
    os.makedirs(out, exist_ok=True)
    os.makedirs(out_side, exist_ok=True)

    # the base pack: every file a symlink, so the token stream is provably identical
    for f in sorted(os.listdir(src)):
        link(os.path.join(src, f), os.path.join(out, f))

    P = np.load(os.path.join(side, "node_path.npy"), mmap_mode="r")
    if a.levels > P.shape[1]:
        raise SystemExit(f"--levels {a.levels} exceeds the source's {P.shape[1]}")
    sliced = np.ascontiguousarray(np.asarray(P[:, :a.levels]))
    np.save(os.path.join(out_side, "node_path.npy"), sliced)
    for f in sorted(os.listdir(side)):
        if f in ("node_path.npy", "manifest.json"):
            continue
        link(os.path.join(side, f), os.path.join(out_side, f))

    man = {}
    mp = os.path.join(side, "manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp))
    man.update(derived_from=side, levels=a.levels, sliced_levels=True,
               slice_rule=("node_path truncated to the first %d level(s); every other array is a "
                           "symlink to the source, so the token stream is byte-identical" % a.levels))
    json.dump(man, open(os.path.join(out_side, "manifest.json"), "w"), indent=1)

    # prove it: the kept columns must equal the source's, and nothing else may have changed
    chk = np.load(os.path.join(out_side, "node_path.npy"), mmap_mode="r")
    n = min(len(chk), 2_000_000)
    assert np.array_equal(np.asarray(chk[:n]), np.asarray(P[:n, :a.levels])), "slice mismatch"
    print(f"{src}  node_path {P.shape} -> {chk.shape}")
    print(f"  wrote {out_side}/node_path.npy "
          f"({os.path.getsize(os.path.join(out_side, 'node_path.npy'))/2**20:.0f} MiB)")
    print(f"  verified the kept columns match the source on {n:,} rows")
    print(f"  everything else symlinked; pack -> {out}")


if __name__ == "__main__":
    main()
