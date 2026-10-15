"""Derive the ANCHOR-SUBSTITUTED Python sidecar: a variable use takes its BINDER's tree path.

WHAT THIS IS, AND WHY IT NEEDS NO MODEL CHANGE
-----------------------------------------------
The spreadsheet arm that works (`rope_ref_anchor`) has a reference OVERWRITE its own tree path
levels with its target's, so a reference and its referent land on the same rotary angle, on the
same planes, by exactly the amount two tokens in one cell do. It adds no plane group at all.

The Python analogue is: a variable USE takes the tree path of its resolved BINDER. And
because the loader builds a token's path by gathering ``node_path[tok_node[t]]``, the whole
mechanism reduces to **substituting tok_node**. So there is no new ``pos_mode``, no new plane
group, no model edit and no risk of a rotation bug: the arm is literally ``rope_both`` reading a
sidecar whose ``tok_node`` points at the referent instead of at the token's own node.

That makes it the cleanest comparison available. The baseline is the EXISTING ``rope_both`` runs
on ``py_ast_423k`` -- same model, same tokens, same schedule, same seeds, already complete at
n=5. The two packs differ in exactly one array, and that difference IS the mechanism.

WHY THE PLANE-BUDGET OBJECTION NO LONGER APPLIES
-------------------------------------------------
``docs/design-reference-rotation.md`` rejected the path anchor for Python and chose a 2-plane
symbol SLOT instead, on the grounds that a full 22-level path would need a separate ~42-plane
group. Overwriting needs zero extra planes, so that objection is void -- and the slot arm is the
one that failed: measured on ``ref_slot_s0``, the model drove ``theta_slot`` from exactly 0 to
1.23 rad (a presence flag) while ``step_slot``, which discriminates WHICH symbol, barely moved.
57% of its tokens had no slot at all and were mutually coincident.

WHICH ARRAY, AND WHY NOT tok_slot
----------------------------------
The substitution reads ``node_ref`` from the SOURCE sidecar: "the prefix-emission id of the
pyscope-resolved canonical binder of a use, -1 when the target is emitted later". Record-local,
9,707,041 backward edges, and the corpus itself drops the 139,028 forward ones -- so it is
leak-free by construction and needs no gate of ours.

It is emphatically NOT ``tok_slot``, which I tried first and which would have been wrong.
``tok_slot`` is a CARRY signal: the manifest's own gate is "a token takes the slot of the most
recently RESOLVED symbol in its record", so 27% of tokens carry a slot they are not an occurrence
of. Substituting on it would have moved a quarter of all tokens to whatever variable was mentioned
most recently -- a different mechanism entirely, and a destructive one. The strict occurrence gate
keeps only 0.57% of tokens, far too sparse. ``node_ref`` is occurrence-level by construction.

Measured: 21.9% of tokens with a node carry a ref pointer, 100% of them resolved before the token,
zero self-references. A token whose node has no ref keeps its own node untouched, so unreferenced
tokens are geometrically identical to ``rope_both`` and the axis has no absent class -- nothing is
added to the geometry at all.
"""
from __future__ import annotations
from topos_paths import PACKS, SOURCES  # noqa: E402

import argparse
import json
import os

import numpy as np

ABSENT = -1
#: Arrays the substitution does not touch. Symlinked rather than copied -- node_path alone is
#: 1.77 GB and duplicating it would waste disk and risk the two copies drifting.
SHARED = ("tok_within.npy", "tok_slot.npy")
#: tok_slot rides along unchanged so the ref_slot arm can still read this pack, but the
#: anchor substitution does NOT use it -- see the module docstring.
PACK_SHARED = ("tokens.npy", "tok_off.npy", "meta.json")


def build(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--src", default=f"{PACKS}/py_ast_423k")
    ap.add_argument("--out", default=f"{PACKS}/py_ast_423k_anchor")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole pack")
    ap.add_argument("--mode", choices=("splice", "whole"), default="splice",
                    help="splice: overwrite only the binder-length PREFIX, keep the use's own "
                         "deeper levels (faithful to the sheets arm). whole: replace the entire "
                         "path (the first attempt -- truncates 92% of paths, measured worse).")
    ap.add_argument("--node-ref-from", default=f"{SOURCES}/"
                    "rep23_textpack_wholefile_apps_kodcode_423k_astcompact_ident32k_treepos_v3",
                    help="source sidecar holding node_ref.npy")
    a = ap.parse_args(argv)

    src_side, out_side = a.src + "_dompath", a.out + "_dompath"
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(out_side, exist_ok=True)

    tok_off = np.load(os.path.join(a.src, "tok_off.npy"))
    node_off = np.load(os.path.join(src_side, "node_off.npy"))
    tok_node = np.load(os.path.join(src_side, "tok_node.npy"), mmap_mode="r")
    # node_ref lives only in the SOURCE sidecar the derived pack was converted from.
    node_ref = np.load(os.path.join(a.node_ref_from, "node_ref.npy"), mmap_mode="r")
    src_noff = np.load(os.path.join(a.node_ref_from, "node_off.npy"))
    if not np.array_equal(src_noff, node_off):
        raise SystemExit("node_off differs between the derived sidecar and the source that "
                         "holds node_ref; the node id spaces are not aligned, refusing.")

    n_rec = len(tok_off) - 1
    if a.limit:
        n_rec = min(n_rec, a.limit)
    new = np.array(tok_node[: int(tok_off[n_rec])], dtype=np.int32)

    node_path = np.load(os.path.join(src_side, "node_path.npy"), mmap_mode="r")
    levels = node_path.shape[1]
    # SPLICE needs a path that is neither the use's nor the binder's, so it cannot be expressed
    # by pointing tok_node at an existing node. We append SYNTHETIC node rows per record and
    # point at those -- which keeps the loader's plain gather and still needs no model change.
    n_sub = n_with_node = n_has_ref = n_trunc_avoided = 0
    out_rows: list[np.ndarray] = []
    new_noff = [0]
    for r in range(n_rec):
        lo, hi = int(tok_off[r]), int(tok_off[r + 1])
        nlo, nhi = int(node_off[r]), int(node_off[r + 1])
        nr = np.asarray(node_ref[nlo:nhi])
        base = np.asarray(node_path[nlo:nhi])
        nodes = np.asarray(tok_node[lo:hi])
        extra: list[np.ndarray] = []
        cache: dict[tuple[int, int], int] = {}
        for i in range(hi - lo):
            nd = int(nodes[i])
            if nd == ABSENT or nd >= len(nr):
                continue
            n_with_node += 1
            tgt = int(nr[nd])
            if tgt < 0 or tgt >= len(base):
                continue
            n_has_ref += 1
            if tgt == nd:
                continue
            key = (nd, tgt)
            if key not in cache:
                if a.mode == "whole":
                    row = base[tgt].copy()
                else:
                    db = int((base[tgt] != ABSENT).sum())
                    du = int((base[nd] != ABSENT).sum())
                    if db < du:
                        n_trunc_avoided += 1
                    # the binder's whole path, then the USE's own levels from that depth on:
                    # identity of the referent in the prefix, the use's own role in the suffix.
                    # Exactly the shape of the sheets arm, which overwrites sheet/row/column and
                    # leaves the position within the formula alone.
                    row = base[nd].copy()
                    row[:db] = base[tgt][:db]
                cache[key] = len(base) + len(extra)
                extra.append(row)
            new[lo + i] = cache[key]
            n_sub += 1
        out_rows.append(base)
        if extra:
            out_rows.append(np.stack(extra))
        new_noff.append(new_noff[-1] + len(base) + len(extra))

    np.save(os.path.join(out_side, "node_path.npy"), np.concatenate(out_rows))
    np.save(os.path.join(out_side, "node_off.npy"), np.asarray(new_noff, dtype=np.int64))
    np.save(os.path.join(out_side, "tok_node.npy"), new)
    for f in SHARED:
        dst = os.path.join(out_side, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(src_side, f), dst)
    for f in PACK_SHARED:
        dst = os.path.join(a.out, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(a.src, f), dst)

    src_man = json.load(open(os.path.join(src_side, "manifest.json")))
    man = dict(src_man)
    man.update({
        "built_by": "build_py_anchor_sidecar.py",
        "derived_from": src_side,
        "anchor_rule": (
            "tok_node of a token whose node has a node_ref pointer is REPLACED by that "
            "pointer's target -- the pyscope-resolved canonical binder of the use. Every other "
            "token keeps its own node. The loader gathers node_path[tok_node], so a use rotates "
            "at its binder's tree position: the Python analogue of the spreadsheet "
            "rope_ref_anchor arm, with no new plane group and no model change."),
        "anchor_causality": (
            "node_ref is -1 when the binder is emitted later, so the corpus drops forward "
            "references itself (139,028 of them) and every surviving edge points backward. "
            "Measured over 4,000 records: 100% of referents were resolved strictly before the "
            "referring token. NOT derived from tok_slot, which is a CARRY signal -- the most "
            "recently resolved symbol -- and would have substituted 27% of tokens to a symbol "
            "they are not an occurrence of."),
        "anchor_n_tokens_substituted": int(n_sub),
        "anchor_n_tokens_with_a_node": int(n_with_node),
        "anchor_n_tokens_whose_node_has_a_ref": int(n_has_ref),
        "anchor_source_of_ref": "node_ref.npy",
        "anchor_mode": a.mode,
        "anchor_n_paths_that_would_have_truncated": int(n_trunc_avoided),
        "anchor_n_records": int(n_rec),
        "anchor_shared_arrays_symlinked": list(SHARED),
        "baseline": (
            "the EXISTING rope_both runs on " + a.src + " -- same model, tokens, schedule and "
            "seeds. The two sidecars differ in tok_node alone, and that difference is the "
            "mechanism."),
    })
    json.dump(man, open(os.path.join(out_side, "manifest.json"), "w"), indent=1)
    print(json.dumps({
        "records": n_rec,
        "tokens": int(len(new)),
        "tokens_with_a_node": n_with_node,
        "tokens_whose_node_has_a_ref": n_has_ref,
        "mode": a.mode,
        "tokens_substituted": n_sub,
        "paths_that_would_have_truncated": n_trunc_avoided,
        "substituted_frac_of_all_tokens": round(n_sub / max(len(new), 1), 4),
        "out": a.out, "sidecar": out_side,
    }, indent=1))


if __name__ == "__main__":
    build()
