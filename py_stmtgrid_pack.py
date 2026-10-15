#!/usr/bin/env python
"""Derive the Python STATEMENT-GRID packs: (statement number, position within the statement) [+ AST depth].

    .venv/bin/python py_stmtgrid_pack.py --pack $TOPOS_PACKS/py_ast_423k

WHY (author, 2026-10-06). The grid family carries the whole gain on spreadsheets (row, column) and
chat (message, position-in-message), and both have the same shape: a GLOBAL unit number every
token of the unit shares, plus the token's position inside the unit. Python had no comparable
coordinate. Its natural row is the STATEMENT: statement 37 is statement 37 everywhere in the file,
which is the property that made the spreadsheet row work and that sibling rank / parent tags lack.

WHICH NODES ARE STATEMENTS. The token stream is a compact prefix serialisation of the AST; node
types are not stored in the pack, but the source tree sidecar (``<source_sidecar>/node_field``,
``node_parent``, ``node_depth``; record-local ids, the same node tables as the pack's) records every
node's ASDL field in its parent. A node is a statement iff its field is ``body`` / ``orelse`` /
``finalbody`` AND its parent is a statement CONTAINER: the Module root, another statement, an
ExceptHandler (field ``handlers``) or a match_case (field ``cases``). The container condition is what
excludes the expression fields that share those names (``Lambda.body``, ``IfExp.body/orelse``).
Field ids come from ``mathesis.repr2x2.pyscope.FIELD_ID`` (62 ids, matching the sidecar manifest's
``n_field_ids``): body 5, orelse 43, finalbody 17, handlers 22, cases 7.

THE COORDINATES, for a token whose causal node is ``n`` (``tok_node``, the node an INCREMENTAL
parser knows at that token):
- unit = the innermost statement containing ``n`` (``n`` itself if it is one);
- A = that statement's ordinal among the record's statements in document (pre-)order, global
  within the record: nested statements get their own numbers;
- B = the token's offset from the statement's first token (the first token whose node lies in the
  statement's subtree). Outer-statement tokens after a nested block keep counting from the outer
  statement's start, so B is "how far into this statement" in tokens.
Tokens outside every statement (framing EOTs, anything attached to the Module root) are ABSENT.
Both are causal: a statement's ordinal is fixed when it opens (pre-order), and its first token is
at or before every token that refers to it.

HOW, WITHOUT NEW MODEL CODE (exactly the chat grid's recipe). ``rope_all`` rotates the per-level
tree planes by ``present * theta + rank * step`` and the 4 token-leaf planes by ``tok_within``:
- ``<pack>_stmtgrid``: one node per statement, path ``[A]``, levels 1..21 ABSENT (zero rotation);
  ``tok_within`` = B.
- ``<pack>_stmtgriddepth``: one node per distinct (statement, AST depth of the causal node) in a
  record, path ``[A, depth]``; ``tok_within`` = B.
The level-0 ladder wraps at ~318 statements and the token ladder at 1,272; this script measures both
and REFUSES a pack where either would wrap. Everything except the sidecar's tok_node / node_path /
node_off / tok_within is a symlink to the source pack, so the token stream is byte-identical.
"""
from __future__ import annotations
from topos_paths import PACKS  # noqa: E402

import argparse
import json
import os

import numpy as np

from npy_stream import StreamNpy

ABSENT = -1
MAX_WITHIN = 1272          # slowest token plane (BASE_STEP / 4**3) wraps here
MAX_LEVEL = 318            # slowest level plane (BASE_STEP / 16) wraps here
STMT_FIELDS = (5, 43, 17)  # body, orelse, finalbody
CONTAINER_FIELDS = (22, 7)  # handlers (ExceptHandler), cases (match_case)


def link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def record_coords(par, fld, dep, tn):
    """Per-token (A, B, ast depth) for one record; A = -1 where the token has no statement."""
    n = len(par)
    is_stmt = np.zeros(n, bool)
    container = np.zeros(n, bool)
    container[0] = True                                   # node 0 is the Module root
    anc = np.full(n, -1, dtype=np.int64)                  # innermost enclosing statement (incl. self)
    order = np.argsort(dep, kind="stable")                # parents before children
    for d in np.unique(dep[1:]):
        idx = order[np.searchsorted(dep[order], d):np.searchsorted(dep[order], d, side="right")]
        p = par[idx]
        is_stmt[idx] = np.isin(fld[idx], STMT_FIELDS) & container[p]
        container[idx] = is_stmt[idx] | np.isin(fld[idx], CONTAINER_FIELDS)
        anc[idx] = np.where(is_stmt[idx], idx, anc[p])
    ordinal = np.cumsum(is_stmt) - 1                      # pre-order rank among statements
    # first token of each node's subtree: first occurrence, then min up the tree (children first)
    t_idx = np.arange(len(tn))
    inside = tn >= 0
    first = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, tn[inside], t_idx[inside])
    for d in np.unique(dep)[::-1]:
        if d == 0:
            continue
        idx = np.nonzero(dep == d)[0]
        np.minimum.at(first, par[idx], first[idx])
    s = np.full(len(tn), -1, dtype=np.int64)
    s[inside] = anc[tn[inside]]
    has = s >= 0
    A = np.full(len(tn), -1, dtype=np.int64)
    B = np.full(len(tn), -1, dtype=np.int64)
    A[has] = ordinal[s[has]]
    B[has] = t_idx[has] - first[s[has]]
    assert (B[has] >= 0).all(), "a token precedes its statement's first token"
    D = np.full(len(tn), -1, dtype=np.int64)
    D[inside] = dep[tn[inside]]
    return A, B, D, int(is_stmt.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source-sidecar", default=f"{PACKS}/"
                    "rep23_textpack_wholefile_apps_kodcode_423k_astcompact_ident32k_treepos_v3")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (testing)")
    a = ap.parse_args()
    src, side, ss = a.pack.rstrip("/"), a.pack.rstrip("/") + "_dompath", a.source_sidecar
    tok_off = np.load(os.path.join(src, "tok_off.npy"))
    tn_all = np.load(os.path.join(side, "tok_node.npy"), mmap_mode="r")
    no_all = np.load(os.path.join(side, "node_off.npy"))
    levels = np.load(os.path.join(side, "node_path.npy"), mmap_mode="r").shape[1]
    s_off = np.load(os.path.join(ss, "node_off.npy"))
    if not np.array_equal(s_off, no_all):
        raise SystemExit("source sidecar node tables do not align with the pack's")
    par_all = np.load(os.path.join(ss, "node_parent.npy"), mmap_mode="r")
    fld_all = np.load(os.path.join(ss, "node_field.npy"), mmap_mode="r")
    dep_all = np.load(os.path.join(ss, "node_depth.npy"), mmap_mode="r")

    outs = {}
    for name in ("stmtgrid", "stmtgriddepth"):
        out = f"{src}_{name}"
        os.makedirs(out, exist_ok=True)
        os.makedirs(out + "_dompath", exist_ok=True)
        for f in sorted(os.listdir(src)):
            link(os.path.join(src, f), os.path.join(out, f))
        o = out + "_dompath"
        outs[name] = dict(dir=o, tn=StreamNpy(os.path.join(o, "tok_node.npy"), np.int32),
                          np=StreamNpy(os.path.join(o, "node_path.npy"), np.int16, cols=levels),
                          tw=StreamNpy(os.path.join(o, "tok_within.npy"), np.int16), off=[0])
    n_rec = len(tok_off) - 1 if not a.limit else a.limit
    n_stmt, max_a, b_vals, absent_tok, tot_tok = [], 0, [], 0, 0
    for r in range(n_rec):
        na, nb = int(no_all[r]), int(no_all[r + 1])
        tn = np.asarray(tn_all[tok_off[r]:tok_off[r + 1]], dtype=np.int64)
        par = np.asarray(par_all[na:nb], dtype=np.int64)
        fld = np.asarray(fld_all[na:nb])
        dep = np.asarray(dep_all[na:nb], dtype=np.int64)
        assert par[0] == -1 and dep[0] == 0, r
        A, B, D, k = record_coords(par, fld, dep, tn)
        n_stmt.append(k)
        has = A >= 0
        tot_tok += len(tn)
        absent_tok += int((~has).sum())
        if has.any():
            max_a = max(max_a, int(A[has].max()))
            b_vals.append(B[has])
        tw = np.where(has, B, ABSENT).astype(np.int16)
        # stmtgrid: one node per statement that has tokens, path [A]
        units = np.unique(A[has])
        g_np = np.full((len(units), levels), ABSENT, dtype=np.int16)
        g_np[:, 0] = units
        g_tn = np.full(len(tn), ABSENT, dtype=np.int32)
        g_tn[has] = np.searchsorted(units, A[has])
        # stmtgriddepth: one node per distinct (A, depth), first-appearance order
        pairs = list(dict.fromkeys(zip(A[has].tolist(), D[has].tolist())))
        pid = {p: i for i, p in enumerate(pairs)}
        gd_np = np.full((len(pairs), levels), ABSENT, dtype=np.int16)
        if pairs:
            gd_np[:, 0] = [p[0] for p in pairs]
            gd_np[:, 1] = [p[1] for p in pairs]
        gd_tn = np.full(len(tn), ABSENT, dtype=np.int32)
        gd_tn[has] = [pid[p] for p in zip(A[has].tolist(), D[has].tolist())]
        for name, (t_n, n_p) in (("stmtgrid", (g_tn, g_np)), ("stmtgriddepth", (gd_tn, gd_np))):
            w = outs[name]
            w["tn"].append(t_n); w["np"].append(n_p); w["tw"].append(tw)
            w["off"].append(w["off"][-1] + len(n_p))
    for name, w in outs.items():
        for k in ("tn", "np", "tw"):
            w[k].close()
        np.save(os.path.join(w["dir"], "node_off.npy"), np.array(w["off"], dtype=np.int64))
    ns = np.asarray(n_stmt)
    bv = np.concatenate(b_vals) if b_vals else np.zeros(1, int)
    stats = dict(records=n_rec, statements_per_record_p50=float(np.percentile(ns, 50)),
                 statements_per_record_p99=float(np.percentile(ns, 99)), statements_per_record_max=int(ns.max()),
                 max_statement_ordinal=max_a, records_over_level_wrap=int((ns > MAX_LEVEL).sum()),
                 within_statement_p50=float(np.percentile(bv, 50)), within_statement_p99=float(np.percentile(bv, 99)),
                 within_statement_max=int(bv.max()), tokens_absent_frac=absent_tok / max(tot_tok, 1))
    for name, w in outs.items():
        man = dict(derived_from=side, source_sidecar=ss, render="python_" + name, levels=levels,
                   absent=ABSENT, statement_rule=("field in {body, orelse, finalbody} and parent is a container "
                                                  "(Module root, a statement, ExceptHandler, match_case)"),
                   node_rule=("one node per statement, path [statement ordinal]" if name == "stmtgrid" else
                              "one node per distinct (statement ordinal, AST depth of the causal node), "
                              "path [statement ordinal, depth]"),
                   within_rule="tok_within = token offset from its innermost statement's first token; -1 if none",
                   arm="rope_all: sequence 16 planes; level 0 = statement ordinal (2 planes)"
                       + ("; level 1 = AST depth (2 planes)" if name == "stmtgriddepth" else "")
                       + "; token planes (4) = position within the statement; other levels ABSENT",
                   **stats)
        json.dump(man, open(os.path.join(w["dir"], "manifest.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    if stats["records_over_level_wrap"]:
        raise SystemExit("statement ordinals exceed the level ladder's wrap")
    if stats["within_statement_max"] >= MAX_WITHIN:
        raise SystemExit("within-statement offsets exceed the token ladder's wrap")


if __name__ == "__main__":
    main()
