#!/usr/bin/env python
"""Per-token category and tree depth for the spreadsheet pack's val/test records.

    .venv/bin/python analysis/sheet_token_categories.py --out $TOPOS_RUNS/analysis/sheets_depth

WHY. Sequence + depth barely beat sequence only on spreadsheets (-0.0025, 7/10). The suspicion is
structural: the sheet, row and cell nodes emit no tokens of their own, so every cell-value token
sits at the same depth and only formula tokens vary. This labels every token so the loss can be
split by what is being predicted.

Depth = number of non-ABSENT levels in the token's node path, exactly as ``rope_pos._rel_angles``
computes it (0 for a token outside the tree). In this pack a cell's own tokens sit at depth 4 and
formula structure is deeper. TAB and NEWLINE separators are outside the tree -- and so are some
characters INSIDE formulas that no node owns (',' ':' ')' and the like), which therefore share
depth 0 with the row separators.

Categories (of the token itself; the loss script labels each position by its TARGET token):
  sep           outside the tree and whitespace/EOT (TAB between cells, NEWLINE between rows)
  formula_punct outside the tree but not whitespace -- formula characters no node owns
  value_num     depth 4, not a formula start, numeric
  value_text    depth 4, anything else that is not a formula start
  formula_eq    the '=' that opens a formula (depth 4, next token deeper)
  ref           a reference address inside a formula (any tok_ref_* axis set)
  func          alphabetic token inside a formula that is not a reference
  formula_num   numeric literal inside a formula
  formula_op    operator / other punctuation inside a formula
  shallow       depth 1-3: defined-names table and other above-cell nodes
"""
from __future__ import annotations
from topos_paths import PACKS  # noqa: E402

import argparse
import json
import os
import re

import numpy as np
import tiktoken

P = f"{PACKS}/sheet_values_v2"
S = P + "_dompath"
CATS = ["sep", "formula_punct", "value_num", "value_text", "formula_eq", "ref", "func",
        "formula_num", "formula_op", "shallow"]
FORMULA = ("formula_eq", "ref", "func", "formula_num", "formula_op", "formula_punct")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(P + "/meta.json"))
    ntr, nv, nte = meta["n_train"], meta["n_val"], meta["n_test"]
    off = np.load(P + "/tok_off.npy")
    toks = np.load(P + "/tokens.npy", mmap_mode="r")
    tn_all = np.load(S + "/tok_node.npy", mmap_mode="r")
    noff = np.load(S + "/node_off.npy")
    npth = np.load(S + "/node_path.npy", mmap_mode="r")
    refs = [np.load(S + f"/tok_ref_{x}.npy", mmap_mode="r") for x in ("sheet", "row", "col")]
    enc = tiktoken.get_encoding("gpt2")
    text = [enc.decode([i]) if i < enc.n_vocab else "" for i in range(meta["vocab_size"])]
    is_ws = np.array([t.strip() == "" for t in text])
    is_eot = np.zeros(len(text), bool)
    is_eot[meta["eot"]] = True
    is_num = np.array([bool(re.fullmatch(r"\s*[0-9.,\-]+", t)) for t in text])
    is_alpha = np.array([bool(re.fullmatch(r"\s*[A-Za-z_][A-Za-z_0-9.]*", t)) for t in text])
    is_eq = np.array([t.strip() == "=" for t in text])

    r0, r1 = ntr, ntr + nv + nte                    # val + test records
    A, B = int(off[r0]), int(off[r1])
    tok = np.asarray(toks[A:B], dtype=np.int64)
    tn = np.asarray(tn_all[A:B], dtype=np.int64)
    rec = np.repeat(np.arange(r0, r1), np.diff(off[r0:r1 + 1]))
    gidx = noff[rec] + np.maximum(tn, 0)
    rows = np.asarray(npth[gidx])
    depth = np.where(tn >= 0, (rows >= 0).sum(1), 0)
    has_ref = np.zeros(len(tok), bool)
    for r in refs:
        has_ref |= np.asarray(r[A:B]) >= 0

    nxt_depth = np.r_[depth[1:], 0]
    C = {c: i for i, c in enumerate(CATS)}
    cat = np.full(len(tok), -1, np.int8)
    outside = tn < 0
    blank = is_ws[tok] | is_eot[tok]
    cat[outside & blank] = C["sep"]
    cat[outside & ~blank] = C["formula_punct"]
    cat[(~outside) & (depth < 4)] = C["shallow"]
    d4 = (~outside) & (depth == 4)
    eq = d4 & is_eq[tok] & (nxt_depth > 4)
    cat[eq] = C["formula_eq"]
    cat[d4 & ~eq & is_num[tok]] = C["value_num"]
    cat[d4 & ~eq & ~is_num[tok]] = C["value_text"]
    deep = (~outside) & (depth > 4)
    cat[deep & has_ref] = C["ref"]
    cat[deep & ~has_ref & is_alpha[tok]] = C["func"]
    cat[deep & ~has_ref & ~is_alpha[tok] & is_num[tok]] = C["formula_num"]
    cat[deep & ~has_ref & ~is_alpha[tok] & ~is_num[tok]] = C["formula_op"]
    assert (cat >= 0).all()

    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "token_labels.npz"), start=A, end=B, r0=r0, r1=r1,
                        cat=cat, depth=depth.astype(np.int8), cats=np.array(CATS))

    n = len(tok)
    rep = {"tokens": int(n), "records": int(r1 - r0)}
    rep["category_share"] = {c: round(float((cat == C[c]).mean()), 4) for c in CATS}
    vals, cnt = np.unique(depth, return_counts=True)
    rep["depth_share"] = {int(v): round(float(k / n), 4) for v, k in zip(vals, cnt)}
    pd = cnt / n
    rep["depth_entropy_bits"] = float(-(pd * np.log2(pd)).sum())
    rep["share_depth0"] = float((depth == 0).mean())
    rep["share_depth4_cell_level"] = float((depth == 4).mean())
    rep["share_deeper_than_cell"] = float((depth > 4).mean())
    formula = np.isin(cat, [C[c] for c in FORMULA])
    rep["share_formula_tokens"] = float(formula.mean())
    rep["share_records_with_formula"] = float(np.mean(
        [formula[off[r] - A:off[r + 1] - A].any() for r in range(r0, r1)]))
    in_tree_nf = (depth > 0) & ~formula
    v, k = np.unique(depth[in_tree_nf], return_counts=True)
    rep["depth_of_in_tree_non_formula_tokens"] = {int(x): round(float(y / in_tree_nf.sum()), 4)
                                                  for x, y in zip(v, k)}
    rep["depth_by_category"] = {}
    for c in CATS:
        m = cat == C[c]
        if m.any():
            v, k = np.unique(depth[m], return_counts=True)
            rep["depth_by_category"][c] = {int(x): int(y) for x, y in zip(v, k)}
    json.dump(rep, open(os.path.join(a.out, "token_stats.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
