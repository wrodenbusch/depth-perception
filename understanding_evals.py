#!/usr/bin/env python
"""Structure-sensitive "understanding" tests on finished checkpoints. No training.

    # 1. score: one pass per finished run, per-token NLL + argmax-correct on held-out records
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python understanding_evals.py score --setting py --shard 0/3
    # 2. analyse: category masks from the pack, paired per-seed comparison vs sequence only
    .venv/bin/python understanding_evals.py analyse --setting py

WHY. Every number in the paper is mean validation loss over all tokens. A structural position
signal could matter far more on the few tokens whose prediction depends on structure -- closing a
subtree, reusing a variable, finding the cell above -- than its average-loss gain shows, because
those tokens are a small share of the stream. These tests split the SAME held-out loss by what
kind of token is being predicted, so each one asks: "on the tokens where structure should matter,
is the structured model's advantage larger than its average advantage?"

WHAT EVERY TEST SHARES.
  * Test split only, the first N records, batched exactly as ``eval_test.py`` (whole records,
    EOT-padded inputs, -1 targets ignored), bf16 autocast, no compile.
  * The FINAL checkpoint (``ckpt.pt``, refused unless its step is the run's planned last step).
  * Per configuration, the per-category mean NLL is compared with the SAME SEED's sequence-only
    run on the SAME tokens: paired difference per seed, seeds won, paired t (paired.py's verified
    t distribution). Negative = the configuration predicts those tokens better.
  * Categories are computed once per setting from the base pack's tokens and tree sidecar. Arms
    that read a derived pack (Python splice/anchor/toplevel) have byte-identical token streams;
    that is asserted, not assumed.
  * "Beyond average" = per seed, (category Delta) - (all-token Delta). If structure helps a
    category more than average, this is negative and the paired t says whether that is reliable.

THE TESTS, per setting, and what each does NOT measure.

Python (compact prefix AST serialisation: node-type tags carry child counts, identifiers are
renamed x0, x1, ... in binding order; no brackets, no indentation):
  py.ascend      target is a node that starts at a SHALLOWER depth than the previous token: the
                 previous subtree has just closed. Knowing where you are requires having counted
                 the children announced by earlier tags -- the closest thing this format has to
                 a closing bracket or dedent.  [Not measured: whether the model "knows" the tree;
                 only how well it predicts these tokens.]
  py.ascend2     the same, closing two or more levels at once (harder bookkeeping).
  py.descend     target starts a child of the previous token's node (deeper): control category.
  py.ident_reuse target is an identifier x<k> that already occurred in the record -- a USE of a
                 bound name; predicting it means picking the right in-scope variable.
  py.ident_new   target is the first occurrence of x<k> -- a new binding. Under binding-order
                 renaming the name itself is nearly forced (the next index), so this mostly tests
                 counting, not understanding.  [Not measured: scope correctness as such -- a
                 reused name might still be the wrong variable; we only score likelihood of the
                 one actually written.]

Spreadsheets (values render: cell contents, TAB between cells, NEWLINE between rows; the tree is
sheet -> row -> column -> cell content -> formula nodes; separators carry no coordinate):
  sh.ref         target is part of a cell address inside a formula (a reference token).
  sh.formula     target is any token of a formula (deeper than plain cell content).
  sh.value       target is a token of a plain value cell: control.
  sh.newline     target is the row-ending NEWLINE: predicting it needs the row's column count.
  sh.cell_start  target is the first token of a cell.
  sh.above_far / sh.above_near   cell-first tokens whose cell DIRECTLY ABOVE (same sheet, same
                 column, nearest earlier row) begins >= 256 / < 64 tokens back. Near in the grid,
                 far (or near) in the sequence.
  sh.copy_above_far              the subset of above_far where the cell's first token equals the
                 first token of the cell above: copying down a column across a long gap -- the
                 case grid coordinates should make easy and sequence position makes hard.
  [Not measured: numeric correctness of values; GPT-2's tokeniser fragments numbers.]

Web text (visible text only; each text node is a line; the tree has no tokens of its own):
  web.node_start target is the first token of a new text node (a new paragraph/line/item).
  web.node_end   target is the NEWLINE that ends a text node: predicting where a block ends.
  web.cont       target is inside a text node: control.
  web.rep_same / web.rep_cross   target is a repeat of a non-frequent token (not among the 500
                 most frequent in the eval set) whose most recent earlier occurrence lies 64-1023
                 tokens back, either under the SAME parent element (rep_same) or in a branch that
                 splits from the target's at the top two levels (rep_cross). Same distance band
                 for both, so sequence distance is controlled; the difference is tree locality.
  [Not measured: comprehension in any task sense; these are all likelihoods.]

DELIBERATELY NOT RUN: structural minimal pairs (shuffle sections, swap statements, corrupt
nesting). Perturbing the text while keeping consistent coordinates is ill-defined -- either the
coordinates follow the perturbed text (then the structured model is told the new structure and
its "surprise" is not comparable to the sequence model's) or they do not (then the structured
model is fed contradictory input it never saw in training). Either way the score measures the
input contract, not understanding.
"""
from __future__ import annotations
from topos_paths import PACKS, RUNS, SOURCES  # noqa: E402

import argparse
import json
import math
import os
import re
import statistics as st
import sys
from contextlib import nullcontext

import numpy as np

ROOT = f"{RUNS}"
OUT = f"{RUNS}/analysis/understanding"
LEVELS = 22

SETTINGS = {
    "py": dict(root="py", final=11200, base_pack=f"{PACKS}/py_ast_423k", records=8192,
               batch=16, seq="rope_seq",
               arms=["rope_seq", "rope_leaf", "rope_both", "depth", "rel", "full_rope", "toplevel",
                     "splice", "anchor", "rope_graph", "none", "drank"]),
    "sheets": dict(root="sheets", final=8240, base_pack=f"{PACKS}/sheet_values_v2",
                   records=1024, batch=2, seq="seq_v2",
                   arms=["seq_v2", "both_v2", "depth_v2", "rel_v2", "ref_anchor", "full_rope_v2",
                         "toplevel_v2", "layer_split_v2", "grid_v2"]),
    "webc": dict(root="webc", final=9375, base_pack=f"{PACKS}/web_whole_b2048", records=1024,
                 batch=4, seq="seq", arms=["seq", "both", "depth", "rel", "full_rope", "drank"]),
}

PRETTY = {
    "rope_seq": "sequence only", "seq_v2": "sequence only", "seq": "sequence only",
    "rope_both": "sequence + tree", "both_v2": "sequence + tree", "both": "sequence + tree",
    "depth": "sequence + depth", "depth_v2": "sequence + depth",
    "rel": "sequence + tree + depth", "rel_v2": "sequence + tree + depth",
    "full_rope": "full RoPE", "full_rope_v2": "full RoPE",
    "rope_leaf": "tree only", "rope_graph": "tree path only (no within-node)",
    "none": "no position", "toplevel": "top-level-only control", "toplevel_v2": "top-level-only control",
    "splice": "reference rotation (prefix splice)", "anchor": "reference rotation (whole path)",
    "ref_anchor": "reference rotation", "layer_split_v2": "layer split",
    "grid_v2": "sequence + grid (sheet, row, column)",
    "drank": "sequence + tree grid (depth, sibling rank)",
}


def runs_of(setting: str):
    """[(arm, seed, run_dir)] for every finished run of the setting's arms."""
    cfg = SETTINGS[setting]
    out = []
    for arm in cfg["arms"]:
        base = os.path.join(ROOT, cfg["root"])
        for d in sorted(os.listdir(base)):
            m = re.fullmatch(re.escape(arm) + r"_s(\d+)", d)
            if m and os.path.exists(os.path.join(base, d, "ckpt.pt")):
                out.append((arm, int(m.group(1)), os.path.join(base, d)))
    return out


# ------------------------------------------------------------------ scoring

def score_run(setting: str, arm: str, seed: int, run_dir: str, device: str = "cuda") -> str:
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import GPT, GPTConfig
    from topos_data import PackedWindows
    from eval_test import sidecar_for
    from rope_pos import SLOT_MODES, REF_COORD_MODES, REF_ANCHOR_MODES

    cfg = SETTINGS[setting]
    out = os.path.join(OUT, setting, f"{arm}_s{seed}.npz")
    if os.path.exists(out):
        return out
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    if int(ck.get("iter_num", -1)) != cfg["final"]:
        raise SystemExit(f"{run_dir}: ckpt.pt is at step {ck.get('iter_num')}, not the final "
                         f"{cfg['final']} -- refusing to score an unfinished run")
    margs = ck["model_args"]
    pack = ck["config"]["pack"]
    mode = margs["pos_mode"]
    model = GPT(GPTConfig(**margs))
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    pw = PackedWindows(pack, need_tree=False,
                       sidecar=sidecar_for(mode, margs.get("rope_layer_plan", ""), margs.get("depth_emb", False)),
                       need_slot=mode in SLOT_MODES, need_ref=mode in REF_COORD_MODES,
                       need_ref_sheet=mode in REF_ANCHOR_MODES)
    lo, hi = pw.splits["test"]
    hi = min(hi, lo + cfg["records"])
    dev = torch.device(device)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else nullcontext()
    nlls, hits = [], []
    with torch.no_grad():
        for start in range(lo, hi, cfg["batch"]):
            ix = list(range(start, min(start + cfg["batch"], hi)))
            X, Y, POS = pw.get_batch_records(ix, dev)
            with ctx:
                logits, _ = model(X, Y, pos=POS)
            logits = logits.float()
            lp = torch.log_softmax(logits, dim=-1)
            yy = Y.clamp(min=0)
            nll = -lp.gather(-1, yy[..., None])[..., 0]
            hit = logits.argmax(-1) == yy
            for i, r in enumerate(ix):
                t = int(pw.tok_off[r + 1] - pw.tok_off[r]) - 1
                nlls.append(nll[i, :t].cpu().numpy().astype(np.float32))
                hits.append(hit[i, :t].cpu().numpy())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, nll=np.concatenate(nlls), hit=np.concatenate(hits), pack=pack, mode=mode,
             records=np.array([lo, hi]))
    return out


# ------------------------------------------------------------------ categories

def _gpt2_ids():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode("\n")[0], enc.encode("\t")[0]


def _py_ident_ids():
    """token id -> identifier string, for tokens that are exactly ' x<k>' in the Python BPE."""
    t = json.load(open(f"{SOURCES}/tokenizers/py_wholefile_astcompact_bpe_32k/"
                       "tokenizer.json"))
    out = {}
    for s, i in t["model"]["vocab"].items():
        m = re.fullmatch("Ġx(\\d+)", s)          # byte-level BPE: Ġ is a leading space
        if m:
            out[i] = "x" + m.group(1)
    return out


def categories(setting: str):
    """Boolean masks over the concatenated TARGET positions of the eval records."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from topos_data import PackedWindows
    cfg = SETTINGS[setting]
    pw = PackedWindows(cfg["base_pack"], need_tree=False, sidecar="path")
    lo, hi = pw.splits["test"]
    hi = min(hi, lo + cfg["records"])
    cats: dict = {}
    tokens_all = []

    def add(name, arr):
        cats.setdefault(name, []).append(arr)

    if setting == "py":
        ident = _py_ident_ids()
    if setting == "sheets":
        refr = np.load(cfg["base_pack"] + "_dompath/tok_ref_row.npy", mmap_mode="r")
    if setting in ("sheets", "webc"):
        NL, TAB = _gpt2_ids()
    if setting == "webc":
        # frequency over the eval records, to exclude function words from the repeat test
        cnt = np.bincount(np.asarray(pw.tokens[pw.tok_off[lo]:pw.tok_off[hi]], dtype=np.int64),
                          minlength=pw.vocab_size)
        frequent = set(np.argsort(-cnt)[:500].tolist())

    for r in range(lo, hi):
        toks, pos = pw.record(r)
        path = pos["path"][:, :LEVELS]
        within = pos["path"][:, LEVELS] if pos["path"].shape[1] > LEVELS else None
        depth = (path >= 0).sum(1)
        a = int(pw.tok_off[r])
        n = len(toks)
        tgt = np.arange(1, n)                 # target positions; prediction made at tgt-1
        tokens_all.append(toks[1:])
        add("all", np.ones(n - 1, bool))
        if setting == "py":
            start = within[tgt] == 0
            add("py.ascend", start & (depth[tgt] < depth[tgt - 1]) & (depth[tgt] > 0))
            add("py.ascend2", start & (depth[tgt] <= depth[tgt - 1] - 2) & (depth[tgt] > 0))
            add("py.descend", start & (depth[tgt] > depth[tgt - 1]))
            seen, reuse, new = set(), np.zeros(n - 1, bool), np.zeros(n - 1, bool)
            for k, i in enumerate(tgt):
                name = ident.get(int(toks[i]))
                if name is None:
                    continue
                (reuse if name in seen else new)[k] = True
                seen.add(name)
            if int(toks[0]) in ident:
                pass                              # position 0 is EOT; never an identifier
            add("py.ident_reuse", reuse)
            add("py.ident_new", new)
        elif setting == "sheets":
            rr = np.asarray(refr[a:a + n])
            inside = depth > 0
            add("sh.ref", rr[tgt] >= 0)
            add("sh.formula", inside[tgt] & (depth[tgt] > 4))
            add("sh.value", inside[tgt] & (depth[tgt] == 4))
            add("sh.newline", (~inside[tgt]) & (toks[tgt] == NL))
            cstart = inside & (within == 0)
            add("sh.cell_start", cstart[tgt])
            # the cell directly above: same sheet (L0), same column (L2), nearest earlier row (L1)
            first_tok, last_pos = {}, {}
            far = np.zeros(n - 1, bool); near = np.zeros(n - 1, bool); copyfar = np.zeros(n - 1, bool)
            for i in range(1, n):
                if not cstart[i] or depth[i] < 3:
                    continue
                key = (int(path[i, 0]), int(path[i, 2]))
                if key in last_pos:
                    gap = i - last_pos[key]
                    if gap >= 256:
                        far[i - 1] = True
                        copyfar[i - 1] = first_tok[key] == int(toks[i])
                    elif gap < 64:
                        near[i - 1] = True
                last_pos[key] = i
                first_tok[key] = int(toks[i])
            add("sh.above_far", far)
            add("sh.above_near", near)
            add("sh.copy_above_far", copyfar)
        else:   # webc
            inside = depth > 0
            add("web.node_start", inside[tgt] & (within[tgt] == 0))
            add("web.node_end", inside[tgt] & (toks[tgt] == NL))
            add("web.cont", inside[tgt] & (within[tgt] > 0) & (toks[tgt] != NL))
            last = {}
            same = np.zeros(n - 1, bool); cross = np.zeros(n - 1, bool)
            for i in range(1, n):
                v = int(toks[i])
                if inside[i] and v not in frequent and v != NL and v in last:
                    j = last[v]
                    gap = i - j
                    if 64 <= gap < 1024 and inside[j]:
                        di, dj = int(depth[i]), int(depth[j])
                        # parent element of a text node = its path minus the last (own-rank) level
                        if di == dj and np.array_equal(path[i, :di - 1], path[j, :dj - 1]):
                            same[i - 1] = True
                        elif min(di, dj) >= 2 and not np.array_equal(path[i, :2], path[j, :2]):
                            cross[i - 1] = True
                last[v] = i
            add("web.rep_same", same)
            add("web.rep_cross", cross)
    masks = {k: np.concatenate(v) for k, v in cats.items()}
    return masks, np.concatenate(tokens_all)


# ------------------------------------------------------------------ statistics

def paired(diffs):
    from paired import p_two_sided
    n = len(diffs)
    mean = st.mean(diffs)
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    t = mean / se if se else float("inf")
    return mean, t, (p_two_sided(t, n - 1) if se else 0.0)


def analyse(setting: str):
    cfg = SETTINGS[setting]
    masks, _ = categories(setting)
    ntok = len(masks["all"])
    res = {}
    for arm, seed, _ in runs_of(setting):
        f = os.path.join(OUT, setting, f"{arm}_s{seed}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        nll, hit = z["nll"], z["hit"]
        if len(nll) != ntok:
            raise SystemExit(f"{f}: {len(nll)} target positions, categories have {ntok} -- token "
                             "streams differ; refusing to compare")
        res[(arm, seed)] = {c: (float(nll[m].mean()), float(hit[m].mean()), int(m.sum()))
                            for c, m in masks.items() if m.any()}
    seq = cfg["seq"]
    rows = []
    for arm in cfg["arms"]:
        if arm == seq:
            continue
        seeds = sorted(s for (a, s) in res if a == arm and (seq, s) in res)
        if not seeds:
            continue
        for c in masks:
            if c not in res[(arm, seeds[0])]:
                continue
            d = [res[(arm, s)][c][0] - res[(seq, s)][c][0] for s in seeds]
            dacc = [res[(arm, s)][c][1] - res[(seq, s)][c][1] for s in seeds]
            beyond = [res[(arm, s)][c][0] - res[(seq, s)][c][0]
                      - (res[(arm, s)]["all"][0] - res[(seq, s)]["all"][0]) for s in seeds]
            m, t, p = paired(d)
            bm, bt, bp = paired(beyond)
            rows.append(dict(setting=setting, arm=arm, name=PRETTY.get(arm, arm), cat=c,
                             n_seeds=len(seeds), n_tok=res[(arm, seeds[0])][c][2],
                             seq_nll=st.mean(res[(seq, s)][c][0] for s in seeds),
                             delta=m, won=sum(x < 0 for x in d), p=p,
                             d_acc=st.mean(dacc), beyond=bm, beyond_p=bp, seeds=seeds,
                             diffs=d, seq_nll_seed=[res[(seq, s)][c][0] for s in seeds],
                             rel=(m / st.mean(res[(seq, s)][c][0] for s in seeds))))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"{setting}_results.json"), "w"), indent=1)
    # category sizes, and depth stats that bear on "why does depth not help on sheets"
    sizes = {c: int(m.sum()) for c, m in masks.items()}
    json.dump(sizes, open(os.path.join(OUT, f"{setting}_category_sizes.json"), "w"), indent=1)
    lines = [f"# {setting}: understanding tests (test split, {cfg['records']} records, "
             f"{ntok:,} target tokens)", "",
             "Δ = configuration − sequence only, mean NLL (nats) on the category, paired per seed; "
             "negative = better. 'beyond' = category Δ − all-token Δ (negative = helps this category "
             "MORE than average). rel = Δ / sequence-only NLL on the category.", ""]
    for arm in cfg["arms"]:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        lines.append(f"## {PRETTY.get(arm, arm)} vs sequence only (n={sub[0]['n_seeds']})")
        lines.append("| category | tokens | seq NLL | Δ | rel | won | p | Δacc | beyond | p(beyond) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            lines.append(f"| {r['cat']} | {r['n_tok']:,} | {r['seq_nll']:.3f} | {r['delta']:+.4f} | "
                         f"{100*r['rel']:+.1f}% | {r['won']}/{r['n_seeds']} | {r['p']:.3g} | "
                         f"{100*r['d_acc']:+.2f}pt | {r['beyond']:+.4f} | {r['beyond_p']:.3g} |")
        lines.append("")
    open(os.path.join(OUT, f"{setting}_results.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("cmd", choices=("score", "analyse", "list"))
    ap.add_argument("--setting", required=True, choices=tuple(SETTINGS))
    ap.add_argument("--shard", default="0/1", help="i/n: score every n-th run starting at i")
    a = ap.parse_args()
    if a.cmd == "list":
        for r in runs_of(a.setting):
            print(*r)
    elif a.cmd == "score":
        i, n = map(int, a.shard.split("/"))
        todo = runs_of(a.setting)[i::n]
        for arm, seed, d in todo:
            print(f"[score] {a.setting} {arm}_s{seed}", flush=True)
            print("   ->", score_run(a.setting, arm, seed, d), flush=True)
    else:
        analyse(a.setting)


if __name__ == "__main__":
    main()
