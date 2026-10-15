#!/usr/bin/env python
"""Build the interactive appendix: one self-contained HTML page, regenerated from the run logs.

    .venv/bin/python make_appendix.py                 # writes docs/index.html (the GitHub Pages site)
    .venv/bin/python make_appendix.py --publish       # ... and copies it to $TOPOS_SITE/appendix

WHAT THIS IS. The short paper reports the headline results; this page carries everything else --
the full setup, every configuration, every per-seed number, the negative results -- written for a
reader who has never seen the project. It is a companion to the code, so it ships with it.

EVERY RESULT NUMBER IS COMPUTED HERE, at build time, from ``train.jsonl`` / ``heldout.jsonl`` via
``paired.read_arm``. Prose quotes those values through format fields rather than typing them, so
the page cannot drift from the logs. Two kinds of number are NOT recomputed and are written as
constants in ``CORPUS_CONST`` with their provenance: corpus measurements that need code outside
this repository (surface-form overlap of references) or a long run of the test suite (the parent
reconstruction check). Corpus sizes, node counts and exclusion counts are read from the pack
manifests; tree entropies and interior-node fractions are measured on a fixed sample of records.

RAGGED COMPARISONS ARE REFUSED, as in ``paired.py``: a contrast uses only seeds whose runs on BOTH
sides reached the setting's planned final step, and asserts that every paired run stopped at that
step. Runs still training are reported as "in progress" from the logs at build time, never
hard-coded.

SELF-CONTAINED BY DESIGN. Inline CSS, inline JS, inline SVG, system fonts, no CDN: it must open
from disk and from a static host. Charts follow the dataviz method: one validated categorical
palette (slots blue / orange / aqua / violet validate all-pairs in light mode; aqua is below 3:1
on the surface, so every chart carries direct labels and a table view), thin marks, a hover and
keyboard-focus tooltip on every mark, and a table-view twin under every chart.
"""
from __future__ import annotations
from topos_paths import PACKS as PACKS_ROOT, RUNS, SITE  # noqa: E402

import argparse
import collections
import datetime as _dt
import html
import json
import math
import os
import re
import shutil
import statistics as st

import numpy as np

from paired import p_two_sided, read_arm

PAPER_TITLE = "Depth Perception for Transformers: Telling Models How Deeply Each Token Is Nested"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = f"{RUNS}"
OUT_DIR = os.path.join(HERE, "docs")   # served by GitHub Pages from main /docs
PUBLISH_DIR = f"{SITE}/appendix"

# ------------------------------------------------------------------------------------ settings

#: root directory, planned final step, display name, config file of the sequence-only baseline
SETTINGS = {
    "py": dict(name="Python", root="py", final=11200, cfg="config/py_rope_seq.py"),
    "sheets": dict(name="Spreadsheets", root="sheets", final=8240, cfg="config/sheet_v2_seq.py"),
    "webs": dict(name="Web pages", root="webc", final=9375, cfg="config/webc_seq.py"),
    "webl": dict(name="Larger web run", root="web7", final=537000, cfg="config/web2_rope_seq.py",
                 jobs="web7/jobs.json", job="rope_seq_p1"),
    "chat": dict(name="Chat", root="chat", final=13100, cfg="config/chat_seq.py"),
}

PACKS = {
    "py": (f"{PACKS_ROOT}/py_ast_423k", f"{PACKS_ROOT}/py_ast_423k_dompath"),
    "sheets": (f"{PACKS_ROOT}/sheet_values_v2", f"{PACKS_ROOT}/sheet_values_v2_dompath"),
    "sheets_addr": (f"{PACKS_ROOT}/sheet_vanilla_v2", f"{PACKS_ROOT}/sheet_vanilla_v2_dompath"),
    "webl": (f"{PACKS_ROOT}/web_text_full_v3", f"{PACKS_ROOT}/web_text_full_v3_dompath"),
    "webs": (f"{PACKS_ROOT}/web_whole_b2048", f"{PACKS_ROOT}/web_whole_b2048_dompath"),
}

#: Corpus measurements not recomputed at build time, each with where it comes from.
CORPUS_CONST = {
    # parent reconstruction from (order, depth) on the Python tree, checked by the repository's
    # test suite (test_depth_and_order_reconstruct_the_tree) over 4,000 records
    "py_parents_checked": 558271,
    # share of a reference's tokens that also occur among its referent's tokens, measured on the
    # packs with the corpus pipeline's own tokenisation
    "overlap_py": 95.8, "overlap_sheets_values": 2.4, "overlap_sheets_addr": 77.2,
    # Python whole-path reference variant: share of substituted paths that were shortened, and
    # the mean number of levels lost (definitions sit shallower than uses: median depth 3 vs 5)
    "anchor_truncated_pct": 91.7, "anchor_levels_lost": 3.13,
    # Python reference splice: share of all tokens whose path is substituted
    "splice_tokens_pct": 20.6,
    # symbol-identity variant: share of tokens with no symbol, and its learned presence offset
    "slot_absent_pct": 57, "slot_theta_rad": 1.23,
    # spreadsheet references that cross to another sheet (sample of 1,500 workbooks)
    "sheets_cross_sheet_ref_pct": 17.25,
}


def load_config(path: str, overrides: str = "") -> dict:
    """Execute a config file the way train.py does, then apply ``--key=value`` overrides."""
    g: dict = {}
    exec(open(os.path.join(HERE, path)).read(), g)
    for m in re.finditer(r"--(\w+)=(\S+)", overrides):
        k, v = m.group(1), m.group(2)
        try:
            g[k] = json.loads(v.lower() if v in ("True", "False") else v)
        except json.JSONDecodeError:
            g[k] = v
    return {k: v for k, v in g.items() if not k.startswith("_")}


def setting_config(key: str) -> dict:
    s = SETTINGS[key]
    over = ""
    if "jobs" in s:
        for j in json.load(open(os.path.join(RUNS, s["jobs"]))):
            if j["name"] == s["job"]:
                over = j["cmd"]
    return load_config(s["cfg"], over)


# ------------------------------------------------------------------------------------ statistics

def t_crit(df: int) -> float:
    """Two-sided 95% critical value, by bisection on paired.py's verified t distribution."""
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_two_sided(mid, df) > 0.05 else (lo, mid)
    return (lo + hi) / 2


def arm_runs(root: str, arm: str) -> dict:
    return read_arm(os.path.join(RUNS, root), arm, "val_nats_tok")


def contrast(setting: str, a: str, b: str) -> dict:
    """Paired comparison of configuration ``a`` against baseline ``b`` at the planned final step.

    Seeds whose run on either side has not reached the final step are left out and counted as
    in progress. Returns the per-seed differences and the paired statistics.
    """
    S = SETTINGS[setting]
    A, B = arm_runs(S["root"], a), arm_runs(S["root"], b)
    fin = S["final"]
    doneA = {s for s, d in A.items() if d[3] >= fin}
    doneB = {s for s, d in B.items() if d[3] >= fin}
    shared = sorted(doneA & doneB)
    # in progress = a seed with a run on BOTH sides, at least one of them not yet finished. A
    # finished seed that simply has no counterpart (Python's sequence-only has 20 seeds, most
    # configurations 5) is unmatched, not pending.
    pending = sorted(s for s in set(A) & set(B) if s not in shared)
    for s in shared:                       # the ragged-comparison guard, as in paired.py
        assert A[s][3] == fin and B[s][3] == fin, f"{setting} {a}/{b} seed {s} not at {fin}"
    diffs = [(s, A[s][0] - B[s][0]) for s in shared]
    r = dict(setting=setting, a=a, b=b, n=len(diffs), diffs=diffs, pending=pending,
             a_mean=st.mean(A[s][0] for s in shared) if shared else None,
             b_mean=st.mean(B[s][0] for s in shared) if shared else None,
             a_seeds=len(A), done_a=len(doneA))
    if not diffs:
        r.update(mean=None)
        return r
    d = [x for _, x in diffs]
    m = st.mean(d)
    r.update(mean=m, won=sum(x < 0 for x in d))
    if len(d) > 1:
        sd = st.stdev(d)
        se = sd / math.sqrt(len(d))
        t = m / se if se else float("inf")
        h = t_crit(len(d) - 1) * se
        r.update(sd=sd, se=se, t=t, p=p_two_sided(t, len(d) - 1), lo=m - h, hi=m + h)
    else:
        r.update(sd=None, se=None, t=None, p=None, lo=m, hi=m)
    r["ppl"] = 100 * (1 - math.exp(m))
    return r


def seed_sd(setting: str, arm: str) -> tuple[float, int]:
    S = SETTINGS[setting]
    v = [d[0] for d in arm_runs(S["root"], arm).values() if d[3] >= S["final"]]
    return st.stdev(v), len(v)


# ---------------------------------------------------------------------------- the larger web run, one seed

WEBL_ARMS = {
    "seq": "rope_seq_p1", "both": "rope_both_p1", "leaf": "rope_leaf_p1",
    "layer_split": "rope_layer_split_p1", "head_split": "stopped_rope_head_split_p1_30pct",
    "head_split_2": "stopped_rope_head_split_p2_9pct", "depth": "rope_depth_p1",
    "graph": "diverged_rope_graph_p1_lr6e-4", "none": "diverged_none_p1_lr1.5e-4",
}


def webl_run(arm: str) -> dict:
    d = os.path.join(RUNS, "web7", WEBL_ARMS[arm])
    val, last = {}, 0
    for line in open(os.path.join(d, "train.jsonl")):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        last = max(last, r.get("step", 0))
        v = r.get("val_nats_tok")
        if v is not None and math.isfinite(v) and r["step"] > 0:
            val[r["step"]] = v
    test = None
    hp = os.path.join(d, "heldout.jsonl")
    if os.path.exists(hp):
        for line in open(hp):
            r = json.loads(line)
            if r.get("split") == "test" and r.get("step") == SETTINGS["webl"]["final"]:
                test = (r["nats_tok"], r["records"])
    return dict(val=val, last=last, test=test)


def webl_data() -> dict:
    runs = {k: webl_run(k) for k in WEBL_ARMS}
    seq = runs["seq"]["val"]
    fin = SETTINGS["webl"]["final"]
    gaps = {}
    for k, r in runs.items():
        if k == "seq":
            continue
        # matched milestones only: a gap exists where BOTH runs have a validation score
        gaps[k] = {s: v - seq[s] for s, v in sorted(r["val"].items()) if s in seq}
    out = dict(runs=runs, gaps=gaps, final=fin, interval=fin // 20)
    g = gaps["both"]
    vals = [g[s] for s in sorted(g)]
    succ = [b - a for a, b in zip(vals, vals[1:])]
    out["both_summary"] = dict(n=len(vals), neg=sum(v < 0 for v in vals), mean=st.mean(vals),
                               lo=min(vals), hi=max(vals), last5=vals[-5:],
                               succ_sd=st.stdev(succ) / math.sqrt(2))
    lg = gaps["leaf"]
    out["leaf_summary"] = dict(n=len(lg), pos=sum(v > 0 for v in lg.values()),
                               final=lg.get(fin), first=lg[min(lg)])
    # share of the joint (sequence + tree) margin each split configuration keeps, milestone by milestone
    for k in ("layer_split", "head_split"):
        sh = {s: gaps[k][s] / g[s] for s in gaps[k] if s in g}
        out[k + "_share"] = sh
    return out


# ------------------------------------------------------------------------------ held-out test

def py_test(records: int) -> dict:
    """``rope_leaf`` vs ``rope_seq`` on the Python test split at the final checkpoint."""
    f = {8192: "test_final_8192.jsonl", 2048: "test_final_ckpt.jsonl"}[records]
    rows = [json.loads(l) for l in open(os.path.join(RUNS, "py", f)) if l.strip()]
    by = collections.defaultdict(dict)
    for r in rows:
        assert r["records"] == records and r["step"] == SETTINGS["py"]["final"]
        arm, seed = r["run"].rsplit("_s", 1)
        by[arm][int(seed)] = r["nats_tok"]
    shared = sorted(set(by["rope_leaf"]) & set(by["rope_seq"]))
    d = [by["rope_leaf"][s] - by["rope_seq"][s] for s in shared]
    m, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(len(d))
    return dict(n=len(d), mean=m, sd=sd, se=se, t=m / se, p=p_two_sided(m / se, len(d) - 1),
                won=sum(x < 0 for x in d), seeds=shared, diffs=dict(zip(shared, d)),
                leaf_mean=st.mean(by["rope_leaf"][s] for s in shared),
                records=records)


# ---------------------------------------------------------------------------- corpus statistics

def manifest(path: str) -> dict:
    return json.load(open(path)) if os.path.exists(path) else {}


def corpus_facts() -> dict:
    f = {}
    for k, (pk, sc) in PACKS.items():
        f[k] = dict(meta=manifest(os.path.join(pk, "meta.json")),
                    side=manifest(os.path.join(sc, "manifest.json")))
    py = f["py"]
    py["nodes_per_token"] = py["side"]["n_nodes"] / py["meta"]["total_tokens"]
    for k in ("sheets", "sheets_addr"):
        f[k]["nodes_per_token"] = f[k]["side"]["nodes_stored_per_token"]
    for k in ("webl", "webs"):
        tot = f[k]["meta"]["total_tokens"]
        f[k]["nodes_per_token"] = f[k]["side"]["nodes_total"] / tot
    ws = f["webs"]["meta"]
    agg = collections.Counter()
    for sh in ws.get("shards", []):
        agg.update({k: v for k, v in sh.items() if isinstance(v, int)})
    f["webs"]["shard_totals"] = dict(agg)
    return f


def _entropy(counter: collections.Counter) -> float:
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values())


def tree_stats(n_records: int = 400, seed: int = 0) -> dict:
    """Per corpus, over a fixed random sample of training records, counted PER TOKEN:
    plug-in entropy of depth and of the full path, the fraction of token-carrying nodes that
    have a token-carrying descendant (interior nodes), and sibling-rank percentiles."""
    out = {}
    for k in ("py", "sheets", "webl", "webs"):
        pk, sc = PACKS[k]
        tok_off = np.load(os.path.join(pk, "tok_off.npy"), mmap_mode="r")
        node_off = np.load(os.path.join(sc, "node_off.npy"), mmap_mode="r")
        node_path = np.load(os.path.join(sc, "node_path.npy"), mmap_mode="r")
        tok_node = np.load(os.path.join(sc, "tok_node.npy"), mmap_mode="r")
        ntr = manifest(os.path.join(pk, "meta.json"))["n_train"]
        recs = np.random.default_rng(seed).choice(ntr, size=n_records, replace=False)
        cd, cp = collections.Counter(), collections.Counter()
        nodes = interior = 0
        ranks = []
        for r in recs:
            a, b = int(tok_off[r]), int(tok_off[r + 1])
            tn = np.asarray(tok_node[a:b])
            na, nb = int(node_off[r]), int(node_off[r + 1])
            P = np.asarray(node_path[na:nb])
            paths = {}
            for t in tn:
                if t < 0:
                    continue
                t = int(t)
                if t not in paths:
                    row = P[t]
                    paths[t] = tuple(int(x) for x in row[row >= 0])
                p = paths[t]
                cd[len(p)] += 1
                cp[p] += 1
            srt = sorted(paths.values())
            nodes += len(srt)
            interior += sum(1 for i in range(len(srt) - 1)
                            if len(srt[i + 1]) > len(srt[i]) and srt[i + 1][:len(srt[i])] == srt[i])
            for p in srt:
                ranks.extend(p)
        ranks = np.array(ranks)
        hd, hp = _entropy(cd), _entropy(cp)
        out[k] = dict(H_depth=hd, H_path=hp, H_path_given_depth=hp - hd,
                      interior=interior / nodes, nodes=nodes, records=n_records,
                      rank_p99=float(np.percentile(ranks, 99)), rank_max=int(ranks.max()))
    return out


# ---------------------------------------------------------------------------------- the results

CONTRASTS = {
    # headline: sequence + tree vs sequence only
    "py_both": ("py", "rope_both", "rope_seq"),
    "sh_both": ("sheets", "both_v2", "seq_v2"),
    "sha_both": ("sheets", "both_vanilla_v2", "vanilla_seq_v2"),
    "ws_both": ("webs", "both", "seq"),
    # Python, every configuration against sequence only
    "py_leaf": ("py", "rope_leaf", "rope_seq"), "py_graph": ("py", "rope_graph", "rope_seq"),
    "py_all": ("py", "rope_all", "rope_seq"), "py_rel": ("py", "rel", "rope_seq"),
    "py_none": ("py", "none", "rope_seq"), "py_alt": ("py", "layer_alt", "rope_seq"),
    "py_seqfirst": ("py", "layer_seq_first", "rope_seq"),
    "py_thirds": ("py", "layer_thirds", "rope_seq"),
    "py_treefirst": ("py", "layer_tree_first", "rope_seq"),
    "py_head": ("py", "head_split", "rope_seq"), "py_depth": ("py", "depth", "rope_seq"),
    "py_top": ("py", "toplevel", "rope_seq"),
    # Python, second-order contrasts
    "py_all_leaf": ("py", "rope_all", "rope_leaf"), "py_all_both": ("py", "rope_all", "rope_both"),
    "py_leaf_graph": ("py", "rope_leaf", "rope_graph"), "py_rel_both": ("py", "rel", "rope_both"),
    "py_alt_both": ("py", "layer_alt", "rope_both"),
    "py_seqfirst_both": ("py", "layer_seq_first", "rope_both"),
    "py_thirds_both": ("py", "layer_thirds", "rope_both"),
    "py_treefirst_both": ("py", "layer_tree_first", "rope_both"),
    "py_head_both": ("py", "head_split", "rope_both"), "py_head_alt": ("py", "head_split", "layer_alt"),
    "py_alt_all": ("py", "layer_alt", "rope_all"),
    "py_depth_both": ("py", "depth", "rope_both"), "py_depth_top": ("py", "depth", "toplevel"),
    "py_splice": ("py", "splice", "rope_both"), "py_anchor": ("py", "anchor", "rope_both"),
    "py_slot": ("py", "ref_slot", "rope_both"),
    # spreadsheets
    "sh_depth": ("sheets", "depth_v2", "seq_v2"), "sh_depth_both": ("sheets", "depth_v2", "both_v2"),
    "sh_depth_top": ("sheets", "depth_v2", "toplevel_v2"), "sh_top": ("sheets", "toplevel_v2", "seq_v2"),
    "sh_ref": ("sheets", "ref_anchor", "both_v2"),
    "sha_ref": ("sheets", "ref_vanilla_v2", "both_vanilla_v2"),
    "sh_delta": ("sheets", "ref_delta", "ref_anchor"), "sh_delta_both": ("sheets", "ref_delta", "both_v2"),
    "sh_split_both": ("sheets", "layer_split_v2", "both_v2"),
    "sh_split_seq": ("sheets", "layer_split_v2", "seq_v2"),
    # web pages
    "ws_depth": ("webs", "depth", "seq"), "ws_depth_both": ("webs", "depth", "both"),
    # sequence + tree + depth beyond Python (queued at 10 seeds each)
    "sh_rel_both": ("sheets", "rel_v2", "both_v2"), "ws_rel_both": ("webs", "rel", "both"),
    "sh_rel": ("sheets", "rel_v2", "seq_v2"), "ws_rel": ("webs", "rel", "seq"),
    "py_rel_depth": ("py", "rel", "depth"), "sh_rel_depth": ("sheets", "rel_v2", "depth_v2"),
    "ws_rel_depth": ("webs", "rel", "depth"),
    "sh_rel_ref": ("sheets", "rel_v2", "ref_anchor"), "sh_rel_split": ("sheets", "rel_v2", "layer_split_v2"),
    # full-width RoPE (all 64 planes by sequence) against the 16-plane baseline
    # sequence + grid: the first three path levels only (sheet, row, column), 6 planes
    "sh_grid": ("sheets", "grid_v2", "seq_v2"), "sh_grid_both": ("sheets", "grid_v2", "both_v2"),
    "sh_grid_rel": ("sheets", "grid_v2", "rel_v2"), "sh_grid_depth": ("sheets", "grid_v2", "depth_v2"),
    "sh_grid_top": ("sheets", "grid_v2", "toplevel_v2"), "sh_grid_ref": ("sheets", "grid_v2", "ref_anchor"),
    "sh_grid_split": ("sheets", "grid_v2", "layer_split_v2"), "sh_grid_full": ("sheets", "grid_v2", "full_rope_v2"),
    # sequence + depth + rank: depth + the node's own sibling rank, 4 planes (Python, web pages)
    "py_tg": ("py", "drank", "rope_seq"), "py_tg_depth": ("py", "drank", "depth"),
    "py_tg_both": ("py", "drank", "rope_both"), "py_tg_rel": ("py", "drank", "rel"),
    "py_tg_full": ("py", "drank", "full_rope"),
    "ws_tg": ("webs", "drank", "seq"), "ws_tg_depth": ("webs", "drank", "depth"),
    "ws_tg_both": ("webs", "drank", "both"), "ws_tg_rel": ("webs", "drank", "rel"),
    "ws_tg_full": ("webs", "drank", "full_rope"),
    "py_full": ("py", "full_rope", "rope_seq"), "sh_full": ("sheets", "full_rope_v2", "seq_v2"),
    "ws_full": ("webs", "full_rope", "seq"),
    # CONTROLS for "information or delivery?" (complete)
    "ad_grid": ("sheets", "grid_vanilla_v2", "vanilla_seq_v2"), "ad_grid_tree": ("sheets", "grid_vanilla_v2", "both_vanilla_v2"),
    "ad_depth": ("sheets", "depth_vanilla_v2", "vanilla_seq_v2"),
    "py_emb": ("py", "depthemb", "rope_seq"), "py_emb_rot": ("py", "depthemb", "depth"),
    "sh_emb": ("sheets", "depthemb_v2", "seq_v2"), "sh_emb_rot": ("sheets", "depthemb_v2", "depth_v2"),
    "ws_emb": ("webs", "depthemb", "seq"), "ws_emb_rot": ("webs", "depthemb", "depth"),
    "py_shuf": ("py", "depthshuf", "rope_seq"), "ws_shuf": ("webs", "depthshuf", "seq"),
    "py_shuf_rot": ("py", "depthshuf", "depth"), "ws_shuf_rot": ("webs", "depthshuf", "depth"),
    # CHAT (WildChat; seeds landing): every configuration against sequence only, and the overlaps
    "ch_grid": ("chat", "grid", "seq"), "ch_grid_depth": ("chat", "grid_depth", "seq"),
    "ch_depth": ("chat", "depth", "seq"), "ch_drank": ("chat", "drank", "seq"),
    "ch_both": ("chat", "both", "seq"), "ch_rel": ("chat", "rel", "seq"),
    "ch_grid_vs_depth": ("chat", "grid", "depth"), "ch_gd_vs_depth": ("chat", "grid_depth", "depth"),
    "ch_gd_vs_grid": ("chat", "grid_depth", "grid"), "ch_drank_vs_depth": ("chat", "drank", "depth"),
    "ch_gd_vs_both": ("chat", "grid_depth", "both"),
    # Python statement grid (statement number + position within statement, 6 planes; + depth, 8)
    "py_sg": ("py", "stmtgrid", "rope_seq"), "py_sg_depth": ("py", "stmtgrid", "depth"),
    "py_sgd": ("py", "stmtgrid_depth", "rope_seq"), "py_sgd_depth": ("py", "stmtgrid_depth", "depth"),
    "py_sgd_both": ("py", "stmtgrid_depth", "rope_both"), "py_sgd_rel": ("py", "stmtgrid_depth", "rel"),
    # labels added to depth, each against depth alone
    "py_dp": ("py", "dparent", "depth"), "py_dpg": ("py", "dparentg", "depth"),
    "py_drp": ("py", "drankparent", "depth"), "py_par": ("py", "parityid", "depth"),
    "ws_dp": ("webs", "dparent", "depth"), "ws_dpg": ("webs", "dparentg", "depth"),
    "sh_dp": ("sheets", "dparent_v2", "depth_v2"), "sh_dpg": ("sheets", "dparentg_v2", "depth_v2"),
    "ch_dp": ("chat", "dparent", "depth"), "ch_dpg": ("chat", "dparentg", "depth"),
}

#: the preliminary spreadsheet build (shorter window) has its own planned length
PRELIM_SHEETS_FINAL = 12880


def prelim_ref_contrast() -> dict:
    A = arm_runs("sheets", "ref")
    B = arm_runs("sheets", "both")
    shared = sorted(s for s in set(A) & set(B)
                    if A[s][3] == PRELIM_SHEETS_FINAL and B[s][3] == PRELIM_SHEETS_FINAL)
    d = [A[s][0] - B[s][0] for s in shared]
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return dict(n=len(d), mean=m, t=m / se, won=sum(x < 0 for x in d), diffs=list(zip(shared, d)))


def progress(root: str, arm: str, final: int, planned: int = 0) -> tuple[int, int, list]:
    """(finished seeds, total seeds, [(seed, fraction done)] for unfinished ones), from the logs.

    ``planned`` is the intended seed count for a queue still launching runs; seeds with no run
    directory yet count as not finished."""
    runs = {}
    base = os.path.join(RUNS, root)
    for dpath in sorted(os.listdir(base) if os.path.isdir(base) else []):
        m = re.fullmatch(re.escape(arm) + r"_s(\d+)", dpath)
        if not m:
            continue
        last = 0
        p = os.path.join(RUNS, root, dpath, "train.jsonl")
        if os.path.exists(p):
            for line in open(p):
                try:
                    last = max(last, json.loads(line).get("step", 0))
                except json.JSONDecodeError:
                    pass
        runs[int(m.group(1))] = last
    done = sum(1 for v in runs.values() if v >= final)
    return done, max(len(runs), planned), [(s, v / final) for s, v in sorted(runs.items()) if v < final]


def collect() -> dict:
    D = dict(c={k: contrast(*v) for k, v in CONTRASTS.items()})
    D["webl"] = webl_data()
    D["test8192"], D["test2048"] = py_test(8192), py_test(2048)
    D["corpus"] = corpus_facts()
    D["tree"] = tree_stats()
    D["prelim_ref"] = prelim_ref_contrast()
    D["cfg"] = {k: setting_config(k) for k in SETTINGS}
    D["sd"] = {k: seed_sd(*v) for k, v in {"py": ("py", "rope_seq"), "sheets": ("sheets", "seq_v2"),
                                            "webs": ("webs", "seq")}.items()}
    D["prog"] = {
        "sh_split": progress("sheets", "layer_split_v2", SETTINGS["sheets"]["final"]),
        "sh_rel": progress("sheets", "rel_v2", SETTINGS["sheets"]["final"], planned=10),
        "ws_rel": progress("webc", "rel", SETTINGS["webs"]["final"], planned=10),
        "py_full": progress("py", "full_rope", SETTINGS["py"]["final"], planned=5),
        "sh_full": progress("sheets", "full_rope_v2", SETTINGS["sheets"]["final"], planned=10),
        "ws_full": progress("webc", "full_rope", SETTINGS["webs"]["final"], planned=10),
        "ad_grid": progress("sheets", "grid_vanilla_v2", SETTINGS["sheets"]["final"], planned=10),
        "ad_depth": progress("sheets", "depth_vanilla_v2", SETTINGS["sheets"]["final"], planned=10),
        "py_emb": progress("py", "depthemb", SETTINGS["py"]["final"], planned=5),
        "sh_emb": progress("sheets", "depthemb_v2", SETTINGS["sheets"]["final"], planned=10),
        "ws_emb": progress("webc", "depthemb", SETTINGS["webs"]["final"], planned=10),
        "py_shuf": progress("py", "depthshuf", SETTINGS["py"]["final"], planned=5),
        "ws_shuf": progress("webc", "depthshuf", SETTINGS["webs"]["final"], planned=10),
        "html_seq": progress("webc_html", "seq", SETTINGS["webs"]["final"], planned=10),
        "html_depth": progress("webc_html", "depth", SETTINGS["webs"]["final"], planned=10),
    }
    return D


# ================================================================================= HTML helpers

MINUS = "−"

#: chart palette: reference categorical slots 1, 2, 3 and 7, validated all-pairs (light mode)
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
C_OFF = "#e6e5df"           # an unrotated plane: a neutral, not a series colour
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def num(x, d: int = 4, sign: bool = True) -> str:
    if x is None:
        return "—"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def pfmt(p) -> str:
    if p is None:
        return "—"
    if p < 0.0001:
        return "< 0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def pct(x, d: int = 1) -> str:
    return f"{x:.{d}f}".replace("-", MINUS) + "%"


def intc(x) -> str:
    return f"{int(x):,}"


# ------------------------------------------------------------------------------------- glossary

#: key -> (display name, plain-English definition). Every [[key]] in the prose becomes a link to
#: the glossary entry with the definition on hover and on keyboard focus.
GLOSSARY = collections.OrderedDict([
    # experimental vocabulary
    ("setting", ("setting", "One corpus trained at one model size, context length and training "
                 "length. Configurations are compared only within a setting; loss values from "
                 "different settings are never compared directly, because their token streams differ.")),
    ("configuration", ("configuration", "One way of filling the rotary planes with position "
                       "information (for example sequence only, or sequence + tree). Within a "
                       "setting, configurations differ in nothing else.")),
    ("Python", ("Python", "Setting: 423,111 Python programs written as a compact pre-order "
                "serialisation of their syntax tree; 4.7M-parameter model, 1,024-token context, "
                "about eight passes; 5 to 20 seeds per configuration.")),
    ("Spreadsheets", ("Spreadsheets", "Setting: 41,245 spreadsheet workbooks written out cell by "
                      "cell; 4.7M-parameter model, 4,096-token context, about eight passes; 10 "
                      "seeds per configuration.")),
    ("web pages", ("web pages", "Setting: Web pages (visible text only), whole pages of at most 2,048 "
               "tokens; 4.7M-parameter model, a third of a pass; 10 seeds per configuration. "
               "The small, many-seed web setting.")),
    ("the larger web run", ("the larger web run", "Setting: Web pages (visible text only) at a 4,096-token context; "
               "31.5M-parameter model, one full pass over 25.4 billion tokens; one seed per "
               "configuration. The large web setting.")),
    ("seed", ("seed", "A random seed fixes the initial weights and the order of training batches. "
              "Two configurations trained at the same seed start from identical weights and see "
              "identical batches, so they differ only in the position signal.")),
    ("paired", ("paired difference", "For one seed, the configuration's loss minus the "
                "baseline's loss at the same step. Comparing per seed removes the seed-to-seed "
                "variation both runs share.")),
    ("delta", ("Δ", "Configuration loss minus baseline loss, in validation nats per token, at the "
               "final step. Negative means the configuration is better. The baseline is always "
               "named.")),
    ("won", ("won k/n", "The number of seeds, out of n, on which the configuration's loss is "
             "lower than the baseline's.")),
    ("nats", ("nats per token", "Average negative log-likelihood of each held-out token under "
              "the model, using the natural logarithm. Lower is better. Perplexity is e raised to "
              "this value.")),
    ("pplred", ("perplexity reduction", "100 × (1 − e^Δ): how much lower, in percent, the "
                "configuration's perplexity is than the baseline's. A Δ of −0.01 nats per token "
                "is about a 1% perplexity reduction. Positive means better.")),
    ("milestone", ("milestone", "A validation score taken every 5% of a run's planned steps, so "
                   "a full run has 20. Configurations are compared only at milestones both runs "
                   "have reached.")),
    ("pass", ("pass", "One epoch: the number of steps it takes to show the model every training "
              "record once.")),
    ("split", ("held-out split", "Records never trained on. The validation split is scored at "
               "every milestone (a fixed set of 1,024 records); the test split is scored once, at "
               "the final step, as an independent check.")),
    # tree vocabulary
    ("tree", ("tree", "The structure a document was generated from: the syntax tree of a "
              "program, the sheet / row / cell structure of a workbook, or the element tree (DOM) "
              "of a web page.")),
    ("node", ("node", "One element of the tree: a syntax-tree node, a spreadsheet cell or formula "
              "part, an HTML element or a run of text. Each token is assigned to one node.")),
    ("path", ("path", "The list of sibling ranks from the root of the tree down to a token's "
              "node — one number per level. Two nodes have the same path only if they are the "
              "same node.")),
    ("rank", ("sibling rank", "A node's position among its parent's children, counting from 0 "
              "in document order. (On spreadsheets the row and column levels use the absolute row "
              "and column index instead, because empty rows are skipped in the data.)")),
    ("level", ("level", "Distance from the root: level 1 is a child of the root, level 2 a "
               "grandchild, and so on. Up to 22 levels are encoded.")),
    ("depth", ("depth", "The number of levels a token's node sits below the root — one integer "
               "per token.")),
    ("within", ("within-node index", "A token's position inside its own node: 0 for the node's "
                "first token, 1 for the next. It restarts at every node, so it is a local copy of "
                "sequence position.")),
    ("npt", ("nodes per token", "Tree nodes divided by tokens in a corpus: how much structure "
             "each token carries. About 0.86 for Python, 0.56 for spreadsheets and 0.13 for web "
             "text.")),
    ("interior", ("interior node", "A node that has tokens of its own AND a descendant that also "
                  "has tokens. Depth can recover the tree only if every ancestor is interior in "
                  "this sense.")),
    ("preorder", ("pre-order serialisation", "Writing a tree out parent first, then each child's "
                  "subtree in order (depth-first). All three corpora are written this way.")),
    ("render", ("render", "How a spreadsheet is written as text. The values render writes only "
                "each cell's content; the addressed render writes 'address:content' (e.g. "
                "G4:=E4+F4). Tabs separate cells, newlines rows.")),
    ("reference", ("reference", "A formula operand that names another cell, such as E4 in "
                   "=E4+F4; in Python, a use of a variable that names its definition.")),
    # mechanism vocabulary
    ("plane", ("plane", "One two-dimensional rotary subspace: two of a head's dimensions rotated "
               "together by one angle. A 128-wide attention head has 64 planes.")),
    ("unrotated", ("unrotated plane", "A plane given angle 0 everywhere. It carries no position "
                   "and acts as an ordinary content channel.")),
    ("fastslow", ("fast and slow plane", "Each tree level has two planes. At initialisation the "
                  "fast plane turns 0.316 rad per sibling (adjacent siblings stay distinguishable) "
                  "and the slow plane 1/16 of that, so it can count about 318 siblings before its "
                  "angle wraps round. Both rates are learned.")),
    ("wrap", ("wrap", "When a plane's angle passes 2π it repeats, so two different ranks become "
              "indistinguishable on that plane.")),
    ("colocation", ("co-location", "Both position signals inside the same attention head, so a "
                    "single attention score can depend on sequence distance and tree position at "
                    "once.")),
    ("surrogate", ("sequence surrogate", "A signal that stands in for sequence position. The "
                   "within-node index is one: it counts tokens in reading order.")),
    ("plctrl", ("plane-matched control", "A configuration with exactly the same number of "
                "rotating planes and learned parameters as another, but spending them on a "
                "different signal. Used to tell 'the signal helps' from 'extra planes help'.")),
    # configurations
    ("seq", ("sequence only", "Configuration: 16 planes rotate by the token's index in the "
             "sequence (standard partial rotary embedding); every other plane is unrotated. The "
             "usual baseline.")),
    ("graph", ("tree only", "Configuration: the 44 tree planes rotate by the token's path; the "
               "sequence planes are unrotated, so the model is never told token order directly.")),
    ("leaf", ("tree + within-node index", "Configuration: the 44 tree planes plus 4 planes for the "
              "within-node index; no sequence position.")),
    ("both", ("sequence + tree", "Configuration: 16 sequence planes plus the 44 tree planes, "
              "inside every head. The main structural configuration.")),
    ("all", ("all signals", "Configuration: sequence + tree + within-node index (16 + 44 + 4 = all "
             "64 planes).")),
    ("depthcfg", ("sequence + depth", "Configuration: 16 sequence planes plus 2 planes rotated by "
                  "the token's depth; the other 46 planes unrotated. No per-level path at all.")),
    ("toplevel", ("top-level-only control", "Configuration: 16 sequence planes plus 2 planes "
                  "rotated by level 1 of the path only (which child of the root the token "
                  "descends from). The plane-matched control for sequence + depth.")),
    ("none", ("no position", "Configuration: no plane rotates. The model gets no positional "
              "information except what causal masking implies.")),
    ("gridcfg", ("sequence + grid", "Spreadsheet configuration: sequence position on 16 planes plus "
                 "the first three levels of the path — sheet, row and column numbers — on 6 planes; "
                 "the formula-nesting levels below the cell are not rotated.")),
    ("drank", ("sequence + depth + rank", "Configuration for nested documents: sequence position on 16 "
                  "planes plus two numbers per token on 2 planes each — its node's depth and the "
                  "node's rank among its parent's children, the tree's analogue of a spreadsheet's "
                  "row and column. Run on Python and web pages.")),
    ("fullrope", ("full RoPE", "Control configuration: all 64 planes rotate by the token's index in "
                  "the sequence, as in standard RoPE (the baseline rotates 16). It checks that the "
                  "structured configurations do not win merely by rotating more planes.")),
    ("rel", ("sequence + tree + depth", "Configuration: sequence + tree plus 2 planes "
             "rotated by depth (62 of 64 planes). Best or tied overall; the paper's recommendation is the cheaper sequence + depth. On the "
             "depth planes a parent and its child always differ by exactly one learned step, so "
             "'is the parent of' is a constant offset.")),
    ("layersplit", ("layer split", "Configuration family: each layer carries ONE signal at full "
                    "width — sequence layers rotate all 64 planes by sequence position, tree "
                    "layers rotate 44 tree + 4 within-node planes. Written as one letter per layer "
                    "(s sequence, t tree, n no rotation): e.g. ststst.")),
    ("headsplit", ("head split", "Configuration: in every layer, half the heads carry sequence "
                   "position at full width and the other half the tree (+ within-node index) at "
                   "full width. No single head sees both.")),
    ("refrot", ("reference rotation", "Configuration: sequence + tree, except that a token "
                "belonging to a reference takes the coordinates of the thing it refers to on the "
                "relevant tree planes, so a reference and its target line up in angle.")),
    ("delta_off", ("directed-offset variant", "Reference rotation where the reference sits a "
                   "learned constant angle away from its target instead of exactly on it, so "
                   "'refers to' differs from 'is next to'. The constant starts at 0.")),
])


def term(key: str, display: str | None = None) -> str:
    name, definition = GLOSSARY[key]
    shown = display or name
    return (f'<a class="term" href="#g-{esc(key)}" data-tip="{esc(name)}&#10;{esc(definition)}">'
            f"{esc(shown)}</a>")


_TERM_RE = re.compile(r"\[\[([A-Za-z_\-]+)(?:\|([^\]]+))?\]\]")


def inline(s: str) -> str:
    """Escape, then apply the small markup used in the prose: [[key|shown]], **b**, *i*, `code`.

    Terms are swapped for placeholders first so that emphasis may span a term."""
    links = []
    def stash(m):
        links.append(term(m.group(1), m.group(2)))
        return f"\x00{len(links) - 1}\x00"
    s = _fmt(esc(_TERM_RE.sub(stash, s)))
    return re.sub("\x00(\\d+)\x00", lambda m: links[int(m.group(1))], s)


def _fmt(s: str) -> str:
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def md(text: str) -> str:
    """Paragraphs on blank lines; a block whose lines all start with '- ' is a list."""
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    out = []
    for b in blocks:
        lines = [ln.strip() for ln in b.split("\n")]
        if all(ln.startswith("- ") for ln in lines):
            out.append("<ul>" + "".join(f"<li>{inline(ln[2:])}</li>" for ln in lines) + "</ul>")
        else:
            out.append(f"<p>{inline(' '.join(lines))}</p>")
    return "\n".join(out)


def table(head: list, rows: list, cls: str = "data", num_cols: tuple = ()) -> str:
    """``head`` and ``rows`` are inline-markup strings; columns in ``num_cols`` align right."""
    th = "".join(f'<th{" class=n" if i in num_cols else ""}>{inline(h)}</th>'
                 for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td{" class=n" if i in num_cols else ""}>{inline(str(c))}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(fid: str, title: str, svg: str, caption: str, table_html: str) -> str:
    return (f'<figure class="fig" id="{fid}"><figcaption class="ft">{inline(title)}</figcaption>'
            f'{svg}<div class="fc">{md(caption)}</div>'
            f'<details class="tv"><summary>Table view</summary>{table_html}</details></figure>')


# ---------------------------------------------------------------------------------------- axes

def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    first = math.ceil(lo / step) * step
    ticks, v = [], first
    while v <= hi + 1e-12:
        ticks.append(round(v, 10))
        v += step
    return ticks


def tick_label(v: float, step: float) -> str:
    d = max(0, -int(math.floor(math.log10(step))) + (1 if step / 10 ** math.floor(math.log10(step)) == 2.5 else 0))
    return f"{v:.{d}f}".replace("-", MINUS)


_uid = [0]


def uid(prefix: str) -> str:
    _uid[0] += 1
    return f"{prefix}{_uid[0]}"


# ------------------------------------------------------------------------- per-seed dot chart

def dot_chart(rows: list[dict], *, axis: str, left: str, right: str, fmt=num,
              color: str = C_BLUE, label_w: int = 270) -> str:
    """One row per contrast: each seed a dot, the mean a tick, its 95% interval a line.

    ``rows``: dict(label, sub, pts=[(seed, value)], mean, lo, hi, n, won, p, note). Values are in
    display units. ``left`` / ``right`` say which way is better.
    """
    W, R, T, RH = 860, 175, 14, 44
    x0, x1 = label_w, W - R
    vals = [v for r in rows for _, v in r["pts"]] + [r["lo"] for r in rows] + [r["hi"] for r in rows] + [0.0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.001
    lo, hi = lo - pad, hi + pad
    ticks = nice_ticks(lo, hi, 6)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    H = T + RH * len(rows) + 62
    g = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(axis)}">']
    yb = T + RH * len(rows)
    for tv in ticks:
        g.append(f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="{T}" y2="{yb}" stroke="{GRID}" stroke-width="1"/>')
        g.append(f'<text x="{X(tv):.1f}" y="{yb + 16}" class="tk" text-anchor="middle">{tick_label(tv, step)}</text>')
    g.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{T - 4}" y2="{yb}" stroke="{MUTED}" stroke-width="1"/>')
    g.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{yb + 36}" class="al" text-anchor="middle">{esc(axis)}</text>')
    g.append(f'<text x="{x0}" y="{yb + 54}" class="dir">{esc("← " + left)}</text>')
    g.append(f'<text x="{x1}" y="{yb + 54}" class="dir" text-anchor="end">{esc(right + " →")}</text>')
    for i, r in enumerate(rows):
        y = T + RH * i + RH / 2
        if i:
            g.append(f'<line x1="12" x2="{W - 8}" y1="{T + RH * i}" y2="{T + RH * i}" stroke="{GRID}" stroke-width="1" opacity=".55"/>')
        g.append(f'<text x="{x0 - 14}" y="{y - 2}" class="rl" text-anchor="end">{esc(r["label"])}</text>')
        if r.get("sub"):
            g.append(f'<text x="{x0 - 14}" y="{y + 13}" class="rs" text-anchor="end">{esc(r["sub"])}</text>')
        # interval and mean
        g.append(f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y}" y2="{y}" stroke="{INK}" stroke-width="2" stroke-linecap="round"/>')
        g.append(f'<line x1="{X(r["mean"]):.1f}" x2="{X(r["mean"]):.1f}" y1="{y - 10}" y2="{y + 10}" stroke="{INK}" stroke-width="2.5"/>')
        n = r["n"]
        ci = (f"95% interval {fmt(r['lo'])} to {fmt(r['hi'])}" if n > 1 else "one run, no interval")
        # the row's sub-label already gives the seed count; keep this short enough to fit
        stat = (f"won {r['won']}/{n}" if r.get("won") is not None else f"n = {n}") +\
               (f" · p {pfmt(r['p'])}" if r.get("p") is not None else "")
        g.append(f'<rect class="hit" x="{min(X(r["lo"]), X(r["mean"]) - 6):.1f}" y="{y - 12}" '
                 f'width="{max(abs(X(r["hi"]) - X(r["lo"])), 12):.1f}" height="24" fill="transparent" tabindex="0" '
                 f'data-tip="{esc("mean " + fmt(r["mean"]))}&#10;{esc(r["label"])}&#10;{esc(ci)}&#10;{esc(stat)}"/>')
        for j, (seed, v) in enumerate(r["pts"]):
            jy = y + ((j % 5) - 2) * 3.2
            did = uid("d")
            g.append(f'<circle id="{did}" class="dot" cx="{X(v):.1f}" cy="{jy:.1f}" r="4" fill="{color}" stroke="{SURF}" stroke-width="2" fill-opacity=".8"/>')
            g.append(f'<circle class="hit" cx="{X(v):.1f}" cy="{jy:.1f}" r="11" fill="transparent" tabindex="0" data-mark="{did}" '
                     f'data-tip="{esc(fmt(v))}&#10;{esc(r["label"])}, seed {seed}"/>')
        ann = fmt(r["mean"])
        g.append(f'<text x="{x1 + 14}" y="{y - 2}" class="av">{esc(ann)}</text>')
        g.append(f'<text x="{x1 + 14}" y="{y + 13}" class="rs">{esc(stat)}</text>')
    g.append("</svg>")
    return "".join(g)


def dot_table(rows: list[dict], fmt=num, unit: str = "Δ") -> str:
    head = ["", "n", f"mean {unit}", "95% interval", "won", "p", "per seed (seed: value)"]
    body = []
    for r in rows:
        per = ", ".join(f"{s}: {fmt(v)}" for s, v in r["pts"])
        body.append([r["label"] + (f" — {r['sub']}" if r.get("sub") else ""), r["n"], fmt(r["mean"]),
                     f"{fmt(r['lo'])} to {fmt(r['hi'])}" if r["n"] > 1 else "—",
                     f"{r['won']}/{r['n']}" if r.get("won") is not None else "—", pfmt(r.get("p")), per])
    return table(head, body, num_cols=(1, 2, 3, 4, 5))


def crow(c: dict, label: str, sub: str = "") -> dict:
    """A dot-chart row in Δ nats from a contrast."""
    return dict(label=label, sub=sub, pts=c["diffs"], mean=c["mean"], lo=c["lo"], hi=c["hi"],
                n=c["n"], won=c["won"], p=c.get("p"))


def to_ppl(row: dict) -> dict:
    """A Δ-nats row mapped to perplexity reduction, 100 (1 - e^Δ). The map is decreasing, so the
    interval's ends swap and 'better' moves from left to right."""
    f = lambda v: 100 * (1 - math.exp(v))
    return dict(row, pts=[(s, f(v)) for s, v in row["pts"]], mean=f(row["mean"]),
                lo=f(row["hi"]), hi=f(row["lo"]))


def ppl_row(c: dict, label: str, sub: str = "") -> dict:
    return to_ppl(crow(c, label, sub))


# ------------------------------------------------------------------------ milestone curve chart

def curve_chart(series: list[dict], final: int, *, axis: str, left: str, right: str) -> str:
    """Gap to the baseline at every milestone, one line per configuration.

    ``series``: dict(label, color, pts={step: value}). Hovering anywhere in a milestone's column
    reads out every series at that milestone (the crosshair snaps to milestones).
    """
    W, L, R, T, B = 860, 64, 190, 34, 64
    x0, x1, y0, y1 = L, W - R, T, 330
    vals = [v for s in series for v in s["pts"].values()] + [0.0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    ticks = nice_ticks(lo, hi, 6)
    step = ticks[1] - ticks[0]
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    X = lambda s: x0 + s / final * (x1 - x0)
    Y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    cid = uid("cx")
    g = [f'<svg class="chart" viewBox="0 0 {W} {y1 + B}" role="img" aria-label="{esc(axis)}">']
    # legend: line keys, above the plot
    lx = x0
    for s in series:
        _da = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        g.append(f'<line x1="{lx}" x2="{lx + 18}" y1="12" y2="12" stroke="{s["color"]}" stroke-width="2.5" stroke-linecap="round"{_da}/>')
        g.append(f'<text x="{lx + 24}" y="16" class="lg">{esc(s["label"])}</text>')
        lx += 34 + 7.2 * len(s["label"])
    for tv in ticks:
        g.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" stroke="{GRID}" stroke-width="1"/>')
        g.append(f'<text x="{x0 - 8}" y="{Y(tv) + 4:.1f}" class="tk" text-anchor="end">{tick_label(tv, step)}</text>')
    for f in (0, .25, .5, .75, 1):
        g.append(f'<text x="{X(f * final):.1f}" y="{y1 + 18}" class="tk" text-anchor="middle">{int(f * 100)}%</text>')
    g.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{MUTED}" stroke-width="1"/>')
    g.append(f'<text x="{x1 + 8}" y="{Y(0) + 4:.1f}" class="rs">sequence only (0)</text>')
    g.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 + 38}" class="al" text-anchor="middle">{esc(axis)}</text>')
    g.append(f'<text x="{x0}" y="{y1 + 56}" class="dir">{esc("↓ " + left)}</text>')
    g.append(f'<text x="{x1}" y="{y1 + 56}" class="dir" text-anchor="end">{esc("↑ " + right)}</text>')
    g.append(f'<line id="{cid}" class="xhair" x1="0" x2="0" y1="{y0}" y2="{y1}" stroke="{MUTED}" stroke-width="1" visibility="hidden"/>')
    ends = []
    for s in series:
        pts = sorted(s["pts"].items())
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(k):.1f},{Y(v):.1f}" for i, (k, v) in enumerate(pts))
        _da = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        _w = "3.2" if s["label"] == "sequence + depth" else "2"
        g.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{_w}" stroke-linejoin="round" stroke-linecap="round"{_da}/>')
        k, v = pts[-1]
        g.append(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="4" fill="{s["color"]}" stroke="{SURF}" stroke-width="2"/>')
        ends.append([X(k), Y(v), s["label"], s.get("end", "")])
    # end labels: nudge apart only when they would overprint, with a leader line back to the dot
    ends.sort(key=lambda e: e[1])
    placed = []
    for ex, ey, lab, extra in ends:
        ly = ey
        for px, py in placed:
            if abs(px - ex) < 150 and abs(py - ly) < 15:
                ly = py + 15
        placed.append((ex, ly))
        if abs(ly - ey) > 1:
            g.append(f'<line x1="{ex + 5:.1f}" x2="{ex + 9:.1f}" y1="{ey:.1f}" y2="{ly - 4:.1f}" stroke="{MUTED}" stroke-width="1"/>')
        g.append(f'<text x="{ex + 10:.1f}" y="{ly:.1f}" class="el">{esc(lab + extra)}</text>')
    steps = sorted({k for s in series for k in s["pts"]})
    for i, k in enumerate(steps):
        left_b = X((steps[i - 1] + k) / 2) if i else X(k) - 10
        right_b = X((k + steps[i + 1]) / 2) if i + 1 < len(steps) else X(k) + 10
        lines = [f"milestone {round(100 * k / final)}% (step {k:,})"]
        for s in series:
            if k in s["pts"]:
                lines.append(f"{s['label']}: {num(s['pts'][k])}")
        tip = "&#10;".join(esc(x) for x in lines)
        g.append(f'<rect class="band" x="{left_b:.1f}" y="{y0}" width="{right_b - left_b:.1f}" height="{y1 - y0}" '
                 f'fill="transparent" tabindex="0" data-xh="{cid}" data-x="{X(k):.1f}" data-tip="{tip}"/>')
    g.append("</svg>")
    return "".join(g)


# --------------------------------------------------------------------------- plane allocation

ROLE_COLOR = {"seq": C_BLUE, "tree": C_ORANGE, "tok": C_AQUA, "depth": C_VIOLET, "off": C_OFF}
ROLE_NAME = {"seq": "sequence position", "tree": "tree path", "tok": "within-node index",
             "depth": "depth", "off": "unrotated"}


def planes(cfg: str) -> list[tuple[str, str]]:
    """(role, description) for each of the 64 planes of one head under a configuration."""
    out = []
    def seq(n):
        for i in range(n):
            out.append(("seq", f"sequence position, frequency 10000^(−{i}/{n}): the angle is t × that "
                               f"frequency, t = the token's index in the sequence"))
    def tree(levels=22):
        for k in range(22):
            for p in ("fast", "slow"):
                if k < levels:
                    out.append(("tree", f"tree level {k + 1}, {p} plane: angle = presence offset + "
                                        f"(sibling rank at level {k + 1}) × learned step"))
                else:
                    out.append(("off", f"tree level {k + 1}, {p} plane: not used by this configuration "
                                       f"(only level 1 is kept), so angle 0"))
    def tok():
        for p in range(4):
            out.append(("tok", f"within-node index, plane {p + 1}: angle = presence offset + (index of "
                               f"the token inside its node) × learned step; wraps after ~{[20, 80, 318, 1270][p]} tokens"))
    def depth():
        for p in ("fast", "slow"):
            out.append(("depth", f"depth, {p} plane: angle = (depth of the token's node) × learned step; "
                                 f"a parent and child differ by exactly one step"))
    def off(n, why="unrotated: an ordinary content channel"):
        for _ in range(n):
            out.append(("off", why))
    if cfg == "seq":
        seq(16); off(48)
    elif cfg == "graph":
        off(16, "sequence plane, held at angle 0 in this configuration"); tree(); off(4)
    elif cfg == "leaf":
        off(16, "sequence plane, held at angle 0 in this configuration"); tree(); tok()
    elif cfg == "both":
        seq(16); tree(); off(4)
    elif cfg == "all":
        seq(16); tree(); tok()
    elif cfg == "depthcfg":
        seq(16); depth(); off(46)
    elif cfg == "toplevel":
        seq(16); tree(levels=1); off(4)
    elif cfg == "rel":
        seq(16); tree(); depth(); off(2)
    elif cfg == "none":
        off(64, "no plane rotates in this configuration")
    elif cfg == "split_seq":
        seq(64)
    elif cfg == "split_tree":
        tree(); tok(); off(16)
    assert len(out) == 64, (cfg, len(out))
    return out


PLANE_ROWS = [
    ("seq", "sequence only"), ("graph", "tree only"), ("leaf", "tree + within-node index"),
    ("both", "sequence + tree"), ("all", "all signals"), ("depthcfg", "sequence + depth"),
    ("toplevel", "top-level-only control"), ("rel", "sequence + tree + depth"),
    ("none", "no position"),
    ("split_seq", "split: a sequence layer or head"), ("split_tree", "split: a tree layer or head"),
]


def plane_diagram() -> tuple[str, str]:
    W, LW, T, RH, CW, GAP = 900, 250, 40, 26, 8.5, 1.5
    H = T + RH * len(PLANE_ROWS) + 34
    g = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Rotary plane allocation per configuration">']
    lx = LW
    for role in ("seq", "tree", "tok", "depth", "off"):
        g.append(f'<rect x="{lx}" y="8" width="12" height="12" rx="2" fill="{ROLE_COLOR[role]}"/>')
        g.append(f'<text x="{lx + 18}" y="18" class="lg">{esc(ROLE_NAME[role])}</text>')
        lx += 30 + 7.2 * len(ROLE_NAME[role])
    for r, (cfg, label) in enumerate(PLANE_ROWS):
        y = T + RH * r
        if cfg == "split_seq":
            g.append(f'<line x1="12" x2="{W - 8}" y1="{y - 3}" y2="{y - 3}" stroke="{GRID}" stroke-width="1"/>')
        g.append(f'<text x="{LW - 12}" y="{y + 13}" class="rl" text-anchor="end">{esc(label)}</text>')
        for i, (role, desc) in enumerate(planes(cfg)):
            x = LW + i * (CW + GAP)
            g.append(f'<rect class="cell" x="{x:.1f}" y="{y}" width="{CW}" height="{RH - 8}" rx="1.5" fill="{ROLE_COLOR[role]}" '
                     f'tabindex="0" data-tip="plane {i + 1} — {esc(ROLE_NAME[role])}&#10;{esc(label)}&#10;{esc(desc)}"/>')
    yb = T + RH * len(PLANE_ROWS) + 6
    for i in (0, 15, 59, 63):
        x = LW + i * (CW + GAP) + CW / 2
        g.append(f'<text x="{x:.1f}" y="{yb + 10}" class="tk" text-anchor="middle">{i + 1}</text>')
    g.append(f'<text x="{LW + 32 * (CW + GAP):.1f}" y="{yb + 26}" class="al" text-anchor="middle">plane number within one attention head (64 planes = 128 dimensions)</text>')
    g.append("</svg>")
    rows = []
    for cfg, label in PLANE_ROWS:
        c = collections.Counter(role for role, _ in planes(cfg))
        rows.append([label] + [c.get(k, 0) for k in ("seq", "tree", "tok", "depth", "off")])
    tbl = table(["configuration", "sequence", "tree path", "within-node index", "depth", "unrotated"],
                rows, num_cols=(1, 2, 3, 4, 5))
    return "".join(g), tbl


# ====================================================================================== PROSE

def summ(c: dict, unit: str = "") -> str:
    """'−0.0095 (won 5/5, p 0.0002)' for a contrast."""
    if c.get("mean") is None:
        return "not yet available"
    tail = f"won {c['won']}/{c['n']}" + (f", p {pfmt(c['p'])}" if c.get("p") is not None else "")
    return f"{num(c['mean'])}{unit} ({tail})"


def subset(c: dict, seeds) -> dict:
    """The same contrast restricted to a subset of seeds (used to show how estimates move)."""
    d = [v for s, v in c["diffs"] if s in seeds]
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return dict(mean=m, n=len(d), won=sum(x < 0 for x in d), p=p_two_sided(m / se, len(d) - 1))


def webl_row(D: dict, key: str, label: str, sub: str = "") -> dict:
    """A one-run larger-web-run row at the final validation milestone, in Δ nats."""
    w = D["webl"]
    v = w["gaps"][key][w["final"]]
    return dict(label=label, sub=sub, pts=[(0, v)], mean=v, lo=v, hi=v, n=1, won=int(v < 0), p=None)


def sec_about(D) -> str:
    c, w, t = D["c"], D["webl"], D["webl"]["runs"]
    webl_test = t["both"]["test"][0] - t["seq"]["test"][0]
    webl_leaf_test = t["leaf"]["test"][0] - t["seq"]["test"][0]
    pr = lambda x: pct(100 * (1 - math.exp(x)), 2)
    return md(f"""
This page is the appendix to the short paper *{PAPER_TITLE}*. The paper's contribution
is a **depth marker**: two of the rotary position embedding's [[plane|planes]] rotated by each
token's *nesting depth* — how many levels of its document's [[tree]] enclose it (the syntax tree of
a program, the element tree of a web page, the Markdown of a chat message), not the network's layer
count — beside unchanged sequence position. The supporting study asks the broader question behind
it: does a transformer language model benefit from being told where each token sits in that tree
**in addition to** where it sits in the sequence? It gives part of the planes to the full tree path,
to grid coordinates and to references, and trains small language models from scratch.

The paper gives the headline. This page gives everything else: the full setup, the mechanism with
every symbol defined, every [[configuration]] we trained, every [[seed]], and the results that
did not go our way. Every result number on this page is recomputed from the training logs when the
page is built. Runs still training are marked as in progress.
""") + "<h3>The findings, in brief</h3>" + md(f"""
- **The recommendation: a [[depth]] marker, two planes.** It is a strong, cheap version of the signal on code and prose, and on grid-shaped data one more small coordinate (row and column; on chat, the speaker and position in the message) carries the rest. On web pages with their markup in the token stream, where depth can be read from the text and so is available while generating, it lowers perplexity on the visible text by about 2.5% on all ten seeds; with the markup stripped, as in standard pretraining pipelines, depth must be supplied from the source, which suits reading but not writing free text (see Generation). Sequence + depth beats sequence alone on Python {summ(c['py_depth'])} and web pages {summ(c['ws_depth'])}; on spreadsheets it helps only inside formulas, {summ(c['sh_depth'])} overall. There, [[gridcfg|sequence + grid]] — sheet, row and column on 6 planes — beats sequence only {summ(c['sh_grid'])} and ties the full 44-plane path ({summ(c['sh_grid_both'])}). Adding the node's rank among its siblings to depth — [[drank|sequence + depth + rank]], a tree's analogue of row and column — adds nothing: {summ(c['py_tg_depth'])} on Python and {summ(c['ws_tg_depth'])} on web pages against sequence + depth.
- **[[rel|Sequence + tree + depth]] is best or tied for best on Python, spreadsheets and web pages, at 46 planes.** Against sequence only: [[Python]] {pr(c['py_rel']['mean'])} ({c['py_rel']['won']}/{c['py_rel']['n']} seeds), [[Spreadsheets]] {pr(c['sh_rel']['mean'])} ({c['sh_rel']['won']}/{c['sh_rel']['n']}), [[web pages]] {pr(c['ws_rel']['mean'])} ({c['ws_rel']['won']}/{c['ws_rel']['n']}). The full path is what pays on spreadsheets and depth what pays on web text; together they win or tie against either alone.
- **Adding tree position to sequence position helps in all four [[setting|settings]], on every seed.** As a [[pplred]] on held-out text: [[Python]] {pr(c['py_both']['mean'])} ({c['py_both']['won']}/{c['py_both']['n']} seeds), [[Spreadsheets]] {pr(c['sh_both']['mean'])} ({c['sh_both']['won']}/{c['sh_both']['n']}), [[web pages]] {pr(c['ws_both']['mean'])} ({c['ws_both']['won']}/{c['ws_both']['n']}), and [[the larger web run]] {pr(webl_test)} on its test split (one run, ahead at all {w['both_summary']['n']} [[milestone|milestones]]).
- **Replacing sequence position with tree position helps on code and hurts on prose.** On Python, the tree with no sequence position beats sequence alone, {summ(c['py_leaf'])}. On the larger web run it is worse at all {w['leaf_summary']['n']} milestones and {num(webl_leaf_test)} nats per token worse on test.
- **The gain lands on the tokens that depend on structure.** On spreadsheets the end of a row improves several times more than the average token under the path or the grid, and on web pages the first token of a new text block improves about two to three times more than average (§ on where the gain lands).
- **Whether the two signals should share a head depends on the corpus.** On Python, all four ways of giving each signal its own layers lose to [[both|sequence + tree]] on every seed, and on the larger web run the layer split keeps about a third of the gain; on spreadsheets the layer split is better than sharing, {summ(c['sh_split_both'])}.
- **Pointing a [[reference]] at its target in angle helps where the text does not already spell the pointer out**: on spreadsheets {summ(c['sh_ref'])}; on Python, where a variable's use and its definition carry the same name, {summ(c['py_splice'])}.

These are small models (4.7 million and 31.5 million non-embedding parameters). We do not show
that the effect survives at scale; we show it exists at this scale, on four kinds of document,
and that it is the size such structural effects usually are.
""")


def sec_read(D) -> str:
    c = D["c"]
    sd = D["sd"]
    ws = D["webl"]["runs"]["seq"]
    return md(f"""
**The loss.** Every number is [[nats|validation nats per token]]: the average negative
log-likelihood the model assigns to each token of a fixed set of 1,024 [[split|held-out]] records,
at the run's final step. Lower is better.

**The difference.** Results are reported as [[delta|Δ]] = configuration − baseline, in nats per
token. **Negative means the configuration is better.** Every table names its baseline. Headline
numbers are also given as a [[pplred]], 100 × (1 − e^Δ), where positive means better: a Δ of
−0.01 is about a 1% lower perplexity.

**How big is big.** The sequence-only baseline sits at {num(c['py_both']['b_mean'], 4, False)}
nats per token on Python, {num(c['sh_both']['b_mean'], 4, False)} on spreadsheets,
{num(c['ws_both']['b_mean'], 4, False)} on web pages and {num(ws['val'][D['webl']['final']], 4, False)}
on the larger web run. The effects measured here are between about 0.002 and 0.025. For comparison, removing
position altogether on Python costs {num(c['py_none']['mean'])}.

**Why everything is paired.** The final loss of one configuration varies from seed to seed with a
standard deviation of {num(sd['py'][0], 4, False)} on Python ({sd['py'][1]} seeds),
{num(sd['sheets'][0], 4, False)} on spreadsheets and {num(sd['webs'][0], 4, False)} on web pages — as
large as the effects. So every configuration is trained at the same [[seed|seeds]] as its
baseline, and we analyse the [[paired|paired differences]], whose spread is far smaller (for sequence +
tree against sequence only: {num(c['py_both']['sd'], 4, False)}, {num(c['sh_both']['sd'], 4, False)}
and {num(c['ws_both']['sd'], 4, False)} respectively).

**The statistics.** For each comparison we give the mean paired difference, its standard error,
a paired t-test on the per-seed differences (n − 1 degrees of freedom, two-sided p), the 95%
interval (mean ± t × standard error), and [[won]]. Seed counts matter: if a configuration made no
difference, winning 5 of 5 seeds would still happen with probability 0.0625 (two-sided), 10 of 10
with probability 0.002 and 20 of 20 with probability 0.000002. We treat 5 of 5 as suggestive, not
conclusive, unless the t-test agrees.

**The charts.** Hover over, or tab to, any mark for its value. Every chart has a table view
underneath with every number it plots. Hover over an underlined term for its definition.
""")


def sec_glossary(D) -> str:
    items = "".join(f'<dt id="g-{esc(k)}">{esc(n)}</dt><dd>{esc(d)}</dd>' for k, (n, d) in GLOSSARY.items())
    return md("Every term below is linked wherever it appears; hover for the definition, click to "
              "come here.") + f'<dl class="gloss">{items}</dl>'


def sec_setup(D) -> str:
    cfg, cor = D["cfg"], D["corpus"]
    keys = ("py", "sheets", "webs", "webl")
    meta = {"py": cor["py"]["meta"], "sheets": cor["sheets"]["meta"], "webs": cor["webs"]["meta"],
            "webl": cor["webl"]["meta"]}
    def mean_len(k):
        m = meta[k]
        return m.get("tokens_per_example_mean") or m.get("tokens_per_record_mean") or m["total_tokens"] / m["n_records"]
    def passes(k):
        return cfg[k]["max_iters"] * cfg[k]["records_per_step"] / meta[k]["n_train"]
    seeds = {"py": "5 (20 for sequence only and tree + within-node index)", "sheets": "10",
             "webs": "10", "webl": "1"}
    corpus = {"py": "Python programs (APPS and KodCode), whole files",
              "sheets": "spreadsheet workbooks (Sheetpedia corpus), whole workbooks",
              "webs": "English web pages (C4 with HTML), visible text only",
              "webl": "English web pages (C4 with HTML), visible text only"}
    tok = {"py": f"BPE trained on the serialised trees, {intc(meta['py']['vocab_size'])}",
           "sheets": "GPT-2, 50,257 (padded to 50,304)", "webs": "GPT-2, 50,257 (padded to 50,304)",
           "webl": "GPT-2, 50,257 (padded to 50,304)"}
    longdocs = {"py": "none longer than the context", "sheets": "dropped whole",
                "webs": "dropped whole", "webl": "cropped to the first 4,096 tokens"}
    npt = {"py": cor["py"]["nodes_per_token"], "sheets": cor["sheets"]["nodes_per_token"],
           "webs": cor["webs"]["nodes_per_token"], "webl": cor["webl"]["nodes_per_token"]}
    rows = [
        ["corpus"] + [corpus[k] for k in keys],
        ["documents: train / validation / test"] + [f"{intc(meta[k]['n_train'])} / {intc(meta[k]['n_val'])} / {intc(meta[k]['n_test'])}" for k in keys],
        ["tokens in the corpus"] + [f"{meta[k]['total_tokens'] / 1e6:,.1f} M" if meta[k]['total_tokens'] < 1e9 else f"{meta[k]['total_tokens'] / 1e9:,.2f} B" for k in keys],
        ["mean tokens per document"] + [f"{mean_len(k):,.0f}" for k in keys],
        ["tokenizer, vocabulary"] + [tok[k] for k in keys],
        ["[[npt]]"] + [f"{npt[k]:.2f}" for k in keys],
        ["context length (tokens)"] + [intc(cfg[k]["block_size"]) for k in keys],
        ["documents longer than the context"] + [longdocs[k] for k in keys],
        ["layers / heads / width"] + [f"{cfg[k]['n_layer']} / {cfg[k]['n_head']} / {cfg[k]['n_embd']}" for k in keys],
        ["head width ([[plane|planes]] per head)"] + [f"{cfg[k]['n_embd'] // cfg[k]['n_head']} ({cfg[k]['n_embd'] // cfg[k]['n_head'] // 2})" for k in keys],
        ["non-embedding parameters"] + [f"{12 * cfg[k]['n_layer'] * cfg[k]['n_embd'] ** 2 / 1e6:.1f} M" for k in keys],
        ["documents per step"] + [intc(cfg[k]["records_per_step"]) for k in keys],
        ["tokens per step (approx.)"] + [f"{cfg[k]['records_per_step'] * mean_len(k) / 1e3:,.1f} k" for k in keys],
        ["steps"] + [intc(cfg[k]["max_iters"]) for k in keys],
        ["[[pass|passes]] over the training split"] + [f"{passes(k):.2f}" for k in keys],
        ["peak learning rate"] + [f"{cfg[k]['learning_rate']:g}" for k in keys],
        ["warm-up steps (2%)"] + [intc(cfg[k]["warmup_iters"]) for k in keys],
        ["[[seed|seeds]] per configuration"] + [seeds[k] for k in keys],
        ["wall-clock per run, one RTX A4000"] + ["about 2 h", "about 1.5 h", "about 0.7 h", "about 180 h"],
    ]
    settings_tbl = table(["", "[[Python]]", "[[Spreadsheets]]", "[[web pages]]", "[[the larger web run]]"], rows)

    py, sh, sha = cor["py"], cor["sheets"], cor["sheets_addr"]
    wl, wsm = cor["webl"], cor["webs"]
    wlm, wls = wl["meta"], wl["side"]
    st_ = wsm["shard_totals"]
    sh_read = sh["meta"]["n_records"] + sh["meta"]["records_dropped_too_long"] + sh["meta"]["records_dropped_deeper_than_levels"]
    ws_elig = st_["kept"] + st_["long"]
    trees = md(f"""
**Python.** Programs are parsed with Python's own syntax-tree parser and written out as a compact
[[preorder|pre-order serialisation]] of the syntax tree: a reversible text form (all
{intc(py['meta']['n_records'])} programs convert back to the original byte for byte). Local
variable names are rewritten by order of definition (x0, x1, …), so every use of a variable
carries exactly the same name as its definition. A byte-pair tokenizer with
{intc(py['meta']['vocab_size'])} entries is trained on this serialisation. Each token belongs to
the syntax-tree [[node]] that emits it, and its [[path]] is the list of [[rank|sibling ranks]] from
the root down to that node. Typical sibling ranks are small (99th percentile
{py['side']['sib_rank_p99']:.0f}, maximum {py['side']['sib_rank_max']}). {intc(py['side']['nodes_deeper_than_levels'])}
nodes ({100 * py['side']['nodes_deeper_than_levels_frac']:.3f}%) sit deeper than the 22 encoded
[[level|levels]]; nothing is dropped.

**Spreadsheets.** Each workbook is a tree: workbook → sheet → row → cell → the cell's content,
which is either a literal value or a formula's expression tree (operators, function calls,
references). The path's first three levels are the sheet's index, the **absolute** row index and
the **absolute** column index (0-based) rather than sibling ranks, because empty rows and cells are
not stored and counting siblings would misnumber them; deeper levels are sibling ranks inside the
formula. A token belongs to the deepest node that produced its first character; the tab and
newline separators belong to no node. Two [[render|renders]] of the same workbooks are used; the
table after this section shows one row of a timesheet in each.

In the values render a cell's own address is never written: the grid position reaches the model
only through the rotation. References inside formulas (the E4 and F4 of `=E4+F4`) are written in
both renders. For the tokens of `=E4+F4` in cell G4 of the first sheet, the path begins
(sheet 0, row 3, column 6) and continues with the token's place inside the formula.
{intc(sh['side']['n_refs_total'])} references occur in the corpus;
{intc(sh['side']['n_refs_with_live_target'])} name a cell in the same workbook and
{intc(sh['side']['n_refs_target_not_a_cell'])} name a defined name rather than a cell.

**Web text.** Pages come from the English C4 crawl with its original HTML. The HTML is parsed into
its element tree, skipping `head`, `script`, `style`, `svg`, `iframe`, `noscript` and `template`.
Nodes are the remaining elements and every text run containing a non-space character; paths start
at the children of `body`, and a rank counts the parent's retained children. **Only the visible
text is written into the token stream; all markup is removed**, so document structure reaches the
model only through the rotation. A token belongs to the text node containing its first non-space
byte. Because only text emits tokens, every token-carrying node is a leaf of the element tree.
""")
    render_tbl = table(
        ["render", "how one row is written (⇥ = tab)"],
        [["values", "`2022-05-01 00:00:00 ⇥ 05:00:00 ⇥ 16:00:00 ⇥ … ⇥ =E4+F4`"],
         ["addressed", "`B4:2022-05-01 00:00:00 ⇥ C4:05:00:00 ⇥ D4:16:00:00 ⇥ … ⇥ G4:=E4+F4`"]])

    excl = table(["setting", "excluded or altered", "share"], [
        ["[[Python]]", "nothing excluded; no program is longer than the context", "0%"],
        ["[[Spreadsheets]]", f"workbooks longer than 4,096 tokens in either render, dropped whole ({intc(sh['meta']['records_dropped_too_long'])} of {intc(sh_read)})",
         pct(100 * sh['meta']['records_dropped_too_long'] / sh_read)],
        ["", f"workbooks nested deeper than 22 levels, dropped ({intc(sh['meta']['records_dropped_deeper_than_levels'])})",
         pct(100 * sh['meta']['records_dropped_deeper_than_levels'] / sh_read)],
        ["[[the larger web run]]", f"pages nested deeper than 22 levels, dropped (of {wls['pages_read'] / 1e6:.2f} M pages read)", pct(100 * wls['pages_dropped_deep_frac'])],
        ["", "pages with a sibling rank above 318 (the slow plane would [[wrap]]), dropped", pct(100 * wls['pages_dropped_alias_frac'])],
        ["", "remaining pages longer than 4,096 tokens, **cropped** to their first 4,096", pct(100 * wlm['records_cropped_frac'])],
        ["", "… which discards this share of all text tokens", pct(100 * wlm['text_tokens_lost_to_crop_frac'])],
        ["[[web pages]]", f"pages nested deeper than 22 levels, dropped (of {intc(st_['read'])} pages read)", pct(100 * st_['deep'] / st_['read'])],
        ["", "pages with a sibling rank above 318, dropped", pct(100 * st_['alias'] / st_['read'], 2)],
        ["", "pages with no body or under 64 text tokens, dropped", pct(100 * (st_['no_body'] + st_['short']) / st_['read'], 2)],
        ["", f"pages longer than 2,048 tokens, dropped whole — none cropped ({intc(st_['long'])}, which is {pct(100 * st_['long'] / ws_elig)} of otherwise-eligible pages)", pct(100 * st_['long'] / st_['read'])],
    ], num_cols=(2,))

    train = md(f"""
**Model.** A decoder-only transformer in the GPT-2 style: pre-layer-norm blocks, a GELU MLP four
times the width, no bias terms, no dropout, input and output embeddings tied. The only change is
the position signal: no learned position embedding, and rotary embedding applied to queries and
keys in every head, with the angles described in the mechanism section.

**Optimisation.** AdamW (β₁ 0.9, β₂ 0.95), weight decay 0.1 on weight matrices — the rotary rate
and offset parameters are excluded from weight decay — gradient clipping at 1.0, bfloat16
autocast. The learning rate rises linearly over the first 2% of steps, then follows a cosine to 10%
of its peak at the final step.

**Batches.** Every step trains on a fixed number of whole documents. Documents are grouped by
length into micro-batches of at most 8,192 padded tokens and gradients are accumulated across the
micro-batches of a step. No document is split across steps.

**Learning rates.** One peak rate per setting, used for every configuration in it:

- [[Python]]: a sweep of five rates (0.0003 to 0.0048, 560 steps each) on four configurations — [[seq|sequence only]], [[graph|tree only]], [[leaf|tree + within-node index]] and [[both|sequence + tree]]. All four were best at 0.0012.
- [[web pages]]: a sweep of three rates (600 steps) on sequence + tree; 0.0012 was best.
- [[Spreadsheets]]: 0.0012, carried over from a sweep on an earlier build of the spreadsheet corpus with a shorter context, not re-swept for this one.
- [[the larger web run]]: 0.0006, from a sweep on an earlier build of the web corpus. The exception is [[none|no position]], which diverged at 0.0006 and 0.0003 in preliminary runs and was run at 0.00015 — where it diverged again.

**Evaluation.** Validation loss is scored every 5% of a run's planned steps (a [[milestone]]) on
the same 1,024 validation documents. Results use the final step: no checkpoint is selected by its
score. The single exception is the the larger web run tree-only run, which diverged and is reported at its best
milestone, labelled as such. As an independent check, final checkpoints are also scored on the
test split: 8,192 test documents for Python (and 2,048 in an earlier pass over six seeds), 2,048
for the larger web run.

**Compilation.** Runs use PyTorch's compiler except every web pages run (documents of many different
lengths would trigger a recompilation per batch shape) and the the larger web run head-split run (it failed
under compilation at four heads). Compilation does not change the arithmetic in principle but is
not a bit-identical path, so the the larger web run head-split run is not configured exactly like its
comparators.
""")
    return (f'<h3 id="setup-settings">The four settings</h3>{settings_tbl}'
            + md("""Only configurations within one setting are compared. The two web settings
answer the same question at different cost: The larger web run at a realistic context and model size, which
buys one seed per configuration; Web pages ten times over, on whole pages, for a fraction of one the larger web run
run. Their losses are on different scales and are never compared directly; only the direction and
stability of the effects are compared.""")
            + f'<h3 id="setup-trees">How each corpus becomes a tree</h3>{trees}{render_tbl}'
            + f'<h3 id="setup-exclusions">What is excluded</h3>{excl}'
            + md("""Two different things happen to over-long web pages. The larger web run **crops** them, keeping
the first 4,096 tokens; Web pages **drops** them, so every web pages page is seen whole. Separately, both
web settings drop pages whose tree is deeper than the 22 encoded levels or wider than 318 siblings
at some level, so that no path is cut short or wraps round.""")
            + f'<h3 id="setup-training">Model, training and evaluation</h3>{train}')


def sec_protocol(D) -> str:
    c = D["c"]
    five = subset(c["sh_depth"], range(5))
    w = D["webl"]
    t2, t8 = D["test2048"], D["test8192"]
    return md(f"""
These are rules we follow throughout; each was learned from a measurement in this study.

- **Compare runs only at the same step.** A run still training sits far higher on its loss curve than a finished one, and that gap dwarfs every effect here. Every comparison on this page pairs runs at the same step and refuses otherwise.
- **Adjacent milestones of one run are not independent evidence.** On the larger web run the sequence + tree gap moves by about ±{num(w['both_summary']['succ_sd'], 4, False)} nats per token from one milestone to the next with no trend. A few consecutive milestones cannot establish a trend in a gap of about 0.02.
- **Five seeds are often not enough.** Seed-to-seed spread is as large as the effects. On spreadsheets, sequence + depth against sequence only reads {num(five['mean'])} (won {five['won']}/5, p {pfmt(five['p'])}) on the first five seeds and {num(c['sh_depth']['mean'])} (won {c['sh_depth']['won']}/{c['sh_depth']['n']}, p {pfmt(c['sh_depth']['p'])}) on ten. We report the seed count with every number and do not treat 5 of 5 alone as conclusive.
- **A test-split score means nothing without its document count.** Scoring the same Python checkpoints on 8,192 instead of 2,048 test documents moved the absolute loss by about {num(t8['leaf_mean'] - t2['leaf_mean'], 3)} nats per token, several times the effects being measured. Paired differences survive such shifts; absolute levels do not.
- **Learning rates were chosen on the validation split** we then report on. The selection was the same for every configuration within a setting, and the test split, never used for selection, agrees in sign (see the held-out check in the results).
""")


def sec_mechanism(D) -> str:
    svg, tbl = plane_diagram()
    fig = figure("fig-planes", "How each configuration spends the 64 rotary planes of one attention head",
                 svg, """Each row is a configuration and each small block one [[plane]]; colour says
what the plane's angle is computed from. Hover over (or tab to) a block for its formula. The last
two rows are the per-layer or per-head allocations used by the [[layersplit|layer split]] and
[[headsplit|head split]], in which each layer or head carries one signal at full width. Grey planes
are [[unrotated]].""", tbl)
    return md("""
**Rotary position embedding in one paragraph.** In each attention head, the query and key vectors
are cut into pairs of dimensions, and each pair — a [[plane]] — is rotated by an angle that depends
on the token's position. When the attention score between two tokens is computed, the rotations
combine so that each plane contributes a term depending only on the **difference** between the two
tokens' angles. Standard rotary embedding makes every angle a multiple of the token's index in the
sequence, at a range of frequencies. Nothing requires that: the angle of any plane can be a function
of any coordinate, and the score will then depend on the difference in that coordinate. We give
some planes to the token's position in the [[tree]].
""") + fig + md("""
**The angles.** A head of width 128 has 64 planes, numbered here 1 to 64 (plane i rotates dimensions
i and i + 64). In every configuration that keeps all signals in each head, the same planes always
carry the same signal; a configuration only decides which groups rotate.

- **Sequence planes 1–16.** Plane i (i = 0 … 15) has angle t × 10000^(−i/16), where t is the token's index in its document. This is the standard rotary spectrum over 16 planes, the partial-rotary arrangement used by GPT-NeoX.
- **Tree planes 17–60: 22 levels × 2 planes.** For tree level k (1 to 22) and plane p ∈ {fast, slow}, the angle is present_k × θ_k,p + r_k × s_k,p. Here r_k is the [[rank|sibling rank]] of the token's ancestor at level k; present_k is 1 if the token's node reaches level k and 0 otherwise; θ_k,p (an offset) and s_k,p (a rate) are learned. Levels a node does not reach contribute angle 0.
- **Within-node planes 61–64.** Angle = present × θ_p + w × s_p, where w is the token's [[within|index inside its own node]] and θ_p, s_p are learned.
- **Depth planes (2, in the depth configurations).** Angle = d × s_p, where d is the token's [[depth]] (the number of levels its node reaches) and s_p is learned. There is no offset: every token has a depth.

**Initial values.** All offsets θ start at exactly 0. Each tree level's [[fastslow|fast plane]] starts
with a rate of √(2 × (1 − 0.95)) ≈ 0.316 radians per sibling — chosen so that adjacent siblings
have a cosine similarity of 0.95 on that plane — and its slow plane with 1/16 of that, so the slow
plane counts up to about 318 siblings before its angle [[wrap|wraps]] round and the fast plane about 20.
The four within-node planes start at 0.316 / 4^p (p = 0 … 3), wrapping after about 20, 80, 318 and
1,270 tokens; the two depth planes at 0.316 and 0.316 / 16. All rates and offsets are then learned.

**Properties that matter for the comparisons.**

- **Identical networks at a given seed.** The rotary parameters are initialised to fixed values, not drawn at random, so two configurations at the same seed start with identical weights everywhere else and see identical batches.
- **No look-ahead.** A rank counts earlier siblings only, and a token's node and path depend only on the document up to that token.
- **Which plane carries which signal is arbitrary.** For a model trained from scratch, permuting the planes is the same as permuting rows of the query and key weight matrices, so only the set of (coordinate, rate) pairs matters, not where in the head they sit.
""") + figure("fig-pathdepth", "What the tree path puts in the attention score, and what sequence + depth has to infer",
           _static_svg("fig_path_vs_depth"),
           """Tokens A and B sit in different paragraphs of one section; C sits in the next section at
the same depth. Each dial is one plane group's rotation (angles exaggerated). Under
[[both|sequence + tree]], B differs from A on the paragraph level and C on the section level. Under
[[depthcfg|sequence + depth]] all three are at depth 3, so the score sees only how far apart they
are; the branch can still be inferred, since a token's parent is the nearest earlier token one level
shallower, but only where ancestors are themselves tokens (bottom): true of our Python corpus, not of
web text, whose stream holds only the visible text. The path costs 44 planes; depth costs 2. That
depth nonetheless beats the path on web text means it helps as a feature of where text sits, not by
rebuilding the tree.""", table(["note"], [["schematic, not data"]])) + figure(
           "fig-refdiagram", "Reference rotation on a three-cell sheet", _static_svg("fig_reference"),
           """Cell A3 holds =A1+A2. Its token "A1" (F) belongs to A3, so under
[[both|sequence + tree]] it differs from the cell it names (R) on the row level. Under
[[refrot|reference rotation]] it takes A1's coordinates and matches R on every level. Angles
exaggerated.""", table(["note"], [["schematic, not data"]])) + "<h3 id=\"mech-variants\">The configurations beyond the plane allocation</h3>" + md(f"""
**[[depthcfg|Sequence + depth]] and its control.** Two planes carry the token's depth and nothing
else of the tree. Its [[plctrl|plane-matched control]], the [[toplevel|top-level-only]]
configuration, has exactly the same 16 + 2 rotating planes and the same two learned rates, but
spends them on level 1 of the path (which child of the root the token descends from) instead of on
depth.

**Depth and the tree.** In a [[preorder|pre-order]] serialisation, a node's parent is the nearest
preceding node one level shallower, so the sequence order plus every token's depth determines the
tree exactly — **provided every ancestor appears in the token stream**. On Python this holds and
was checked directly: reconstructing every node's parent from order and depth alone recovered
{intc(CORPUS_CONST['py_parents_checked'])} parents with no error. On web text it cannot hold:
only text nodes emit tokens, so no ancestor of a token-carrying node is ever in the stream.

**[[rel|Sequence + tree + depth]].** Sequence + tree plus the two depth planes. Because
depth goes up by exactly one from parent to child, on those planes a parent and its child always
differ by exactly one learned step, with a sign: the relation "is the parent of" becomes a constant
angular offset, and a grandparent is two steps away.

**[[refrot|Reference rotation]] on spreadsheets.** A token that is part of a reference (the E4 in
`=E4+F4`) has its sheet, row and column levels replaced by those of the cell it names; its deeper
levels, its place inside the formula, are kept. A reference and the cell it names then have
identical angles on those planes, exactly as two tokens of one cell do. It rotates the same planes
as sequence + tree, which is its baseline, on the same data and seeds. The coordinate is computed
from the reference's own text, so it uses no future information even when the named cell comes
later in the workbook. {CORPUS_CONST['sheets_cross_sheet_ref_pct']}% of references point to another
sheet, which is why the sheet level is replaced too.

**Reference rotation on Python.** A use of a variable takes the path of the variable's definition
for the levels the definition occupies, and keeps its own path below that, so path length is
unchanged. {CORPUS_CONST['splice_tokens_pct']}% of tokens are affected; every definition precedes
its uses. Baseline: sequence + tree.

**Variants of reference rotation**, reported with the results:

- *Whole-path overwrite* (Python): a use takes its definition's entire path. Definitions sit shallower than uses, so {CORPUS_CONST['anchor_truncated_pct']}% of the overwritten paths become shorter, losing {CORPUS_CONST['anchor_levels_lost']} levels on average.
- *[[delta_off|Directed offset]]* (spreadsheets): the reference sits a learned constant angle away from its target rather than exactly on it, so "refers to" is distinguishable from "is in the same cell", with a direction. The constant starts at 0, so the variant starts identical to plain reference rotation.
- *Own plane group* (spreadsheets, an earlier build): references rotate a separate group of planes rather than the tree planes.
- *Symbol identity* (Python): two planes rotated by the order in which each variable is first mentioned, so all mentions of one variable coincide.

**[[layersplit|Layer splits]] and the [[headsplit|head split]].** Instead of sharing every head,
each signal gets whole layers or whole heads at full width: a sequence layer rotates all 64 planes
by sequence position (frequencies 10000^(−i/64)), and a tree layer rotates 44 tree planes plus the 4
within-node planes, leaving 16 unrotated. On Python's six layers four patterns are trained, written
one letter per layer (s sequence, t tree, n no rotation): **alternating** ststst,
**sequence-first** sssttt, **tree-first** tttsss and **thirds** ssttnn. On the larger web run (ten layers) and
spreadsheets the layer split is alternating. The head split gives half the heads of every layer
the sequence and the other half the tree: one head each in Python's two-head model, two each in
the larger web run's four-head model.
""")


def stat_cells(c: dict) -> list:
    if c.get("mean") is None:
        return ["—"] * 6
    return [c["n"], num(c["mean"]), num(c["se"], 4, False) if c.get("se") else "—",
            num(c["t"], 2) if c.get("t") is not None else "—", pfmt(c.get("p")), f"{c['won']}/{c['n']}"]


STAT_HEAD = ["n", "Δ", "se", "t", "p", "won"]


def res_headline(D) -> str:
    c, w = D["c"], D["webl"]
    runs = w["runs"]
    rows = [ppl_row(c["py_both"], "Python", "5 seeds"),
            ppl_row(c["sh_both"], "Spreadsheets, values render", "10 seeds"),
            ppl_row(c["sha_both"], "Spreadsheets, addressed render", "10 seeds"),
            ppl_row(c["ws_both"], "web pages", "10 seeds"),
            to_ppl(webl_row(D, "both", "the larger web run", "one run, final milestone"))]
    fmtp = lambda v: pct(v, 2)
    fig = figure("fig-headline", "Adding tree position to sequence position: perplexity reduction on validation text",
                 dot_chart(rows, axis="perplexity reduction (%) from adding tree position", left="worse with the tree",
                           right="better with the tree", fmt=fmtp),
                 """Each dot is one [[seed]]: a pair of runs identical except that one adds the
tree planes to the sequence planes. Its position is 100 × (1 − e^Δ) for that pair; **right of zero
means the tree helped**. The black tick is the mean, the black line its 95% interval. Validation
split, final step. The larger web run is a single pair of runs, so it has no interval. Loss levels differ
between settings, which is why this chart uses perplexity rather than nats.""",
                 dot_table(rows, fmt=fmtp, unit="% reduction"))
    trows = []
    for key, label in (("seq", "[[seq|sequence only]]"), ("both", "[[both|sequence + tree]]"),
                       ("leaf", "[[leaf|tree + within-node index]]"), ("layer_split", "[[layersplit|layer split]] (ststst…)")):
        r = runs[key]
        v = r["val"].get(w["final"])
        tt = r["test"]
        trows.append([label, num(v, 4, False) if v else f"in progress ({pct(100 * r['last'] / w['final'], 0)} of steps)",
                      num(tt[0], 4, False) if tt else "not yet scored",
                      "—" if key == "seq" or not tt else num(tt[0] - runs["seq"]["test"][0])])
    tbl = table(["the larger web run configuration", "validation, final", "test, final (2,048 documents)", "test Δ vs sequence only"],
                trows, num_cols=(1, 2, 3))
    hs = [["[[Python]]", "sequence only", *stat_cells(c["py_both"]), pct(c["py_both"]["ppl"], 2)],
          ["[[Spreadsheets]], values", "sequence only", *stat_cells(c["sh_both"]), pct(c["sh_both"]["ppl"], 2)],
          ["[[Spreadsheets]], addressed", "sequence only", *stat_cells(c["sha_both"]), pct(c["sha_both"]["ppl"], 2)],
          ["[[web pages]]", "sequence only", *stat_cells(c["ws_both"]), pct(c["ws_both"]["ppl"], 2)]]
    htbl = table(["setting", "baseline"] + STAT_HEAD + ["[[pplred|perplexity reduction]]"], hs, num_cols=(2, 3, 4, 5, 6, 7, 8))
    bs = w["both_summary"]
    return fig + md("**[[both|Sequence + tree]] against [[seq|sequence only]], validation split, "
                    "final step:**") + htbl + md(f"""
**Every seed in every setting favours the tree.** On the larger web run, a single pair of runs, sequence + tree
is ahead at {bs['neg']} of {bs['n']} milestones, by {num(bs['mean'])} on average (range
{num(bs['hi'])} to {num(bs['lo'])}); the gap does not shrink late in training — the last five
milestones read {', '.join(num(x) for x in bs['last5'])}. The test split, scored once at the end,
agrees:
""") + tbl


def res_recommended(D) -> str:
    """Sequence + tree + depth against sequence only and against its two simpler parts."""
    c = D["c"]
    rows = []
    for st_, lab, n in (("py", "Python", "5 seeds"), ("sh", "Spreadsheets", "10 seeds"), ("ws", "web pages", "10 seeds")):
        both = c["py_both"] if st_ == "py" else c[f"{st_}_both"]
        depth = c["py_depth"] if st_ == "py" else c[f"{st_}_depth"]
        rel = c["py_rel"] if st_ == "py" else c[f"{st_}_rel"]
        rows += [ppl_row(both, f"{lab}: sequence + tree", n), ppl_row(depth, f"{lab}: sequence + depth", n),
                 ppl_row(rel, f"{lab}: sequence + tree + depth", n)]
    fmtp = lambda v: pct(v, 2)
    fig = figure("fig-recommended", "Three structured configurations against sequence only, per setting",
                 dot_chart(rows, axis="perplexity reduction (%) against sequence only", left="worse",
                           right="better", fmt=fmtp),
                 """Each dot is one [[seed]], paired with [[seq|sequence only]] at the same seed;
right of zero is better. The path pays on spreadsheets, depth on web text, and
[[rel|sequence + tree + depth]] keeps whichever pays.""", dot_table(rows, fmt=fmtp, unit="% reduction"))
    hs = []
    for lab, k_seq, k_both, k_depth in (("[[Python]]", "py_rel", "py_rel_both", "py_rel_depth"),
                                        ("[[Spreadsheets]]", "sh_rel", "sh_rel_both", "sh_rel_depth"),
                                        ("[[web pages]]", "ws_rel", "ws_rel_both", "ws_rel_depth")):
        hs.append([lab, summ(c[k_seq]), summ(c[k_both]), summ(c[k_depth])])
    tbl = table(["setting", "vs sequence only", "vs sequence + tree", "vs sequence + depth"], hs)
    return fig + md("**[[rel|Sequence + tree + depth]] minus each baseline, validation nats/token, "
                    "final step (won = seeds where sequence + tree + depth is better):**") + tbl + md(f"""
**It is the one structured configuration that is best or tied for best in every setting.** Each
simpler part fails somewhere: sequence + depth barely helps on spreadsheets, and sequence + tree loses
to sequence + depth on web text. It ties [[both|sequence + tree]] on spreadsheets and [[depthcfg|sequence + depth]]
on web pages, and its edge over sequence + tree on Python is consistent but not significant at five seeds.

**It is not the best spreadsheet result.** [[refrot|Reference rotation]] beats it there
({summ(c['sh_rel_ref'])} for sequence + tree + depth against reference rotation), and so does a
[[layersplit|layer split]] ({summ(c['sh_rel_split'])}). Combining depth with either is untested.
The larger web run has no sequence + tree + depth run.
""")


def _static_svg(name: str) -> str:
    """A figure produced by ``make_figures.py``, inlined so the page stays self-contained.

    Run ``make_figures.py`` first: the appendix embeds whatever is in docs/figures, and the
    figure script is what reads the logs for these two charts.
    """
    import re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "figures", f"{name}.svg")
    svg = open(path).read()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"', r'<svg\1', svg, count=1)
    return svg.replace("<svg", '<svg style="width:100%;height:auto" role="img"', 1)


# ------------------------------------------------------------ interactive loss curves (#r-curves)
#
# One chart per setting, built from the run logs at every validation milestone. Python computes
# every plotted number (means, bands, per-seed lines) so the page's script only draws; the same
# numbers fill each chart's table view. NO RAGGED RUNS: a configuration contributes only seeds that
# reached the setting's final step, and the gap view pairs each with sequence only at the SAME seed
# and the SAME step.

IC_SETTINGS = [
    ("py", "Python", "5 seeds (sequence only and tree only: 20)", [
        ("depth", "depth"), ("drank", "drank"), ("stmtgrid", "stmtgrid"),
        ("stmtgrid_depth", "stmtgrid_depth"), ("rope_both", "both"), ("rel", "rel"),
        ("full_rope", "full"), ("rope_leaf", "leaf"), ("none", "none")], "rope_seq"),
    ("sheets", "Spreadsheets", "10 seeds", [
        ("depth_v2", "depth"), ("grid_v2", "grid"), ("both_v2", "both"), ("rel_v2", "rel"),
        ("full_rope_v2", "full")], "seq_v2"),
    ("webc", "web pages (short pages)", "10 seeds", [
        ("depth", "depth"), ("drank", "drank"), ("both", "both"), ("rel", "rel"),
        ("full_rope", "full")], "seq"),
    ("chat", "Chat (WildChat)", "10 seeds", [
        ("depth", "depth"), ("drank", "drank"), ("grid", "cgrid"), ("grid_depth", "cgrid_depth"),
        ("both", "both"), ("rel", "rel")], "seq"),
    ("web7", "the larger web run (long run)", "1 seed per configuration", [
        ("rope_depth", "depth"), ("rope_both", "both"), ("rope_leaf", "leaf")], "rope_seq"),
]


def _ic_styles():
    """Configuration -> (display name, colour, SVG dash). Colours follow the paper's figures."""
    import make_figures as mf

    def dash(d):
        if d in ("-", None):
            return ""
        if d == "--":
            return "6 4"
        return " ".join(str(round(v * 2, 1)) for v in d[1])
    s = {k: (v[0], v[1], dash(v[3])) for k, v in mf.CURVE_STYLE.items()}
    s["stmtgrid"] = ("statement grid", mf.S2, "2 3")
    s["stmtgrid_depth"] = ("statement grid + depth", mf.S2, "")
    s["cgrid"] = ("sequence + grid (exchange, speaker, position)", mf.S2, "")
    s["cgrid_depth"] = ("grid + Markdown depth", mf.S2, "8 3")
    s["seq"] = ("sequence only", mf.INK2, "")
    return s


def _ic_setting(root: str, arms: list, seq_prefix: str) -> dict:
    import make_figures as mf
    fin = mf.FINAL[root]
    styles = _ic_styles()

    def finished(prefix):
        return {s: c for s, c in mf.run_curves(root, prefix).items() if max(c) == fin}
    base = finished(seq_prefix)
    if not base:
        raise SystemExit(f"interactive curves: {root} has no finished sequence-only run")
    r4 = lambda v: round(v, 4)
    tc = {}

    def tcrit(df):
        if df not in tc:
            tc[df] = t_crit95(df)
        return tc[df]
    out = {"title": "", "final": fin, "series": []}
    for prefix, key in [(seq_prefix, "seq")] + list(arms):
        runs = base if key == "seq" else finished(prefix)
        if not runs:
            continue
        steps = sorted(set.intersection(*(set(c) for c in runs.values())))
        raw_seeds = {s: [r4(c[t]) for t in steps] for s, c in sorted(runs.items())}
        raw_mean = [r4(st.mean(c[t] for c in runs.values())) for t in steps]
        raw_lo = [r4(min(c[t] for c in runs.values())) for t in steps]
        raw_hi = [r4(max(c[t] for c in runs.values())) for t in steps]
        ser = dict(key=key, name=styles[key][0], color=styles[key][1], dash=styles[key][2],
                   n=len(runs), x=[r4(100 * t / fin) for t in steps], step=steps,
                   raw=dict(mean=raw_mean, lo=raw_lo, hi=raw_hi, seeds=list(raw_seeds.values())))
        if key != "seq":
            pair = sorted(set(runs) & set(base))
            gsteps = [t for t in steps if all(t in base[s] for s in pair)]
            if pair and gsteps:
                per = {s: [100 * (1 - math.exp(runs[s][t] - base[s][t])) for t in gsteps] for s in pair}
                n = len(pair)
                gm, glo, ghi = [], [], []
                for i in range(len(gsteps)):
                    v = [per[s][i] for s in pair]
                    m = st.mean(v)
                    h = tcrit(n - 1) * st.stdev(v) / math.sqrt(n) if n > 1 else 0.0
                    gm.append(r4(m)); glo.append(r4(m - h)); ghi.append(r4(m + h))
                ser["gap"] = dict(n=n, x=[r4(100 * t / fin) for t in gsteps], step=gsteps,
                                  mean=gm, lo=glo, hi=ghi,
                                  seeds=[[r4(v) for v in per[s]] for s in pair])
        out["series"].append(ser)
    return out


def t_crit95(df: int) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_two_sided(mid, df) > 0.05 else (lo, mid)
    return (lo + hi) / 2


def _ic_table(d: dict) -> str:
    """The plotted means at every milestone: gap view first, raw loss second."""
    gs = [s for s in d["series"] if "gap" in s]
    xs = sorted({x for s in gs for x in s["gap"]["x"]})
    rows = []
    for x in xs:
        r = [f"{x:.0f}%"]
        for s in gs:
            g = s["gap"]
            r.append(f"{g['mean'][g['x'].index(x)]:+.2f}%" if x in g["x"] else "")
        rows.append(r)
    t1 = table(["% of training"] + [f"{s['name']} (n={s['gap']['n']})" for s in gs], rows,
               num_cols=tuple(range(1, len(gs) + 1)))
    ss = d["series"]
    xs = sorted({x for s in ss for x in s["x"]})
    rows = []
    for x in xs:
        r = [f"{x:.0f}%"]
        for s in ss:
            r.append(f"{s['raw']['mean'][s['x'].index(x)]:.4f}" if x in s["x"] else "")
        rows.append(r)
    t2 = table(["% of training"] + [f"{s['name']} (n={s['n']})" for s in ss], rows,
               num_cols=tuple(range(1, len(ss) + 1)))
    return ("<p><strong>Perplexity reduction against sequence only</strong> (mean over paired seeds; "
            "positive = better)</p>" + t1 +
            "<p><strong>Raw validation loss</strong> (nats per token, mean over seeds)</p>" + t2)


def interactive_curves() -> str:
    data, blocks = {}, []
    for root, name, seeds_note, arms, seq_prefix in IC_SETTINGS:
        d = _ic_setting(root, arms, seq_prefix)
        d["title"] = f"{name}, {seeds_note}"
        data[root] = d
        cid = f"ic-{root}"
        chips = "".join(
            f'<button type="button" class="chip" aria-pressed="true" data-key="{esc(s["key"])}">'
            f'<svg class="sw" viewBox="0 0 18 6" aria-hidden="true"><line x1="1" x2="17" y1="3" y2="3" stroke="{s["color"]}" stroke-width="{4 if s["key"] == "depth" else 2.5}" stroke-linecap="round"{(' stroke-dasharray="' + s["dash"] + '"') if s["dash"] else ""}/></svg>{esc(s["name"])}</button>'
            for s in d["series"] if s["key"] != "seq")
        blocks.append(
            f'<figure class="fig ic" id="{cid}" data-setting="{root}">'
            f'<figcaption class="ft">{esc(d["title"])}</figcaption>'
            f'<div class="icbar" role="group" aria-label="View">'
            f'<button type="button" class="vbtn" aria-pressed="true" data-view="gap">gap to sequence only (perplexity %)</button>'
            f'<button type="button" class="vbtn" aria-pressed="false" data-view="raw">raw validation loss</button>'
            f'<label class="seedopt"><input type="checkbox" class="seedbox"> show individual seeds</label></div>'
            f'<div class="icchips" role="group" aria-label="Configurations (click to show or hide)">{chips}</div>'
            f'<div class="icplot"></div>'
            f'<details class="tv"><summary>Table view</summary>{_ic_table(d)}</details></figure>')
    payload = json.dumps(data, separators=(",", ":"))
    note = md("""Each chart is drawn in your browser from the numbers embedded in this page, computed
from the training logs at every validation milestone (every 5% of training). **Gap view** (default):
each configuration's perplexity reduction against [[seq|sequence only]] at the same seed and step,
100·(1 − e^Δ); above zero is better, the line is the mean over paired seeds and the band its 95%
confidence interval. **Raw view**: validation loss in nats per token, mean over seeds, the band the
seed range. Only seeds that finished their run are drawn. Click a configuration to hide or show it;
hover (or focus the plot and use the arrow keys) to read every visible configuration at one milestone.
[[depth|Sequence + depth]], the paper's recommendation, is the heavy line. No position and tree only
start hidden (they sit far below the rest); their buttons turn them on.""")
    return (note + "".join(blocks) +
            f'<script type="application/json" id="ic-data">{payload}</script>')


IC_CSS = """
.ic .icbar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:6px 0; }
.ic .vbtn, .ic .chip { font:inherit; font-size:13px; border:1px solid var(--axis); background:var(--surface);
  color:var(--ink); border-radius:999px; padding:3px 11px; cursor:pointer; }
.ic .vbtn[aria-pressed="true"] { background:var(--ink); color:var(--surface); border-color:var(--ink); }
.ic .chip { display:inline-flex; align-items:center; gap:6px; }
.ic .chip[aria-pressed="false"] { color:var(--muted); text-decoration:line-through; }
.ic .chip[aria-pressed="false"] .sw { opacity:.25; }
.ic .vbtn:focus-visible, .ic .chip:focus-visible, .ic .icplot svg:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.ic .sw { width:18px; height:6px; display:inline-block; }
.ic .icchips { display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 8px; }
.ic .seedopt { font-size:13px; color:var(--ink2); margin-left:8px; }
.ic .icplot svg { width:100%; height:auto; display:block; }
.ic .icplot text { font-size:12px; fill:var(--ink2); }
.ic .icplot .lab { font-size:12px; fill:var(--ink); paint-order:stroke; stroke:var(--surface); stroke-width:3px; }
"""

IC_JS = r"""
(function () {
  const el = document.getElementById('ic-data');
  if (!el) return;
  const DATA = JSON.parse(el.textContent);
  const tip = document.getElementById('tip');
  const NS = 'http://www.w3.org/2000/svg';
  const W = 860, H = 380, L = 64, R = 210, T = 16, B = 46;
  function mk(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function ticks(lo, hi, n) {
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    let step = mag * 10;
    for (const m of [1, 2, 2.5, 5, 10]) { if (m * mag >= raw) { step = m * mag; break; } }
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return {ticks: out, step: step};
  }
  function fmt(v, step) {
    const d = Math.max(0, -Math.floor(Math.log10(step)) + (step / Math.pow(10, Math.floor(Math.log10(step))) === 2.5 ? 1 : 0));
    return v.toFixed(d).replace('-', '−');
  }
  function setup(fig) {
    const d = DATA[fig.dataset.setting];
    // side points (no position, tree only) start hidden: they sit far below the rest and are not
    // what the chart is for; their chips turn them on
    const st = {view: 'gap', seeds: false, hidden: new Set(['none', 'leaf']), cur: null};
    fig.querySelectorAll('.chip').forEach(function (c) { if (st.hidden.has(c.dataset.key)) c.setAttribute('aria-pressed', 'false'); });
    const plot = fig.querySelector('.icplot');
    function visible() {
      return d.series.filter(function (s) {
        if (st.hidden.has(s.key)) return false;
        return st.view === 'raw' ? true : (s.key !== 'seq' && s.gap);
      });
    }
    function block(s) { return st.view === 'raw' ? Object.assign({x: s.x, step: s.step, n: s.n}, s.raw) : s.gap; }
    function draw() {
      plot.textContent = '';
      const ser = visible();
      const svg = mk('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', tabindex: '0',
        'aria-label': d.title + ': ' + (st.view === 'gap' ? 'perplexity reduction against sequence only' : 'raw validation loss') + ' over training'}, plot);
      // the y range ignores the first milestones (x < 25%), whose seed-to-seed swings would
      // flatten the rest; anything off scale there is clipped and the axis says so
      let lo = Infinity, hi = -Infinity, offscale = false;
      ser.forEach(function (s) {
        const b = block(s);
        b.x.forEach(function (x, i) {
          const v = st.seeds ? b.seeds.map(function (sd) { return sd[i]; }) : [b.lo[i], b.hi[i], b.mean[i]];
          if (x >= 25) v.forEach(function (u) { lo = Math.min(lo, u); hi = Math.max(hi, u); });
        });
      });
      ser.forEach(function (s) {
        const b = block(s);
        b.x.forEach(function (x, i) {
          const v = st.seeds ? b.seeds.map(function (sd) { return sd[i]; }) : [b.mean[i]];
          if (x < 25 && x >= (st.view === 'raw' ? 50 : 0)) v.forEach(function (u) { if (u < lo - (hi - lo) * 0.1 || u > hi + (hi - lo) * 0.1) offscale = true; });
        });
      });
      if (st.view === 'gap') { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      // raw loss falls by several nats while the gaps are ~1%: show the second half of training on
      // a tight axis, where the configurations separate (the gap view shows the whole run)
      const XMIN = st.view === 'raw' ? 50 : 0;
      if (st.view === 'raw') {
        const late = [];
        ser.forEach(function (s) { const b = block(s); b.x.forEach(function (x, i) {
          if (x >= XMIN) { if (st.seeds) b.seeds.forEach(function (sd) { late.push(sd[i]); }); else late.push(b.hi[i], b.lo[i], b.mean[i]); } }); });
        if (late.length) { lo = Math.min.apply(null, late); hi = Math.max.apply(null, late); }
      }
      const pad = (hi - lo) * 0.08 || 0.1; lo -= pad; hi += pad;
      const tk = ticks(lo, hi, 6);
      lo = Math.min(lo, tk.ticks[0]); hi = Math.max(hi, tk.ticks[tk.ticks.length - 1]);
      const x0 = L, x1 = W - R, y0 = T, y1 = H - B;
      const X = function (x) { return x0 + (x - XMIN) / (100 - XMIN) * (x1 - x0); };
      const Y = function (v) { return y1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (y1 - y0); };
      tk.ticks.forEach(function (v) {
        mk('line', {x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: 'var(--grid)', 'stroke-width': 1}, svg);
        mk('text', {x: x0 - 8, y: Y(v) + 4, 'text-anchor': 'end'}, svg).textContent = fmt(v, tk.step) + (st.view === 'gap' ? '%' : '');
      });
      (XMIN ? [50, 60, 70, 80, 90, 100] : [0, 25, 50, 75, 100]).forEach(function (p) {
        mk('text', {x: X(p), y: y1 + 18, 'text-anchor': 'middle'}, svg).textContent = p + '%';
      });
      mk('text', {x: (x0 + x1) / 2, y: y1 + 38, 'text-anchor': 'middle'}, svg).textContent = XMIN ? '% of training (second half; the gap view shows the whole run)' : ('% of training' + (offscale ? ' (first milestones partly off scale)' : ''));
      mk('line', {x1: x0, x2: x1, y1: y1, y2: y1, stroke: 'var(--axis)'}, svg);
      if (st.view === 'gap' && lo < 0 && hi > 0) {
        mk('line', {x1: x0, x2: x1, y1: Y(0), y2: Y(0), stroke: 'var(--ink2)', 'stroke-width': 1}, svg);
        mk('text', {x: x0 + 4, y: Y(0) - 5}, svg).textContent = 'sequence only';
      }
      const clip = 'icclip-' + fig.id;
      mk('rect', {x: x0, y: y0, width: x1 - x0, height: y1 - y0}, mk('clipPath', {id: clip}, mk('defs', {}, svg)));
      const g = mk('g', {'clip-path': 'url(#' + clip + ')'}, svg);
      const ends = [];
      ser.slice().sort(function (a, b) { return (a.key === 'depth') - (b.key === 'depth'); }).forEach(function (s) {
        const b = block(s);
        if (st.seeds) {
          b.seeds.forEach(function (sd) {
            mk('path', {d: sd.map(function (v, i) { return (i ? 'L' : 'M') + X(b.x[i]).toFixed(1) + ',' + Y(v).toFixed(1); }).join(''),
              fill: 'none', stroke: s.color, 'stroke-width': 1, 'stroke-opacity': 0.35}, g);
          });
        } else if (b.lo && b.hi && (b.n > 1)) {
          const top = b.hi.map(function (v, i) { return X(b.x[i]).toFixed(1) + ',' + Y(v).toFixed(1); });
          const bot = b.lo.map(function (v, i) { return X(b.x[i]).toFixed(1) + ',' + Y(v).toFixed(1); }).reverse();
          mk('path', {d: 'M' + top.join('L') + 'L' + bot.join('L') + 'Z', fill: s.color, 'fill-opacity': 0.13, stroke: 'none'}, g);
        }
        mk('path', {d: b.mean.map(function (v, i) { return (i ? 'L' : 'M') + X(b.x[i]).toFixed(1) + ',' + Y(v).toFixed(1); }).join(''),
          fill: 'none', stroke: s.color, 'stroke-width': s.key === 'depth' ? 3.4 : 2, 'stroke-dasharray': s.dash || null,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, g);
        ends.push({s: s, y: Y(b.mean[b.mean.length - 1]), x: X(b.x[b.x.length - 1])});
      });
      // direct labels at the line ends, nudged apart so they never overlap
      ends.sort(function (a, b) { return a.y - b.y; });
      for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 15) ends[k].y = ends[k - 1].y + 15;
      if (ends.length && ends[ends.length - 1].y > y1 + 4) {
        const sh = ends[ends.length - 1].y - (y1 + 4);
        ends.forEach(function (e) { e.y -= sh; });
      }
      ends.forEach(function (e) {
        mk('circle', {cx: e.x, cy: Math.min(Math.max(e.y, y0), y1), r: 3, fill: e.s.color}, svg);
        mk('text', {x: x1 + 10, y: e.y + 4, class: 'lab', 'font-weight': e.s.key === 'depth' ? 700 : 400}, svg).textContent = e.s.name.replace(' (exchange, speaker, position)', '');
      });
      // hover layer: crosshair snapped to milestones
      const xs = Array.from(new Set([].concat.apply([], ser.map(function (s) { return block(s).x; })))).filter(function (x) { return x >= XMIN; }).sort(function (a, b) { return a - b; });
      const xh = mk('line', {y1: y0, y2: y1, stroke: 'var(--ink)', 'stroke-width': 1, visibility: 'hidden'}, svg);
      const hit = mk('rect', {x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: 'transparent'}, svg);
      function readout(i, cx, cy) {
        if (i == null || !xs.length) return;
        st.cur = i;
        const x = xs[i];
        xh.setAttribute('x1', X(x)); xh.setAttribute('x2', X(x)); xh.setAttribute('visibility', 'visible');
        tip.textContent = '';
        let stepv = null;
        const rows = [];
        ser.forEach(function (s) {
          const b = block(s), j = b.x.indexOf(x);
          if (j < 0) return;
          stepv = b.step[j];
          const v = b.mean[j];
          rows.push([v, s.name + ': ' + (st.view === 'gap' ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) + '%' : v.toFixed(4)) + '  (n=' + b.n + ')']);
        });
        rows.sort(function (a, b) { return st.view === 'gap' ? b[0] - a[0] : a[0] - b[0]; });
        const h = document.createElement('div'); h.className = 't1';
        h.textContent = x.toFixed(0) + '% of training' + (stepv != null ? ' (step ' + stepv.toLocaleString() + ')' : '');
        tip.appendChild(h);
        rows.forEach(function (r) { const e = document.createElement('div'); e.textContent = r[1]; tip.appendChild(e); });
        tip.hidden = false;
        const rr = tip.getBoundingClientRect();
        let Lp = cx + 14, Tp = cy + 16;
        if (Lp + rr.width > innerWidth - 8) Lp = Math.max(8, cx - rr.width - 14);
        if (Tp + rr.height > innerHeight - 8) Tp = Math.max(8, cy - rr.height - 14);
        tip.style.left = Lp + 'px'; tip.style.top = Tp + 'px';
      }
      function nearest(clientX) {
        const box = svg.getBoundingClientRect();
        const vx = (clientX - box.left) / box.width * W;
        let best = 0, bd = Infinity;
        xs.forEach(function (x, i) { const dd = Math.abs(X(x) - vx); if (dd < bd) { bd = dd; best = i; } });
        return best;
      }
      function off() { xh.setAttribute('visibility', 'hidden'); tip.hidden = true; }
      hit.addEventListener('pointermove', function (e) { readout(nearest(e.clientX), e.clientX, e.clientY); });
      hit.addEventListener('pointerleave', off);
      svg.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        e.preventDefault();
        let i = st.cur == null ? xs.length - 1 : st.cur + (e.key === 'ArrowRight' ? 1 : -1);
        i = Math.max(0, Math.min(xs.length - 1, i));
        const box = svg.getBoundingClientRect();
        readout(i, box.left + X(xs[i]) / W * box.width, box.top + box.height * 0.2);
      });
      svg.addEventListener('blur', off);
    }
    fig.querySelectorAll('.vbtn').forEach(function (b) {
      b.addEventListener('click', function () {
        st.view = b.dataset.view;
        fig.querySelectorAll('.vbtn').forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
        draw();
      });
    });
    fig.querySelector('.seedbox').addEventListener('change', function (e) { st.seeds = e.target.checked; draw(); });
    fig.querySelectorAll('.chip').forEach(function (c) {
      c.addEventListener('click', function () {
        const on = c.getAttribute('aria-pressed') === 'true';
        c.setAttribute('aria-pressed', on ? 'false' : 'true');
        if (on) st.hidden.add(c.dataset.key); else st.hidden.delete(c.dataset.key);
        draw();
      });
    });
    draw();
  }
  document.querySelectorAll('figure.ic').forEach(setup);
})();
"""


def res_curves(D) -> str:
    """Every setting over training, interactive: gap to sequence only and raw loss, per seed."""
    return interactive_curves()


def res_fullrope(D) -> str:
    """The 16-plane baseline against all 64 planes rotated by sequence."""
    c, P = D["c"], D["prog"]
    rows, txt = [], []
    for k, lab in (("py_full", "Python"), ("sh_full", "Spreadsheets"), ("ws_full", "web pages")):
        done, n, _ = P[k]
        if c[k].get("mean") is not None and c[k]["n"] > 1:
            rows.append(crow(c[k], lab, f"{c[k]['n']} finished seeds"))
        txt.append(f"{lab}: {summ(c[k])} ({done} of {n} seeds finished)")
    fig = ""
    if rows:
        fig = figure("fig-fullrope", "Full-width RoPE (all 64 planes by sequence) against the 16-plane baseline",
                     dot_chart(rows, axis="Δ validation nats per token, full-width − 16-plane",
                               left="full width better", right="16 planes better"),
                     """One dot per finished seed, paired with [[seq|sequence only]] (16 planes) at the
same seed. Runs still training are left out and listed below.""", dot_table(rows))
    return md("""
Every baseline here rotates 16 of the head's 64 planes by sequence position — the 25% partial
rotation GPT-NeoX and Pythia use — while the structured configurations rotate up to 62. The 16
frequencies are exactly every fourth of standard RoPE's 64 (base 10,000), so the baseline covers
the same wavelengths more coarsely and keeps 48 planes free of position. To check that the
structured configurations do not win simply by rotating more planes, a control rotates all 64
planes by sequence position, otherwise identical to sequence only.
""") + fig + md("- " + "\n- ".join(txt) + "\n\nThis section fills in as the remaining seeds finish.")


CHAT_ARMS_APPX = [("grid", "sequence + grid", "8 planes"),
                  ("grid_depth", "grid + Markdown depth", "10 planes"),
                  ("depth", "sequence + depth", "2 planes"),
                  ("drank", "sequence + depth + rank", "4 planes"),
                  ("both", "sequence + tree", "44 planes"), ("rel", "sequence + tree + depth", "46 planes")]


def res_chat(D) -> str:
    """Chat (WildChat-1M): seven configurations x 10 seeds, generated from the logs as seeds land."""
    c = D["c"]
    m = json.load(open(f"{PACKS_ROOT}/chat_wildchat_b2048/meta.json"))
    prog = {a: progress("chat", a, SETTINGS["chat"]["final"], planned=10) for a in ["seq"] + [k for k, _, _ in CHAT_ARMS_APPX]}
    rows = [ppl_row(c[f"ch_{a}"], lab, f"{sub}; {c[f'ch_{a}']['n']} seed(s)")
            for a, lab, sub in CHAT_ARMS_APPX if c[f"ch_{a}"].get("mean") is not None]
    over = [ppl_row(c[k], lab) for k, lab in (("ch_grid_vs_depth", "grid against depth"),
                                               ("ch_gd_vs_depth", "grid + depth against depth"),
                                               ("ch_gd_vs_grid", "grid + depth against grid"),
                                               ("ch_drank_vs_depth", "depth + rank against depth"))
            if c[k].get("mean") is not None]
    nmin = min((c[f"ch_{a}"]["n"] for a, _, _ in CHAT_ARMS_APPX), default=0)
    out = [md(f"""
**The data.** WildChat-1M (Allen AI; ODC-BY): real users' conversations with ChatGPT. English and
non-toxic only; each conversation is one record, kept whole — a conversation longer than 2,048
tokens is dropped, never cut, and so is one containing a single message over 1,272 tokens (the
point at which the within-message position would wrap). Of {m['conversations_read']:,} conversations
read, {m['n_records']:,} are kept ({m['n_train']:,} train, {m['n_val']:,} validation, {m['n_test']:,} test,
split by conversation); {m['total_tokens'] / 1e6:.0f} M tokens, {m['tokens_per_record_mean']:.0f} per
conversation on average. Training runs {SETTINGS['chat']['final']:,} steps of 32 conversations, about 1.4
passes over the training set; model, context and learning rate are web pages's. **No conversation text
is reproduced anywhere in this work**, since it is real users' data; only aggregate statistics are.

**Who is speaking is explicit.** Every message starts on its own line with `User:` or `Assistant:`
(each always the same two GPT-2 tokens), messages are separated by a blank line, and in the tree
every message is its own node: conversation → exchange (a user message and its reply) → speaker
(0 user, 1 assistant) → the message's Markdown blocks (paragraphs, headings with the blocks below
them nested under them, lists, list items, nested lists, code blocks, quotes, tables). Every
conversation kept alternates strictly; users write about a quarter of the tokens. Most tokens (77%)
sit at depth 3, a top-level block of a message; about a fifth sit deeper, in lists and under headings.

**Two chat coordinates.** The *grid* gives each token (exchange number, speaker, position within its
message): the first two on 2 planes each, the position on 4, the other tree levels empty. *Markdown
depth* is the token's nesting in that tree. The grid's inputs need nothing but what a chat template
already holds; depth needs each message's Markdown parsed.
"""),
           md(f"**Status.** {nmin} or more of 10 seeds finished for every configuration; per configuration: "
              + "; ".join(f"{a} {prog[a][0]}/10" for a in prog) + ". Figures and tables fill in as seeds land.")]
    if rows:
        out.append(figure("fig-chat", "Chat: every configuration against sequence only",
                          dot_chart(rows, axis="perplexity reduction against sequence only, %",
                                    left="sequence only better", right="configuration better", fmt=lambda v: pct(v, 2)),
                          """Each dot is one finished seed, paired with [[seq|sequence only]] at the same seed;
marker and line, the mean and its 95% interval (shown from two seeds; with few seeds it is wide).""",
                          dot_table(rows, fmt=lambda v: pct(v, 2), unit="perplexity reduction")))
    if over:
        out.append(figure("fig-chat-overlap", "Chat: do grid and depth carry the same information?",
                          dot_chart(over, axis="perplexity reduction, first configuration against the second, %",
                                    left="second better", right="first better", fmt=lambda v: pct(v, 2)),
                          """Paired at the seeds both configurations have finished. If the grid and depth
carried different information, grid + depth would beat both; a gap near zero between grid and depth, and
little added by combining them, is what overlap looks like.""", dot_table(over, fmt=lambda v: pct(v, 2), unit="perplexity reduction")))
    out.append(md(f"""
**Reading.** Depth alone does not help on chat ({summ(c['ch_depth'])} against sequence only): three
quarters of the tokens sit at one depth, so it varies only inside lists and headed sections. The grid
alone is suggestive ({summ(c['ch_grid'])}); grid + depth is the cheapest configuration that clearly
beats sequence only ({summ(c['ch_grid_depth'])}), ahead of the grid alone on most seeds
({summ(c['ch_gd_vs_grid'])}) and tied with the full tree ({summ(c['ch_gd_vs_both'])}). An earlier,
interim reading of this section — that grid and depth carry the same information — did not survive
the full ten seeds: depth adds to the grid but does nothing alone. The grid says who is speaking and
how far into the message; depth, how deep in the message's formatting.
"""))
    return "\n".join(out)


def res_generation(D) -> str:
    """Generation: Web pages depth models fed predicted / fixed depth instead of the true depth (gen_depth_eval.py)."""
    import glob as _g
    runs = [json.load(open(f)) for f in sorted(_g.glob(f"{RUNS}/analysis/gen_depth/seed*.json"))]
    if not runs:
        return md("*The predicted-depth evaluation has not been run on this machine.*")
    labels = [("true", "true depth, from the page's DOM (the paper's web pages setting)"),
              ("pred", "depth predicted from the text so far, fed back causally"),
              ("pred_onpolicy", "predicted, head also trained on its own predictions"),
              ("stale", "previous text block's true depth (one block late)"),
              ("const", "one fixed depth for every token (training median)")]
    rows = []
    for a, lab in labels:
        d = [r["nats"][a] - r["nats"]["seq"] for r in runs]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        t = m / (sd / math.sqrt(len(d)))
        rows.append([lab, pct(100 * (1 - math.exp(m)), 2), f"{sum(x < 0 for x in d)}/{len(d)}",
                     pfmt(p_two_sided(t, len(d) - 1))])
    acc = sum(r["depth_accuracy"]["pred"]["exact"] for r in runs) / len(runs)
    w1 = sum(r["depth_accuracy"]["pred"]["within1"] for r in runs) / len(runs)
    mae = sum(r["depth_accuracy"]["pred"]["mae"] for r in runs) / len(runs)
    med = sum(r["depth_accuracy"]["const"]["exact"] for r in runs) / len(runs)
    tf = sum(r["head_acc_teacher_forced"] for r in runs) / len(runs)
    tok = runs[0]["tokens"]
    return md(f"""
**The question.** When a model writes, each new token needs its depth before it is fed back in.
Where the structure is written in the text (code, Markdown, JSON, HTML, chat roles) depth follows
from what has been generated: every coordinate here depends only on earlier tokens. The tags-kept
web control (§ Controls) is that case for web pages, and there depth still adds its gain. Where the
markup is stripped, as in standard pretraining pipelines and in our web pages and the larger web run settings, nothing
in the text says how deep the next block sits, so depth must be predicted.

**The test.** For each of the ten web pages depth models, a small two-layer head on the frozen model's
final hidden state is trained (4,096 training pages, true depths fed in) to predict the next text
block's depth at the newline that ends the previous block. On {runs[0]['n_val']:,} validation pages
({tok:,} tokens) the model is then run block by block with its own predicted depths fed back, exactly
as during generation, and scored against the sequence-only model at the same seed. No language model
is retrained.
""") + table(["depth fed to the model", "perplexity reduction vs sequence only", "seeds better", "p"], rows) + md(f"""

**Depth is not predictable from visible text.** Fed back causally the head is exact {pct(100 * acc, 0)}
of the time (within one level {pct(100 * w1, 0)}, mean error {mae:.1f} levels); always guessing the
training median is exact {pct(100 * med, 0)}. Even given the true depths of all earlier blocks it is
exact only {pct(100 * tf, 0)}.

**A model trained with depth relies on it.** Wrong depths make it much worse than sequence position
alone, and no depth information at all (a fixed value) still leaves it worse: the gain comes from
knowing each new block's depth as the block starts. On stripped text the depth marker therefore helps
a model *read* documents whose structure is available, not *write* free text. A model that must also
handle text without structure should be trained with depth withheld on part of the data (untested
here) or without depth.
""")


def res_controls(D) -> str:
    """The referee's objection: is the gain information the baseline lacks, or its delivery?"""
    c, P = D["c"], D["prog"]
    groups = [
        ("Addresses written out (spreadsheets)", [("ad_grid", "sequence + grid vs sequence only", "ad_grid"),
                                                  ("ad_grid_tree", "sequence + grid vs sequence + tree", "ad_grid"),
                                                  ("ad_depth", "sequence + depth vs sequence only", "ad_depth")]),
        ("Depth as an input embedding", [("py_emb", "Python: vs sequence only", "py_emb"),
                                         ("py_emb_rot", "Python: vs rotary depth", "py_emb"),
                                         ("sh_emb", "Spreadsheets: vs sequence only", "sh_emb"),
                                         ("sh_emb_rot", "Spreadsheets: vs rotary depth", "sh_emb"),
                                         ("ws_emb", "web pages: vs sequence only", "ws_emb"),
                                         ("ws_emb_rot", "web pages: vs rotary depth", "ws_emb")]),
        ("Shuffled depth (same two planes, depths permuted within each record)",
         [("py_shuf", "Python: vs sequence only", "py_shuf"), ("ws_shuf", "web pages: vs sequence only", "ws_shuf"),
          ("py_shuf_rot", "Python: vs true depth", "py_shuf"), ("ws_shuf_rot", "web pages: vs true depth", "ws_shuf")]),
    ]
    rows, lines = [], []
    for title, items in groups:
        lines.append(f"**{title}.**")
        for k, lab, pk in items:
            done, n, _ = P[pk]
            if c[k].get("mean") is not None and c[k]["n"] > 1:
                rows.append(crow(c[k], lab, f"{c[k]['n']} paired seeds"))
            lines.append(f"- {lab}: {summ(c[k])} ({done} of {n} seeds finished)")
    lines.append("**Web pages with HTML tags kept** (loss on the visible-text tokens only, the same pages "
                 "and tokens for every model; 10 seeds each):")
    lines.append("")
    for split in ("val", "test"):
        fp = f"{RUNS}/analysis/visible_text/compare_{split}.md"
        if os.path.exists(fp):
            lines.append(open(fp).read().replace("# Visible-text loss", f"*{split} split*: visible-text loss"))
            lines.append("")
    fig = ""
    if rows:
        fig = figure("fig-controls", "Controls: information or delivery?",
                     dot_chart(rows, axis="Δ validation nats per token (first − second)",
                               left="first better", right="second better"),
                     """One dot per finished seed. Negative favours the first configuration named in the
row.""", dot_table(rows))
    return md("""
A referee asked whether the gains come from information the baseline lacks (our web and spreadsheet
baselines see no tags or addresses) rather than from delivering structure through the rotation.
Four controls separate the two. **Addresses written out**: the spreadsheet rendering that writes
every cell's address as text, so the baseline can read the grid. **Depth as an input embedding**:
the same depth values as a learned, zero-initialised table added to the token embedding, beside
16-plane sequence RoPE; at initialisation the model is bit-identical to sequence only. **Shuffled
depth**: the same two rotary planes, with each record's depths permuted among its in-tree nodes, so
the planes rotate but carry no information about which token is where. **HTML kept**: the same
pages with their tags (no attributes) in the token stream.
""") + fig + md("\n".join(lines) + """

**Reading.** Writing the structure into the text does not give the baseline the gain: the tagged
baseline matches the text-only one, and the address-reading baseline still loses to the grid. The
rotary coordinate helps on top of the written structure. Shuffling depth removes the whole gain, so
the extra rotation carries nothing by itself. On web text the rotary form beats the same depth as an
input embedding by about three times the embedding's own gain; on code they tie.""")


def res_python(D) -> str:
    c = D["c"]
    spec = [("py_rel", "sequence + tree + depth", "16 seq + 44 tree + 2 depth"),
            ("py_both", "sequence + tree", "16 seq + 44 tree"),
            ("py_all", "all signals", "16 seq + 44 tree + 4 within-node"),
            ("py_depth", "sequence + depth", "16 seq + 2 depth"),
            ("py_alt", "layer split, alternating", "ststst"),
            ("py_head", "head split", "1 sequence head, 1 tree head"),
            ("py_seqfirst", "layer split, sequence-first", "sssttt"),
            ("py_leaf", "tree + within-node index", "no sequence; 20 seeds"),
            ("py_thirds", "layer split, thirds", "ssttnn (n = no rotation)"),
            ("py_treefirst", "layer split, tree-first", "tttsss"),
            ("py_graph", "tree only", "no sequence"),
            ("py_top", "top-level-only control", "16 seq + 2 (level 1 only)")]
    rows = [crow(c[k], lab, sub) for k, lab, sub in spec]
    fig = figure("fig-python", "Python: every configuration against sequence only",
                 dot_chart(rows, axis="Δ validation nats per token, configuration − sequence only",
                           left="better than sequence only", right="worse"),
                 f"""One dot per seed, paired with the sequence-only run at the same seed; tick =
mean, line = 95% interval. [[none|No position]] is left off the chart because it would compress
the rest: it is {summ(c['py_none'])}.""", dot_table(rows))
    dec = table(["step", "", *STAT_HEAD], [
        ["1. replace sequence by the tree", "sequence only → tree + within-node index", *stat_cells(c["py_leaf"])],
        ["2. add sequence back", "tree + within-node index → all signals", *stat_cells(c["py_all_leaf"])],
        ["end to end", "sequence only → all signals", *stat_cells(c["py_all"])],
    ], num_cols=(2, 3, 4, 5, 6, 7))
    gap12 = c["py_all_leaf"]["mean"] - c["py_leaf"]["mean"]
    share = c["py_both"]["mean"] / -c["py_none"]["mean"]
    return fig + md(f"""
**Tree position beats sequence position on code, and the two add.** [[graph|Tree only]] — no
sequence planes, no within-node index — beats sequence only by {summ(c['py_graph'])}. That is the
claim in its plainest form, against a baseline that already encodes relative position. Adding the
tree to sequence position is worth {summ(c['py_both'])}.

The comparison decomposes into two steps of similar size:
""") + dec + md(f"""
Replacing sequence position with the tree buys one step; putting sequence position back buys a
second of similar size. The two steps differ by {num(abs(gap12), 4, False)}, well inside the second
step's standard error of {num(c['py_all_leaf']['se'], 4, False)}, so they are similar; we have not
shown them equal, and the second rests on five seeds. (The steps use different seed sets, which is
why they do not sum exactly to the end-to-end figure.) On Python, then, the tree behaves like a
second position signal of roughly the same value as the sequence index, rather than a weaker
substitute for it.

**A yardstick.** Removing position altogether costs {num(c['py_none']['mean'])} relative to
sequence only. Adding the tree on top of sequence position recovers a further
{num(-c['py_both']['mean'], 4, False)}, about {pct(100 * share, 0)} of what sequence position
itself is worth over nothing. (No-position runs trained stably on Python; on web text they
diverged, so this yardstick exists only here.)

**Position is needed somewhere, not everywhere.** The thirds pattern (ssttnn) gives its last two
layers no positional signal at all and still beats sequence only, {summ(c['py_thirds'])} — a
small-scale echo of the published finding that some position-free layers do no harm.
""")


def res_web(D) -> str:
    w, c = D["webl"], D["c"]
    runs, g = w["runs"], w["gaps"]
    fin = w["final"]
    ls_done = runs["layer_split"]["last"] >= fin
    series = [dict(label="sequence + depth", color=C_AQUA, pts=g["depth"]),
              dict(label="sequence + tree", color=C_BLUE, pts=g["both"]),
              dict(label="tree + within-node index", color=C_ORANGE, pts=g["leaf"]),
              dict(label="layer split", color=C_ORANGE, pts=g["layer_split"], dash="5 3",
                   end="" if ls_done else " (running)"),
              dict(label="head split", color=C_VIOLET, pts=g["head_split"], end=" (stopped)")]
    steps = sorted(set().union(*[s["pts"].keys() for s in series]))
    trows = [[f"{round(100 * k / fin)}%", intc(k)] + [num(s["pts"][k]) if k in s["pts"] else "—" for s in series] for k in steps]
    tbl = table(["milestone", "step"] + [s["label"] for s in series], trows, num_cols=(1, 2, 3, 4, 5, 6))
    fig = figure("fig-webl", "the larger web run: gap to sequence only at every milestone (one run per configuration)",
                 curve_chart(series, fin, axis="share of the planned training steps",
                             left="below 0: better than sequence only", right="above 0: worse"),
                 """Validation nats per token of each configuration minus sequence only's, at the
same step (y axis), across the run (x axis). Hover anywhere in a column to read every configuration
at that milestone. One seed each. The head split was stopped at 30% of the run; the layer split's
line ends where the run currently is. Tree only and no position diverged and are listed below
instead — their gaps (above +0.17) would flatten everything else.""", tbl)
    div = []
    for key, label, lr in (("graph", "tree only", "0.0006"), ("none", "no position", "0.00015")):
        gg = g[key]
        div.append([label, lr, ", ".join(f"{round(100 * k / fin)}%: {num(v, 3)}" for k, v in sorted(gg.items())),
                    intc(runs[key]["last"])])
    divt = table(["configuration", "learning rate", "gap to sequence only by milestone", "stopped at step"], div)
    graph_best = min(runs["graph"]["val"].items(), key=lambda kv: kv[1])
    ls = w["leaf_summary"]
    leaf_test = runs["leaf"]["test"][0] - runs["seq"]["test"][0]
    npt = D["corpus"]
    return fig + md(f"""
**Adding the tree helps on prose; replacing sequence position with it does not.** Sequence + tree
is ahead of sequence only throughout (above). [[leaf|Tree + within-node index]] — the configuration
that beat sequence only on all 20 Python seeds — is **behind** at all {ls['pos']} of {ls['n']}
milestones on the larger web run: the gap narrows from {num(ls['first'], 4)} at the first milestone to
{num(ls['final'], 4)} at the last, and is {num(leaf_test)} on the test split. On Python the same
replacement gains {num(c['py_leaf']['mean'])}. The sign of replacing sequence position flips with the
corpus; the sign of adding the tree does not.

**A measured reason.** An abstract syntax tree supplies {npt['py']['nodes_per_token']:.2f} [[node|nodes]]
per token: almost every token has its own node, so the tree nearly fixes the token order. A web
page supplies {npt['webl']['nodes_per_token']:.2f}: one node covers a whole run of text and says
nothing about the order of words inside it. Without a sequence signal, the model cannot order the
words of a paragraph; with one, the tree is a useful coarse supplement. The one earlier full-path
tree encoding applied to web pages that we know of (Rousselet, 2021; see the references) found it
worse than no position at all — but it added the path to the input embedding and replaced sequence
position, which is the combination our results also find harmful on prose.

**The configurations without any sequence signal were unstable on prose.** Tree only diverged at
the learning rate every other the larger web run configuration trained at: its validation loss reached its best,
{num(graph_best[1], 4, False)} at step {intc(graph_best[0])}, then rose, and the run was stopped.
No position diverged at every learning rate tried. Tree + within-node index — tree only plus the
within-node index, which counts tokens in reading order — trained stably for the full run. On this
evidence a model needs some local ordering signal to train stably on prose, and the tree alone does
not provide one. Tree only is the one result on this page read at its best milestone rather than
its last:
""") + divt


def res_within(D) -> str:
    c = D["c"]
    t = table(["question", "comparison", *STAT_HEAD], [
        ["within-node index added to sequence + tree", "all signals − sequence + tree", *stat_cells(c["py_all_both"])],
        ["within-node index added to tree only", "tree + within-node index − tree only", *stat_cells(c["py_leaf_graph"])],
    ], num_cols=(2, 3, 4, 5, 6, 7))
    return t + md(f"""
**The [[within|within-node index]] helps only when sequence position is absent.** With sequence
position present it is worth nothing ({num(c['py_all_both']['mean'])}, the seeds split
{c['py_all_both']['won']} to {c['py_all_both']['n'] - c['py_all_both']['won']}). Without it, it
helps on every seed, though at five seeds the t-test is not decisive (p {pfmt(c['py_leaf_graph']['p'])}).
The reason is that the index counts tokens in reading order within a node: it is a local copy of
sequence position, a [[surrogate|sequence surrogate]], not a tree coordinate. In the configurations
that keep sequence position its four planes are free for other use.
""")


def res_splits(D) -> str:
    c, w = D["c"], D["webl"]
    spec = [("py_alt_both", "layer split, alternating", "ststst"),
            ("py_head_both", "head split", "1 + 1 heads"),
            ("py_seqfirst_both", "layer split, sequence-first", "sssttt"),
            ("py_thirds_both", "layer split, thirds", "ssttnn"),
            ("py_treefirst_both", "layer split, tree-first", "tttsss")]
    rows = [crow(c[k], lab, sub) for k, lab, sub in spec]
    fig = figure("fig-splits", "Python: giving each signal its own layers or heads, against sequence + tree",
                 dot_chart(rows, axis="Δ validation nats per token, configuration − sequence + tree",
                           left="better than sequence + tree", right="worse than sequence + tree"),
                 """One dot per seed, paired with sequence + tree (both signals in every head) at
the same seed. Right of zero means splitting the signals apart did worse.""", dot_table(rows))
    keep = []
    for k, lab in (("py_alt", "alternating (ststst)"), ("py_head", "head split"), ("py_seqfirst", "sequence-first (sssttt)"),
                   ("py_thirds", "thirds (ssttnn)"), ("py_treefirst", "tree-first (tttsss)")):
        keep.append([lab, num(c[k]["mean"]), num(c["py_both"]["mean"]), pct(100 * c[k]["mean"] / c["py_both"]["mean"], 0)])
    keept = table(["Python configuration", "Δ vs sequence only", "sequence + tree vs sequence only", "share of the sequence + tree gain kept"],
                  keep, num_cols=(1, 2, 3))
    fin = w["final"]
    lsh, hsh = w["layer_split_share"], w["head_split_share"]
    ls_steps = sorted(lsh)
    half = [lsh[s] for s in ls_steps if s <= fin // 2]
    last5 = [lsh[s] for s in ls_steps[-5:]]
    ls_last = ls_steps[-1]
    prog_done, prog_n, prog_running = D["prog"]["sh_split"]
    sh = c["sh_split_both"]
    sh_seq = c["sh_split_seq"]
    if sh.get("mean") is not None and sh["n"] >= 1:
        both_same = st.mean(v for s, v in c["sh_both"]["diffs"] if s in {x for x, _ in sh["diffs"]})
        sh_txt = (f"**Spreadsheets: in progress.** {prog_done} of {prog_n} seeds of the alternating "
                  f"layer split have finished. On those {sh['n']}, it is {num(sh['mean'])} against "
                  f"sequence + tree (won {sh['won']}/{sh['n']}) and {num(sh_seq['mean'])} against sequence "
                  f"only, where sequence + tree is {num(both_same)} on the same seeds. This is too few "
                  f"seeds to read; it is shown because it is the one setting where the layer split "
                  f"is not, so far, behind.")
    else:
        sh_txt = f"**Spreadsheets: in progress** ({prog_done} of {prog_n} seeds finished)."
    if prog_done == prog_n and sh.get("p") is not None:
        sh_txt = (f"**Spreadsheets.** The alternating layer split against sequence + tree: "
                  f"{summ(sh)}; against sequence only {summ(sh_seq)}.")
    lay = [c[k] for k in ("py_alt_both", "py_seqfirst_both", "py_thirds_both", "py_treefirst_both")]
    lay_n, lay_lost = sum(x["n"] for x in lay), sum(x["n"] - x["won"] for x in lay)
    hb = c["py_head_both"]
    return fig + md(f"""
**On Python every way of splitting the signals apart loses to mixing them in every head** — the
layer splits on {lay_lost} of {lay_n} paired seeds, the head split on {hb['n'] - hb['won']} of
{hb['n']}. Every split configuration
still beats sequence only (they are all real position schemes); they are worse ways of spending
the same planes. Mixing more finely does better: inside every head, then alternating layers, then
contiguous blocks.
""") + keept + md(f"""
**Resolution is not the explanation.** In sequence + tree, sequence position gets 16 planes and the
tree 44. In the layer split each signal gets a whole layer: 64 sequence planes, or 44 tree planes
plus 4 within-node planes. Every signal has **more** angular resolution in the split, and it still
loses — {summ(c['py_alt_both'])} against sequence + tree, and {summ(c['py_alt_all'])} against
[[all|all signals]], which has exactly the same signal content. What the split removes is
[[colocation|co-location]]: no single head can form an attention score that depends on sequence
distance and tree position at once ("near me in the text **and** inside my subtree"). Information
still flows between layers through the residual stream; it is the joint query that is lost.

**Which unit matters — the head or the layer — is not settled.** On Python, the head split, which
puts both signals in every layer but never in the same head, is indistinguishable from the
alternating layer split: the difference is {summ(c['py_head_alt'])}. That would say the head is
the unit. But on
the larger web run, the head split kept {pct(100 * min(hsh.values()), 0)}–{pct(100 * max(hsh.values()), 0)} of
sequence + tree's margin at each of its {len(hsh)} milestones before it was stopped (one seed, the
first {round(100 * max(hsh) / fin)}% of the run), far more than the layer split. The two settings
differ in head count — one head per signal on Python, two on the larger web run — which could produce exactly
this pattern, and the the larger web run head-split run was not compiled while its comparators were. We
report the disagreement rather than choose between the readings.

**On the larger web run the layer split keeps a growing share of the margin.** It is ahead of sequence only at
{sum(1 for s in ls_steps if w['gaps']['layer_split'][s] < 0)} of {len(ls_steps)} milestones so far.
As a share of sequence + tree's margin at the same milestone, it averages {pct(100 * st.mean(half), 0)}
over the first half of the run and {pct(100 * st.mean(last5), 0)} over its last five milestones
(latest, at {round(100 * ls_last / fin)}% of the run: {pct(100 * lsh[ls_last], 0)}). On Python the
same split keeps {pct(100 * c['py_alt']['mean'] / c['py_both']['mean'], 0)}. Separating the signals
costs more on prose, where they carry different information, than on code, where the tree nearly
determines the order.

{sh_txt}
""")


def rel_text(D) -> str:
    """Status of sequence + tree + depth on Spreadsheets and web pages, from the logs."""
    parts = []
    for key, pkey, name in (("sh_rel_both", "sh_rel", "Spreadsheets"), ("ws_rel_both", "ws_rel", "web pages")):
        done, n, _ = D["prog"][pkey]
        r = D["c"][key]
        if done >= n and r.get("p") is not None:
            parts.append(f"on {name} it is {summ(r)} against sequence + tree")
        else:
            parts.append(f"on {name} it is in progress ({done} of {n} seeds finished)")
    return ("The same comparison beyond Python: " + "; ".join(parts) + ". Results are filled in "
            "here when every seed has finished.")


def res_depth(D) -> str:
    c, T, cor = D["c"], D["tree"], D["corpus"]
    rows = [crow(c["py_depth"], "Python: vs sequence only", "5 seeds"),
            crow(c["py_depth_both"], "Python: vs full path", "sequence + tree, 5 seeds"),
            crow(c["py_depth_top"], "Python: vs control", "plane-matched, 5 seeds"),
            crow(c["sh_depth"], "Spreadsheets: vs sequence only", "10 seeds"),
            crow(c["sh_depth_both"], "Spreadsheets: vs full path", "sequence + tree, 10 seeds"),
            crow(c["sh_depth_top"], "Spreadsheets: vs control", "plane-matched, 10 seeds"),
            crow(c["ws_depth"], "web pages: vs sequence only", "10 seeds"),
            crow(c["ws_depth_both"], "web pages: vs full path", "sequence + tree, 10 seeds")]
    fig = figure("fig-depth", "Two planes of depth: sequence + depth against three baselines, per setting",
                 dot_chart(rows, axis="Δ validation nats per token, sequence + depth − baseline",
                           left="depth better", right="baseline better"),
                 """Each row pairs [[depthcfg|sequence + depth]] with one baseline at the same seeds.
**Full path** is sequence + tree (44 tree planes); the **plane-matched control** has the same
16 + 2 rotating planes spent on level 1 of the path instead of depth. Web pages has no control run. The
the larger web run sequence + depth run (one seed) is in the curves section and the paper: it ends 1.68% lower in
test perplexity than sequence only, 78% of sequence + tree's gain.""", dot_table(rows))
    webl_depth = D["webl"]["runs"]["depth"]["last"] / D["webl"]["final"]
    ex = []
    for k, lab, n_struct, seqp in (("py_depth", "sequence + depth", 2, 16), ("py_top", "top-level-only control", 2, 16),
                                   ("py_rel", "sequence + tree + depth", 46, 16), ("py_both", "sequence + tree", 44, 16),
                                   ("py_all", "all signals", 48, 16), ("py_leaf", "tree + within-node index", 48, 0),
                                   ("py_graph", "tree only", 44, 0)):
        ex.append([lab, seqp, n_struct, num(c[k]["mean"]), num(c[k]["mean"] / n_struct, 5)])
    ext = table(["Python configuration", "sequence planes", "other rotating planes", "Δ vs sequence only", "Δ per other plane"],
                ex, num_cols=(1, 2, 3, 4))
    rec = table(["", "[[Python]]", "[[Spreadsheets]] (values)", "[[the larger web run]]", "[[web pages]]"], [
        ["[[npt|nodes per token]] (whole corpus)"] + [f"{cor[k]['nodes_per_token']:.2f}" for k in ("py", "sheets", "webl", "webs")],
        ["token-carrying nodes that are [[interior]]"] + [pct(100 * T[k]["interior"]) for k in ("py", "sheets", "webl", "webs")],
        ["does order + depth recover the tree?", "yes (checked: no errors)", "only inside a cell's formula", "no", "no"],
        ["entropy of depth, H(depth), bits"] + [f"{T[k]['H_depth']:.2f}" for k in ("py", "sheets", "webl", "webs")],
        ["entropy of the path, H(path), bits"] + [f"{T[k]['H_path']:.2f}" for k in ("py", "sheets", "webl", "webs")],
        ["path information beyond depth, H(path | depth), bits"] + [f"{T[k]['H_path_given_depth']:.2f}" for k in ("py", "sheets", "webl", "webs")],
        ["99th percentile of path coordinates in the sample"] + [f"{T[k]['rank_p99']:.0f}" for k in ("py", "sheets", "webl", "webs")],
        ["does the full path beat depth?", "no: a tie", "yes, 10 of 10 seeds", "slightly (one run, test: 1.68% against 2.15%)", "no: depth wins"],
    ], num_cols=(1, 2, 3, 4))
    pd, pt_ = c["py_depth"], c["py_top"]
    eff = (c["py_depth"]["mean"] / 2) / (c["py_both"]["mean"] / 44)
    return fig + md(f"""
**On Python, two planes of depth recover most of what 44 planes of path give.** Sequence + depth
beats sequence only by {summ(pd)}, against {num(c['py_both']['mean'])} for the full path —
{pct(100 * pd['mean'] / c['py_both']['mean'], 0)} of the gain for one twenty-second of the
planes. The gain is depth's, not the two extra planes': the [[plctrl|plane-matched control]], with
the same 16 + 2 rotating planes and the same two learned rates spent on level 1 of the path, buys
only {summ(pt_)}, and depth beats it directly, {summ(c['py_depth_top'])}.

**On web text depth does better than the whole path.** On web pages, sequence + depth beats sequence
only on every seed, {summ(c['ws_depth'])}, and beats sequence + tree, {summ(c['ws_depth_both'])}.
A the larger web run sequence + depth run is {pct(100 * webl_depth, 0)} of the way through.

**On spreadsheets depth helps only inside formulas.** Overall, against sequence only it is
{summ(c['sh_depth'])}, and against its control {summ(c['sh_depth_top'])}: neither distinguishable
from zero, and the full path beats depth on every seed, {summ(c['sh_depth_both'])}. Split by token
(validation split, all 10 seeds paired): inside formulas, 59% of tokens, sequence + depth helps
consistently (−0.0032, 7/10, p = 0.03; −0.0030 on the test split), and most on the next comma,
colon or closing bracket (−0.0141, 10/10, p < 0.0001) — exactly what nesting depth predicts. Adding
depth on top of the path roughly doubles the path's gain on those characters (−0.0145 against
−0.0068). It does nothing for cell values or the tab and newline between cells and rows
(−0.0011, 5/10, p = 0.82 on non-formula tokens), which carry most of the loss and its seed-to-seed
noise and need the column, not the nesting. Depth delivery was checked end to end: the depth each
trained model receives matches an independent count with no mismatches, and its learned depth step
moved well away from initialisation on every seed (0.67–0.78 rad on spreadsheets).

**So whether depth can stand in for the path depends on the corpus — and not in the way the
reconstruction argument suggests.** Depth plus order recovers the tree exactly when every ancestor
of a token-carrying node also carries tokens. That holds on Python (checked directly) and fails
completely on web text, where only text nodes carry tokens (0% interior). If depth worked by
letting the model rebuild the tree, it should be weakest on web text. It is strongest there. Depth
is useful as a feature in its own right — how deeply nested this text is — whether or not the tree
can be rebuilt from it.
""") + rec + md(f"""
The table is measured on {T['py']['records']} randomly chosen training documents per corpus,
counting every token; the entropies are plug-in estimates from those samples, so their absolute
values are biased low and only their ordering should be read.

**The obvious information-theoretic explanation fails.** If the path were worth more than depth in
proportion to the information it adds, the path should win where H(path | depth) is largest. Web
text has the most path information beyond depth and is where depth wins. We do not have an
explanation. On spreadsheets the explanation is now measured: the sheet, row and cell nodes emit no tokens of
their own, so a plain value has the same depth anywhere on the grid, and depth says nothing about
which row or column it is in. The grid levels alone carry the full path's gain there (see the
grid section).

**Depth and the path carry different information.** [[rel|Sequence + tree + depth]] is
the full path plus the two depth planes, and it beats sequence + tree {summ(c['py_rel_both'])}. If
depth were only a compressed copy of the path, adding it to the path would add nothing. {rel_text(D)}

**What a plane buys, on Python.** Per rotating plane beyond the sequence planes, depth is about
{eff:.0f} times as productive as the full path. Every configuration that spends a full path buys a
similar amount per plane whichever way it is delivered; the plane-matched control shows that a
plane is not valuable merely for being one of few:
""") + ext + md("""
**A limit on depth.** Depth is a property of the whole preceding text: it cannot name a location
on its own, cannot describe a fragment, and cannot be used out of order. An explicit path can. For
models that read complete documents in order — the setting tested here — depth is the cheap
option; for editing, streaming or out-of-order generation the path buys addressability.
""")


def res_reference(D) -> str:
    c, K = D["c"], CORPUS_CONST
    rows = [crow(c["sh_ref"], "Spreadsheets, values render", "references not otherwise matched"),
            crow(c["sha_ref"], "Spreadsheets, addressed render", "every address also written out"),
            crow(c["py_splice"], "Python", "use and definition share a name")]
    fig = figure("fig-ref", "Reference rotation against sequence + tree",
                 dot_chart(rows, axis="Δ validation nats per token, reference rotation − sequence + tree",
                           left="reference rotation better", right="worse"),
                 """One dot per seed, paired with sequence + tree on the same data and seeds;
reference rotation rotates exactly the same planes, so only the coordinates of reference tokens
differ.""", dot_table(rows))
    ov = table(["corpus and render", "reference tokens also found in the referent's tokens", "reference rotation vs sequence + tree"], [
        ["Python", pct(K["overlap_py"]), summ(c["py_splice"])],
        ["Spreadsheets, values render", pct(K["overlap_sheets_values"]), summ(c["sh_ref"])],
        ["Spreadsheets, addressed render", pct(K["overlap_sheets_addr"]), summ(c["sha_ref"])],
    ], num_cols=(1,))
    pr = D["prelim_ref"]
    var = table(["variant", "baseline", *STAT_HEAD], [
        ["whole-path overwrite (Python)", "sequence + tree", *stat_cells(c["py_anchor"])],
        ["[[delta_off|directed offset]] (spreadsheets)", "plain reference rotation", *stat_cells(c["sh_delta"])],
        ["directed offset (spreadsheets)", "sequence + tree", *stat_cells(c["sh_delta_both"])],
        ["own plane group (spreadsheets, earlier build)", "sequence + tree", pr["n"], num(pr["mean"]), "—", num(pr["t"], 2), "—", f"{pr['won']}/{pr['n']}"],
        ["symbol identity (Python)", "sequence + tree", *stat_cells(c["py_slot"])],
    ], num_cols=(2, 3, 4, 5, 6, 7))
    half = c["sh_ref"]["mean"] / 2
    return fig + md(f"""
**Reference rotation helps on spreadsheets and not on Python, and the difference tracks whether the
pointer is already visible in the text.** We measured, for each render, how often a reference's
tokens also appear among the tokens of what it refers to:
""") + ov + md(f"""
On Python the measured overlap is very high because of how the corpus is written, not because of
anything about code: local variables are renamed by order of definition, so a use of a variable is
spelled exactly like its definition and the link is already in the text. This setting therefore
cannot test the mechanism, and we do not count its null result as evidence about code.

**A control within one corpus.** The addressed spreadsheet render writes every cell's address in
front of its content, so a reference such as E4 can be matched by its spelling to the text `E4:`
at the start of its target cell. We predicted, before these runs finished, that the benefit would
shrink or vanish there; our written threshold was "less than half of the then-current estimate",
which was a five-seed −0.0101. The result, {summ(c['sha_ref'])}, is less than half of that.
Against the final ten-seed estimate of {num(c['sh_ref']['mean'])}, half would be
{num(half, 4)}, so the result sits close to the line. The honest reading: spelling the address out
removes roughly half of the benefit, not all of it. Some of it survives even when a string match
is available — plausibly because attention still has to find the match across thousands of
tokens, which identical angles do directly. We have no test that isolates this.

**Variants.**
""") + var + md(f"""
- **Whole-path overwrite** hurts: definitions sit shallower than uses, so {K['anchor_truncated_pct']}% of overwritten paths lost {K['anchor_levels_lost']} levels on average, and those levels contribute no rotation. The working Python variant keeps the use's own deeper levels.
- **Directed offset** — putting the reference a learned angle away from its target instead of exactly on it, which gives the relation a direction — is indistinguishable from plain reference rotation. The simplest form, taking the target's coordinate, is the one to use.
- **Own plane group**, on an earlier build of the spreadsheet corpus, was a wash at two seeds. Its construction had a flaw: for the 63% of tokens with no reference, the separate group duplicated a coordinate the token already had, so two tokens in one cell agreed on more planes than a reference and its target did.
- **Symbol identity** (Python, one seed) was a dead heat. The learned parameters showed why: the offset moved from 0 to {K['slot_theta_rad']} radians while the rate that distinguishes one symbol from another barely moved, so with {K['slot_absent_pct']}% of tokens carrying no symbol the planes became a "symbol here" flag rather than an identity code.
""")


def res_grid(D) -> str:
    """Sequence + grid: only the sheet, row and column levels of the spreadsheet path."""
    c = D["c"]
    rows = [crow(c["sh_grid"], "vs sequence only", "10 seeds"),
            crow(c["sh_grid_both"], "vs sequence + tree", "all 22 levels, 44 planes"),
            crow(c["sh_grid_rel"], "vs sequence + tree + depth", "46 planes"),
            crow(c["sh_grid_depth"], "vs sequence + depth", "2 planes"),
            crow(c["sh_grid_top"], "vs top-level-only control", "sheet only, 2 planes"),
            crow(c["sh_grid_ref"], "vs reference rotation", "10 seeds"),
            crow(c["sh_grid_split"], "vs layer split", "5 seeds"),
            crow(c["sh_grid_full"], "vs full RoPE", "64 sequence planes")]
    fig = figure("fig-grid", "Spreadsheets: sheet, row and column alone against every other configuration",
                 dot_chart(rows, axis="Δ validation nats per token, sequence + grid − baseline",
                           left="grid better", right="baseline better"),
                 """Each row pairs [[gridcfg|sequence + grid]] with one configuration at the same
seeds. Sequence + grid rotates the first three levels of a cell's path — its sheet, row and column
numbers — on 6 planes, alongside the 16 sequence planes; the 38 planes the full path spends on a
formula's own nesting stay unrotated. The data are the same token stream with the tree cut to its
first three levels.""", dot_table(rows))
    g = c["sh_grid"]
    share = 100 * g["mean"] / c["sh_both"]["mean"]
    return fig + md(f"""
**The grid carries the spreadsheet gain.** The first three levels of a spreadsheet cell's path are
its sheet index, row number and column number — real coordinates, not counts of cells present.
Rotating only those, on 6 planes, beats sequence only by {summ(g)} and ties the full path:
{summ(c['sh_grid_both'])} against [[both|sequence + tree]] and {summ(c['sh_grid_rel'])} against
[[rel|sequence + tree + depth]]. It recovers {pct(share, 0)} of the full path's gain over sequence
only with 6 planes instead of 44. The 38 planes of formula nesting add nothing measurable overall.

**It beats the cheap alternatives.** Against [[depthcfg|sequence + depth]] {summ(c['sh_grid_depth'])};
against the [[plctrl|top-level-only control]], which rotates the sheet level only,
{summ(c['sh_grid_top'])}; against [[fullrope|full RoPE]] {summ(c['sh_grid_full'])}. Row and column
are what pays: the sheet alone buys almost nothing.

**It is not the best spreadsheet result.** [[refrot|Reference rotation]] beats it
({summ(c['sh_grid_ref'])}, grid minus reference rotation) and so does the [[layersplit|layer split]]
({summ(c['sh_grid_split'])}). Both carry the full path; whether either keeps its advantage when cut
to the grid is untested.

**Not a new coordinate.** Row and column coordinates for tables are long established — learned
row/column embeddings (TAPAS, TUTA) and per-head mixtures of row-wise and column-wise RoPE orderings
(2D-TPE). What is measured here is that, alongside sequence position on disjoint planes of the same
rotary heads in a model trained from scratch, the grid accounts for the whole spreadsheet gain of
the tree path. The practical reading: **add a depth marker everywhere; on grid data, row and column
are the only tree coordinates that add to it.**
""")


def res_treegrid(D) -> str:
    """Sequence + depth + rank: the tree's two numbers (depth, own sibling rank) on Python and web pages."""
    c = D["c"]
    rows = []
    for k, setting in (("py", "Python"), ("ws", "web pages")):
        rows += [crow(c[f"{k}_tg"], f"{setting}: vs sequence only", ""),
                 crow(c[f"{k}_tg_depth"], f"{setting}: vs sequence + depth", "2 planes"),
                 crow(c[f"{k}_tg_both"], f"{setting}: vs sequence + tree", "44 planes"),
                 crow(c[f"{k}_tg_rel"], f"{setting}: vs sequence + tree + depth", "46 planes"),
                 crow(c[f"{k}_tg_full"], f"{setting}: vs full RoPE", "64 sequence planes")]
    fig = figure("fig-treegrid", "Sequence + depth + rank against every other configuration, Python and web pages",
                 dot_chart(rows, axis="Δ validation nats per token, sequence + depth + rank − baseline",
                           left="depth + rank better", right="baseline better"),
                 """Each row pairs [[drank|sequence + depth + rank]] with one configuration at the same
seeds. Sequence + depth + rank rotates two numbers per token — its node's depth and the node's rank among its
siblings — on 2 planes each, alongside the 16 sequence planes. The data are the same token stream
with each node's 22-level path replaced by those two numbers.""", dot_table(rows))
    return fig + md(f"""
**The tree's two numbers reduce to depth.** On spreadsheets two numbers — row and column — carry the
whole path's gain (previous section). The analogue for a tree is depth and the node's rank among its
siblings, [[drank|sequence + depth + rank]]. The rank adds nothing measurable: against
[[depthcfg|sequence + depth]] it is {summ(c['py_tg_depth'])} on Python and {summ(c['ws_tg_depth'])} on web pages. Against sequence
only it is {summ(c['py_tg'])} and {summ(c['ws_tg'])}, the same as depth. On web pages it therefore beats
[[both|sequence + tree]] ({summ(c['ws_tg_both'])}) and ties [[rel|sequence + tree + depth]]
({summ(c['ws_tg_rel'])}); on Python it trails both ({summ(c['py_tg_both'])}; {summ(c['py_tg_rel'])}).
Its per-token profile on the held-out test split matches depth's too (next section).

**Why the analogy breaks.** A spreadsheet's row and column are global: every cell in column C shares
the column coordinate however far apart. A sibling rank is local: it restarts under every parent, so
two tokens with the same rank share nothing unless they also share the parent, which neither depth
nor the rank says. What reaches the attention score is essentially depth, and for nested text depth
alone is the cheap coordinate.
""")


UND = f"{RUNS}/analysis/understanding"
UND_CATS = {
    "sheets": [("all", "all tokens (the average)"), ("sh.newline", "row-ending newline"),
               ("sh.ref", "cell reference in a formula"), ("sh.copy_above_far", "copy from the cell above, ≥256 tokens back"),
               ("sh.above_far", "cell whose cell above is ≥256 tokens back"), ("sh.formula", "any formula token"),
               ("sh.value", "plain cell value"), ("sh.cell_start", "first token of a cell")],
    "webc": [("all", "all tokens (the average)"), ("web.node_start", "first token of a new text block"),
             ("web.node_end", "end of a text block"), ("web.rep_cross", "word last seen in another branch"),
             ("web.rep_same", "word last seen under the same parent"), ("web.cont", "inside a text block")],
    "py": [("all", "all tokens (the average)"), ("py.ascend", "a subtree has just closed"),
           ("py.ascend2", "two or more levels have just closed"), ("py.descend", "opening a child node"),
           ("py.ident_reuse", "reusing a variable"), ("py.ident_new", "binding a new variable")],
}
UND_ARMS = {
    "sheets": ["both_v2", "grid_v2", "depth_v2", "rel_v2", "ref_anchor", "layer_split_v2", "full_rope_v2"],
    "webc": ["both", "depth", "rel", "drank", "full_rope"],
    "py": ["rope_both", "depth", "rel", "drank", "rope_leaf", "full_rope", "splice"],
}


def res_cheap(D) -> str:
    """Gain against cost in planes, every setting -- the paper's Figure 4 with every variant."""
    return figure("fig-cheap", "Gain against the number of extra rotary planes, every setting",
                  _static_svg("fig_planes_vs_gain_full"),
                  """Perplexity reduction against [[seq|sequence only]] against the extra planes each
configuration costs (log scale); marker, mean over seeds of the paired gap; line, 95% interval. The larger web run
is one run each, on its test split. Green: depth and depth with a label (sibling rank, parent tag,
parent/child IDs); orange: a grid (the statement grid on code; row and column on spreadsheets;
exchange, speaker and position on chat), with depth where marked; blue: the full tree; violet: tree +
depth; grey: full RoPE.""", table(["note"], [["numbers in the sections that follow"]])) + md("""
**Reading.** In every setting a configuration of 2–10 planes reaches the 44-plane tree. Which one
depends on what the data's structure varies on: depth on nested text (web pages), a grid on tables
(row, column) and chat (speaker, position in the message, with depth), depth with a statement grid on
code. Adding a label to depth that only marks siblings or a parent never helps.
""")


def res_labels(D) -> str:
    """Code's statement grid, and the labels tried on top of depth."""
    c = D["c"]
    rows = [crow(c["py_sg"], "statement grid vs sequence only", "6 planes"),
            crow(c["py_sg_depth"], "statement grid vs depth", ""),
            crow(c["py_sgd"], "statement grid + depth vs sequence only", "8 planes"),
            crow(c["py_sgd_depth"], "statement grid + depth vs depth", ""),
            crow(c["py_sgd_both"], "statement grid + depth vs full tree", "44 planes"),
            crow(c["py_sgd_rel"], "statement grid + depth vs tree + depth", "46 planes")]
    lrows = [crow(c["py_dp"], "Python: parent tag (per depth)", ""), crow(c["py_dpg"], "Python: parent tag (whole document)", ""),
             crow(c["py_drp"], "Python: rank + parent tag", ""), crow(c["py_par"], "Python: parent/child IDs (parity-split)", "9 planes"),
             crow(c["ws_dp"], "web pages: parent tag (per depth)", ""), crow(c["ws_dpg"], "web pages: parent tag (whole document)", ""),
             crow(c["sh_dp"], "Spreadsheets: parent tag (per depth)", ""), crow(c["sh_dpg"], "Spreadsheets: parent tag (whole document)", ""),
             crow(c["ch_dp"], "Chat: parent tag (per depth)", ""), crow(c["ch_dpg"], "Chat: parent tag (whole document)", "")]
    out = [figure("fig-stmtgrid", "Python: the statement grid, alone and with depth",
                  dot_chart(rows, axis="Δ validation nats per token, first − second", left="first better",
                            right="second better"),
                  """The statement grid gives each token its statement's number in the file (counted
across the whole file, nested statements included; 2 planes) and its position within that statement (4
planes): the code analogue of a spreadsheet's row and column. A statement is any node in a block body
(module, function, class, loop, branch, handler); both coordinates are known when the statement opens.
5 seeds.""", dot_table(rows)),
           md(f"""
**The statement grid helps only with depth.** Alone it gains little ({summ(c['py_sg'])} against
sequence only) and is worse than depth on every seed ({summ(c['py_sg_depth'])}). With depth it gains
{summ(c['py_sgd'])}, ties the full tree ({summ(c['py_sgd_both'])}) and falls just short of tree + depth
({summ(c['py_sgd_rel'])}). We had predicted the position within a statement would carry it, as a
column-like axis; that was wrong. A statement is a contiguous stretch of tokens, like a sibling group.
"""),
           figure("fig-labels", "Labels added to depth, each against depth alone",
                  dot_chart(lrows, axis="Δ validation nats per token, variant − depth", left="variant better",
                            right="depth better"),
                  """Each variant keeps depth's planes and adds a label: the parent node's ID, counted
among parents at the same depth or across the whole document (2 more planes); the node's sibling rank
and parent ID together (4 more); or parent/child IDs on two alternating plane groups, A carrying the
nearest even-depth node of (self, parent) and B the nearest odd-depth one, so that every child matches
its parent exactly in one group (3 planes per level, 9 in all). Python 5 seeds, others 10.""",
                  dot_table(lrows)),
           md("""
**None helps.** Every label ties depth or trails it slightly. In our parent-first streams a sibling
group is one contiguous run of tokens, so a label that marks it marks a span sequence position already
resolves; and the parent/child IDs, which do link a child to a distant parent's header, did no better.
The coordinates that pay either describe the token itself (depth) or line up tokens that are far
apart (a spreadsheet column, the start of every chat message). The parent/child construction was
proposed by the author; it is recorded here as a tested negative result.
""")]
    return "\n".join(out)


def _md_tables(text: str) -> str:
    """docs/mechanism/results.md -> HTML: '## ' headings, markdown tables, prose blocks."""
    out, para, rows = [], [], []

    def flush_para():
        if para:
            out.append(md("\n".join(para)))
            para.clear()

    def flush_rows():
        if rows:
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            body = [r for r in cells[1:] if not all(set(c) <= set("-: ") for c in r)]
            out.append(table(cells[0], body, num_cols=tuple(range(1, len(cells[0])))))
            rows.clear()

    for ln in text.splitlines():
        if ln.startswith("# "):
            continue
        if ln.startswith("### "):
            flush_para(); flush_rows()
            out.append(f"<h5>{inline(ln[4:])}</h5>")
        elif ln.startswith("## "):
            flush_para(); flush_rows()
            out.append(f"<h4>{inline(ln[3:])}</h4>")
        elif ln.startswith("|"):
            flush_para()
            rows.append(ln)
        elif not ln.strip():
            flush_rows(); flush_para()
        else:
            flush_rows()
            para.append(ln)
    flush_rows(); flush_para()
    return "\n".join(out)


def res_mechanism(D) -> str:
    """How the structure is used: analysis/mechanism.py on final checkpoints, test split, seeds 0-4."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = [md("""
Four analyses on the final checkpoints, test split, seeds 0–4, each paired against the same seed's
[[seq|sequence only]] model; no training. **(1) Removing the structure at test time:** score the
same text with every token's coordinate absent, or with each record's coordinates shuffled among its
own tokens. **(2) Where attention goes:** the share of each head's attention on structurally related
earlier tokens, against the share uniform attention would give. **(3) How much the rotation moves
the scores:** hold each head's queries and keys fixed and switch the structure rotation off.
**(4) The learned rotation steps,** which are shared by every head and layer.

**The models rely on the structure, and come to depend on it.** Shuffling costs far more than the
structure gained, on every seed: sequence + grid on spreadsheets gains 1.4% and loses 29% when
shuffled; [[depthcfg|sequence + depth]] on web pages gains 2.0% and loses 18%. On Python, shuffling depth
multiplies perplexity about 7.5×: the compact syntax-tree render has no brackets or indentation, so
a sequence-only model counts children from the node tags, and the structured model hands that
bookkeeping to the rotation. Removing the structure is an input the Python models never saw in
training, so shuffling is the cleaner of the two tests there.

**Attention moves where the gain lands.** On spreadsheets the grid moves attention onto earlier cells
of the same column and onto the referenced cell; on web pages depth moves it onto text at the same depth
elsewhere on the page; on Python the average shift is only suggestive. Individual heads move a great
deal in both directions, so the structure reassigns heads rather than nudging all of them. A
sequence-only model also grows a column-reading head, because row lengths repeat and a fixed
sequence offset often lands on the cell above; the grid's effect is that more of the network reads
down the column, not that the ability appears from nothing.
"""),
           figure("fig-mech-map", "Where the average head attends, in grid coordinates",
                  _static_svg("fig_mechanism_example"),
                  """Attention of the average head (all 12 heads, 5 seeds) from a spreadsheet cell to
cells in the same row and the rows above, by column offset; sequence only against sequence + grid.""",
                  table(["note"], [["full tables below"]])),
           figure("fig-mech-attn", "Attention on structurally related keys, every head of every seed",
                  _static_svg("fig_mechanism_attention"),
                  """Change in each head's share of attention on each relation, structured model minus
sequence only at the same seed; each dot is one head of one seed; marker and line, mean over heads
and its 95% confidence interval.""",
                  table(["note"], [["full tables below"]]))]
    out.append(_md_tables(open(os.path.join(here, "docs", "mechanism", "results.md")).read()))
    return "\n".join(out)


def res_where(D) -> str:
    """Perplexity change per kind of token (understanding_evals.py), on the test split."""
    g = lambda d: 100 * (1 - math.exp(d))
    out = [figure("fig-where", "Where the gain lands, per kind of token", _static_svg("fig_understanding"),
                  """[[pplred|Perplexity reduction]] on each kind of token against [[seq|sequence
only]], test split: 100·(1 − exp(Δ)) with Δ the paired change in mean loss on that category (right =
better). Large marker: mean; line: 95% confidence interval; faint markers: seeds. The shaded row is
the all-token average; each panel has its own scale.""",
                  table(["note"], [["full tables below"]]))]
    out.append(md("""
The averages elsewhere on this page are over every token, and most tokens do not depend on
structure. Here the same paired comparison is split by what is being predicted, on the held-out
**test** split. **Δ** is the configuration's change in mean loss (nats/token) on that category;
**perplexity** is the same change as a perplexity reduction, the unit used throughout; **won** is
seeds better; **beyond** is the category's Δ minus the configuration's all-token Δ, with its own
paired test — a negative, significant *beyond* means the configuration helps that kind of token
*more* than its average. These are likelihoods, not task accuracy. Structural minimal pairs
(shuffling sections, corrupting nesting) were deliberately not run: perturbing the text while
keeping the coordinates consistent measures the input contract, not understanding.
"""))
    R = {}
    for setting, title in (("sheets", "Spreadsheets (test split, 1,024 workbooks)"),
                           ("webc", "web pages (test split, 1,024 pages)"),
                           ("py", "Python (test split, 8,192 programs)")):
        rows = json.load(open(os.path.join(UND, f"{setting}_results.json")))
        R[setting] = {(r["arm"], r["cat"]): r for r in rows}
        body = []
        for arm in UND_ARMS[setting]:
            for cat, lab in UND_CATS[setting]:
                r = R[setting].get((arm, cat))
                if r is None:
                    continue
                body.append([r["name"] if cat == "all" else "", lab, f"{r['n_tok']:,}", num(r["delta"]),
                             f"{g(r['delta']):+.2f}%", f"{r['won']}/{r['n_seeds']}", pfmt(r["p"]),
                             "—" if cat == "all" else num(r["beyond"]),
                             "—" if cat == "all" else pfmt(r["beyond_p"])])
        out.append(f"<h4>{esc(title)}</h4>" + table(
            ["configuration", "kind of token", "tokens", "Δ", "perplexity", "won", "p", "beyond", "p (beyond)"],
            body, num_cols=(2, 3, 4, 5, 6, 7, 8)))
    S, W, P = R["sheets"], R["webc"], R["py"]
    v = lambda T, a, c: g(T[(a, c)]["delta"])
    nl = [v(S, a, "sh.newline") for a in ("both_v2", "grid_v2", "layer_split_v2", "ref_anchor")]
    av = [v(S, a, "all") for a in ("both_v2", "grid_v2", "layer_split_v2", "ref_anchor")]
    ns = {a: v(W, a, "web.node_start") for a in ("both", "depth", "rel", "drank")}
    ne = {a: v(W, a, "web.node_end") for a in ("both", "depth", "rel", "drank")}
    asc = {a: v(P, a, "py.ascend") for a in ("rope_both", "depth", "rel", "drank")}
    out.append(md(f"""
**Spreadsheets.** Every configuration that carries row and column — the full path, the grid, the
layer split, reference rotation — improves the row-ending newline by {min(nl):.1f}–{max(nl):.1f}%,
against {min(av):.1f}–{max(av):.1f}% on average: predicting where a row ends needs the column count.
Values copied from the cell above across a long gap improve by {v(S, 'both_v2', 'sh.copy_above_far'):.1f}%
under the path, {v(S, 'grid_v2', 'sh.copy_above_far'):.1f}% under the grid and
{v(S, 'layer_split_v2', 'sh.copy_above_far'):.1f}% under the layer split. Reference rotation improves
cell references by {v(S, 'ref_anchor', 'sh.ref'):.1f}% (the path alone: {v(S, 'both_v2', 'sh.ref'):.1f}%).
[[gridcfg|Sequence + grid]] matches the full path on every category. Sequence + depth helps none of
these; its one consistent effect is inside formulas (see the depth section). Full RoPE is worse than
the baseline on most categories.

**web pages.** The largest effect is the first token of a new text block: {min(ns.values()):.1f}–{max(ns.values()):.1f}%
under every structured configuration, on all 10 seeds, against {v(W, 'both', 'all'):.1f}–{v(W, 'rel', 'all'):.1f}%
on average. The end of a text block, a short and easy token (sequence only: {W[('both', 'web.node_end')]['seq_nll']:.2f}
nats), improves by {min(ne.values()):.1f}–{max(ne.values()):.1f}%, no more than average. Repeats of a
word last seen in a different branch improve by {v(W, 'both', 'web.rep_cross'):.1f}–{v(W, 'depth', 'web.rep_cross'):.1f}%.
[[drank|Sequence + depth + rank]]'s profile matches depth's on every category. Full RoPE helps no
category.

**Python.** For configurations that keep sequence position the gain is spread evenly; no category
exceeds the average significantly at five seeds. The most negative values, on the token right after
a subtree closes ({min(asc.values()):+.2f}% to {max(asc.values()):+.2f}%), are not significant on any
configuration. Tree only (20 seeds) helps child-opening tokens more than average, and [[none|no
position]] is hurt most where a subtree closes.
"""))
    return "".join(out)


def res_test(D) -> str:
    c, t2, t8 = D["c"], D["test2048"], D["test8192"]
    v = c["py_leaf"]
    tt = table(["instrument", "documents", "n", "Δ", "se", "t", "p", "won"], [
        ["validation, final step", "1,024", v["n"], num(v["mean"]), num(v["se"], 4, False), num(v["t"], 2), pfmt(v["p"]), f"{v['won']}/{v['n']}"],
        ["test, final checkpoint", "2,048", t2["n"], num(t2["mean"]), num(t2["se"], 4, False), num(t2["t"], 2), pfmt(t2["p"]), f"{t2['won']}/{t2['n']}"],
        ["test, final checkpoint", "8,192", t8["n"], num(t8["mean"]), num(t8["se"], 4, False), num(t8["t"], 2), pfmt(t8["p"]), f"{t8['won']}/{t8['n']}"],
    ], num_cols=(1, 2, 3, 4, 5, 6, 7))
    six = t2["seeds"]
    d8 = [t8["diffs"][s] for s in six]
    sd8 = st.stdev(d8)
    sd2 = t2["sd"]
    v2, v8 = sd2 ** 2, sd8 ** 2
    C = (v2 - v8) / (1 / 2048 - 1 / 8192)
    M = v8 - C / 8192
    floor = math.sqrt(M) if M > 0 else 0.0
    samp = math.sqrt(C / 2048) if C > 0 else 0.0
    w = D["webl"]["runs"]
    return md("""
**Python, tree + within-node index against sequence only, on the validation and test splits.** No
checkpoint selection anywhere: validation at the fixed final step, test at the final checkpoint.
""") + tt + md(f"""
Both splits find the same direction and a similar size at full seed count. The test estimate is
noisier at the same n (standard error {num(t8['se'], 4, False)} against {num(v['se'], 4, False)}).
On the six seeds scored on both test sets, the paired standard deviation falls from
{num(sd2, 4, False)} at 2,048 documents to {num(sd8, 4, False)} at 8,192 — a factor of
{sd2 / sd8:.2f}, between the 2.0 that document sampling alone would give and the 1.0 of a pure
model-to-model floor. Fitting variance = M + C / documents to the two points gives a floor of
{num(floor, 4, False)} and a sampling component of {num(samp, 4, False)} at 2,048 documents. So
two checkpoints that are indistinguishable on one held-out set can differ on another drawn from the
same corpus, which is why every comparison here is a per-seed paired difference.

**the larger web run** agrees on its test split: sequence + tree {num(w['both']['test'][0] - w['seq']['test'][0])}
and tree + within-node index {num(w['leaf']['test'][0] - w['seq']['test'][0])} against sequence only,
matching the signs on validation.
""")


def res_sheets(D) -> str:
    c, cor, w = D["c"], D["corpus"], D["webl"]
    sha, sh = cor["sheets_addr"], cor["sheets"]
    prog_done, prog_n, _ = D["prog"]["sh_split"]
    return md(f"""
**Grid position beats sequence position on every seed, in both renders**: {summ(c['sh_both'])} on
the values render and {summ(c['sha_both'])} on the addressed render. This is the corpus where a
negative result was most plausible: TableLoRA (2025) added row and column indices to the position
embedding of a pretrained model on hierarchical tables and scored below a baseline that did
nothing. Partitioning the rotary planes of a model trained from scratch does not reproduce that
loss. (TableLoRA's setting — additive, into a pretrained model — is different enough that this is
not a contradiction of it.)

**The addressed render benefits more, which we did not expect.** Writing every cell's address into
the text gives the model the grid position as ordinary tokens, so the rotation should be partly
redundant and the gain smaller. It is larger. The two renders are not directly comparable, though:
the addressed one has {sha['meta']['total_tokens'] / 1e6:.1f} M tokens against
{sh['meta']['total_tokens'] / 1e6:.1f} M, the extra tokens are highly predictable addresses, and
nats per token average over different token sets. What can be said is that the addressed render
does not show the shrinkage that redundancy would predict.

**A prediction recorded in advance.** Before the spreadsheet runs finished, we wrote down that
spreadsheets, whose {sh['nodes_per_token']:.2f} [[npt|nodes per token]] sit between Python's
{cor['py']['nodes_per_token']:.2f} and web text's {cor['webl']['nodes_per_token']:.2f}, would also
sit between them on two quantities. The record is in the project's version-controlled notes; a
reader cannot verify its timing independently.

- **The margin of sequence + tree over sequence only**: Python {num(c['py_both']['mean'])}, spreadsheets {num(c['sh_both']['mean'])}, the larger web run {num(w['both_summary']['mean'])} (mean over its 20 milestones) — in the predicted order. This compares nats per token across settings, which we otherwise avoid, so it is weak evidence.
- **The share of that margin kept by the alternating layer split**: Python {pct(100 * c['py_alt']['mean'] / c['py_both']['mean'], 0)}, the larger web run about {pct(100 * st.mean([w['layer_split_share'][s] for s in sorted(w['layer_split_share'])[-5:]]), 0)} late in the run. On spreadsheets {prog_done} of {prog_n} seeds have finished; see the splits section for the provisional figure.
""")



def res_cells(D) -> str:
    """The mixed-corpus cells, the stripped-text workarounds and the prefix-causal chat tree.

    Each is written up in full in its own doc under docs/; those docs are generated from the same
    scored JSON the paper's audit reads, so this section carries their tables verbatim rather than
    recomputing them a second way."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = [md("""
Four groups of runs that the paper reports in one or two sentences each. All of them are finished;
the full tables, intervals and caveats are below, exactly as the result docs state them.

- **Four corpora in one model** (§4.6 of the paper): Web pages, Python, spreadsheets and chat
  trained together over a union vocabulary, with one shared [[depth]] coordinate against four
  private ones, a grid for the two corpora that have one, the full 44-plane path, and two
  rescalings of depth. Seven configurations, ten paired seeds, scored per corpus.
- **Two corpora in one model**: the smaller precursor — web pages and spreadsheets — which asked
  whether one model keeps each corpus's coordinate preference and whether two corpora can share one
  plane group.
- **Stripped text**: two ways to make depth usable when the markup is not in the token stream.
  Withholding depth on part of the training data works; writing depth into the text as compact
  markers does not.
- **A chat tree a generator can supply**: the chat pack parsed whole messages, so a few tokens'
  Markdown depth depends on later text. These runs retrain on a tree rebuilt from each token's own
  prefix, which makes the reading gain and the generation gain the same number.
""")]
    for sub, title in (("mix4", "Four corpora in one model"),
                       ("mixed", "Two corpora in one model"),
                       ("stripped", "Stripped text: depth dropout, and depth written as text markers"),
                       ("chat_causal", "Chat with a prefix-causal Markdown tree")):
        path = os.path.join(here, "docs", sub, "results.md")
        if not os.path.exists(path):
            continue
        out.append(f"<h4>{esc(title)}</h4>")
        out.append(_md_tables(open(path).read()))
    return "\n".join(out)


def sec_related(D) -> str:
    coord = table(["coordinate", "example", "identifies a node uniquely on its own?"], [
        ["depth, or depth + sibling rank", "Bartkowiak & Graliński (2025)", "no"],
        ["depth, together with retained sequence position", "Wang, Tu, Wang & Shi (2019), additively; this work, rotary", "yes, when every ancestor carries tokens in a pre-order serialisation"],
        ["depth + parent's width + own width", "Guo et al. (2022)", "no"],
        ["full root-to-node path of sibling ranks", "Shiv & Quirk (2019); Peng et al. (2022); this work", "yes"],
        ["root path as a product of learned orthogonal generators, one per branch choice", "Kogkalidis, Bernardy & Garg (2024)", "yes"],
        ["root path of node types", "HiT (Zhang et al., 2023)", "no (types repeat)"],
        ["path compressed by a recurrent encoder", "TPTrans (Peng et al., 2021); CRPE (2025)", "approximately"],
        ["path length from the root, replacing the token index", "D-RoPE (Li et al., 2026)", "no"],
    ])
    site = table(["where the coordinate enters the model", "does it work?", "examples"], [
        ["added or concatenated to the token embedding", "largely no", "Shiv & Quirk; TUTA; MarkupLM; HiT; HDT; Tree-Enhanced CodeBERTa"],
        ["as a bias on attention scores", "yes", "Peng et al.; TPTrans; Code Transformer; TableFormer"],
        ["rotating queries and keys", "this work", "HiRoPE (hierarchy levels); algebraic encodings (Kogkalidis et al., 2024, synthetic tasks)"],
    ])
    return md("""
**Where the tree coordinate comes from.**""") + coord + md("""
**How it enters the model** turns out to matter more than the coordinate.""") + site + md("""
Six papers put a full path into the input embedding, across four domains, with a poor record.
Chirkova & Troshin (2021) reimplemented Shiv & Quirk's scheme in one controlled framework and
concluded that structure in the input embedding "is not effective" while structure in the attention
mechanism "is much more effective": on their code-completion task, tree position scored 53.82 mean
reciprocal rank (MRR, higher is better) against 54.53 for sequential relative attention, beating
only absolute sequence position (53.50). TPTrans saw no gain from its path in the token embedding
and kept it only as an attention bias. A probing study (CSA-Trans; Oh & Yoo, 2024) found the full
path far more recoverable from the model than the depth-and-width triplet, yet the downstream
difference was about 0.01 BLEU. Rotating queries and keys puts structure in the attention mechanism,
which is the main reason to expect anything here at all.

**Splitting rotary planes between coordinates** is well established for images and video: M-RoPE in
Qwen2-VL splits each head between time, height and width; axial RoPE for vision transformers gives
half the planes to each image axis (Heo et al., 2024), and finds that mixing all axes into every
plane does better; VideoRoPE (Wei et al., 2025) reorders which axis gets which frequencies. Two
papers argue against narrow bands: Huang et al. (2025) ask that no axis be confined to a narrow
frequency band, and Li et al. (2025) replace fixed splits with a learned per-head allocation. Their
recommended alternative, a whole head or layer per coordinate, is what our split configurations
test; on Python it loses.

**What is already published, and what is not.** Plane partitioning across axes (M-RoPE, axial
RoPE, VideoRoPE); partitioning by hierarchy level with two levels (HiRoPE, Zhang et al., 2024,
which keeps sequence position and is applied without training to pretrained code models); a unique
root-path coordinate for every syntax-tree node (Peng et al., 2022); a per-level embedding for the
whole root path plus a within-cell token index, added into a bidirectional encoder (TUTA, Wang et
al., 2021 — the same coordinate design as ours, used additively); learned per-axis rotation rates
(RoPE-Mixed, in Heo et al., 2024); rotary encodings driven by graph structure (WIRE, Reid et al.,
2025) — which subsumes our *grid* result, since recovering RoPE on Cartesian grids is WIRE's own
theorem and a table is a grid, but not depth: Laplacian spectra are direction-blind while depth in a
rooted tree is directed, which is why direction-aware encodings reach for a Magnetic Laplacian
(Geisler et al., 2023); per-level tag and sibling index for HTML (MarkupLM); a tree's full root path as a rotation —
products of learned orthogonal operators, one per branch choice, applied to queries and keys and
initialised to mimic RoPE, tested with encoder-decoder models on synthetic tree tasks and without a
retained sequence index (algebraic positional encodings, Kogkalidis et al., 2024); and tree depth
as a structural position fused with sequential position, additively, in machine translation
(Wang, Tu, Wang & Shi, 2019). The rotation family itself has been characterised for continuous
spatial coordinates — learned dense rotation generators for 2D and 3D vision (LieRE, Ostmeier et
al., 2025) and a universality result for translation-invariant encodings expressible by matrix
multiplication, applied to vision and robotics (STRING, Schenck et al., 2025) — but that line does
not address which discrete structural coordinate is worth spending planes on in a language model.
What we did not find published is
giving **each level of a full root-to-node path its own rotary planes, rotated by the sibling rank
at that level, in a causal language model trained from scratch, measured in nats per token** —
with and without sequence position.

HiRoPE's own ablations do not show that hierarchy is what helps it: its split-ratio sweep is flat,
and its function-level segmentation scores 1.9121 against 1.9203 for fixed 128-token chunks, a
difference of 0.008 perplexity. Its gains plausibly come from the position clipping it shares
with ReRoPE (Su, 2023) and Self-Extend (Jin et al., 2024), two training-free methods that extend
context by capping relative distances.

**Relations as rotations.** RotatE (Sun et al., 2019) defines each knowledge-graph relation as a
rotation from source to target — the group argument behind our relation offsets, as an embedding
rather than a positional encoding. Position Coupling (Cho et al., 2024) gives related tokens shared
position ids, including a rotary variant, in every head and in place of sequence position. LESA
(2021) gives each token its dependency head's position additively; Trans+C-Embedding (2024) gives
every token in a coreference cluster the position of its first mention, additively. The use of a
variable as a link to its previous use is a standard program-graph edge (LastLexicalUse,
Allamanis et al., 2018), used as an attention bias by GREAT (Hellendoorn et al., 2020). What we
did not find published is relation offsets on their own planes **alongside** ordinary sequence
position and tree rank inside every head.

**The closest published negative.** He, Sedoc & Rodu (2021) analyse theoretically whether a
transformer can represent trees, and test it on synthetic tree tasks: a standard transformer
reached accuracy similar to one given an explicit tree position encoding, converging more slowly.
If a standard transformer can recover tree structure on its own, explicit tree position should
buy little. Our setting differs in four ways that bear on that conclusion: real corpora (code,
spreadsheets, web pages) rather than synthetic trees; a fixed training budget, so faster
convergence is itself a gain in loss at the step we measure; next-token language-modelling loss
rather than task accuracy; and tree position **added alongside** sequence position rather than
replacing it. Their result predicts that the gain should shrink with longer training, which our
single long run (the larger web run) does not show over one pass — its margin is as large at the end as at
the start — but a longer budget could still close it.

**Two other published results point the other way.** TableLoRA (2025): row and column indices added to
the position embedding of a pretrained model on hierarchical tables scored 39.00 against 43.06 for
a baseline that added nothing; supplied per layer instead, the same signal scored 48.94. Rousselet
(2021, a master's thesis we have only at second hand) applied Shiv & Quirk's scheme to web pages and
found it significantly worse than no position at all. Both add a structural signal additively, and
the second also replaces sequence position.

**Closest setup.** SiPE (Riaz et al., 2026) trains a causal model from scratch with a
parse-derived signal in the position pathway, added to sequence position, and reports perplexity
18.63 → 16.95 against a sequence-only control.

**Position-free layers.** Cohere's Command A and Meta's Llama 4 interleave rotary layers with
layers that get no position at a 3:1 ratio (Yang et al., 2025): a transformer does not need
position in every layer. Our thirds pattern agrees at small scale.

**The plane budget does not grow with scale.** The number of rotary planes per head is set by the
head width, which is roughly constant across model sizes: 128 in GPT-3, Llama 3 70B and 405B, and
Qwen2.5 72B; 256 across PaLM's 67-fold range of sizes; and DeepSeek-V3, at 671 billion parameters,
rotates only 64 dimensions per head. How a model spends its positional planes is therefore a design
question at every scale, not an artefact of small models.

**Effect sizes in this literature are small.** HiRoPE's advantage over its strongest baseline is
about 0.016; Pushdown Layers report "similar perplexities" while gaining on syntax; Tree-Planted
Transformers move perplexity 47.5 → 45.5 while moving a syntactic benchmark 71.7 → 77.1. Our
effects, 0.002 to 0.025 nats per token, are of that order.
""")


def sec_limits(D) -> str:
    w = D["corpus"]["webl"]["side"]
    return md(f"""
- **Scale.** 4.7 million and 31.5 million non-embedding parameters. Nothing here licenses a claim about large models.
- **the larger web run is one seed per configuration.** Its direction is replicated ten times over on web pages, which uses a smaller model, a shorter context and differently filtered data, so web pages corroborates the direction and cannot confirm the larger web run's magnitudes.
- **Learning rates were selected on validation loss**, the split we report. The rule was the same within a setting, the test split agrees in sign, but the selection was never repeated on the test split. Spreadsheets reuse a rate tuned on an earlier build of their corpus.
- **Settings differ in more than the corpus.** Python uses its own tokenizer and about eight passes; The larger web run the GPT-2 tokenizer and one pass. The corpus contrasts (replacing sequence position helps on code, hurts on prose) are confounded with tokenizer and training length, even though the nodes-per-token explanation is a property of the corpora.
- **Excluded data.** Both web settings drop pages deeper than 22 levels or wider than 318 siblings ({pct(100 * (w['pages_dropped_deep_frac'] + w['pages_dropped_alias_frac']))} of the larger web run pages). The tree-position literature usually clips such coordinates instead of dropping the page. Dropping keeps every coordinate exact, but it is a choice other work did not make.
- **The fast tree planes [[wrap]].** The fast plane of each level counts only about 20 siblings at its initial rate; {pct(100 * w['ranks_fast_wrap_frac'])} of the larger web run sibling ranks exceed that. The slow plane disambiguates, but the allocation of two planes per level was not tuned.
- **Uncontrolled differences in the head-split comparison**: head count per signal (one on Python, two on the larger web run) and compilation (the the larger web run head-split run alone was not compiled).
- **Grid coordinates were tested only on spreadsheets.** Row and column for other tables (HTML tables, CSV) are untested.
- **Python's test split shows about half the validation gain** for the configurations that keep sequence position (sequence + tree −0.0042, 3/5; sequence + tree + depth −0.0075, 4/5, on 8,192 test programs); spreadsheets and web pages reproduce their validation gains on test.
- **Sources read at second hand** are marked [S] in the references. Two matter: CRPE (paywalled), the one prior rotary scheme for syntax trees; and Rousselet (2021), the one prior full-path encoding on web pages, whose title we have not confirmed.
""")


def sec_future(D) -> str:
    return md("""
- **Why depth wins on prose and the path on spreadsheets.** The measurement that would test the leading hypothesis is how much a token's content depends on its coordinate: the conditional entropy of the token given its full path against given its depth, per corpus. Plug-in estimates of both are biased in the direction that favours the path, so a result showing the path less predictive on web text would be robust and the reverse would not.
- **A no-position floor for web text**, with a recipe that trains stably (lower peak rate, longer warm-up), so web effects can be expressed as a share of what position is worth.
- **Reference rotation on code with real variable names**, where a use and its definition do not share a spelling.
- **The head-versus-layer question with head count held fixed** within one corpus.
- **The experiment at scale**: sequence only against sequence + tree (and sequence + depth) at a billion parameters or more, several billion tokens, at least three seeds, scored in nats per token at a fixed step on a fixed held-out split, with both signals interleaved inside every head and the four within-node planes given to tree levels instead.
""")


REFERENCES = [
    ("Allamanis, Brockschmidt & Khan (2018)", "Learning to Represent Programs with Graphs. ICLR 2018. arXiv:1711.00740.", "P"),
    ("Barbero, Vitvitskyi, Perivolaropoulos, Pascanu & Veličković (2025)", "Round and Round We Go! What makes Rotary Positional Encodings useful? ICLR 2025. arXiv:2410.06205.", "P"),
    ("Bartkowiak & Graliński (2025)", "Depth-and-sibling tree positional encoding for code. Title and venue unconfirmed.", "S"),
    ("Biderman et al. (2023)", "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML 2023. arXiv:2304.01373.", "P"),
    ("BigScience (2022)", "c4-en-html-with-metadata (dataset, Hugging Face Hub). Source of the web pages.", "P"),
    ("Black et al. (2022)", "GPT-NeoX-20B: An Open-Source Autoregressive Language Model. arXiv:2204.06745. Rotary on the first 25% of dimensions.", "P"),
    ("Brown et al. (2020)", "Language Models are Few-Shot Learners (GPT-3). NeurIPS 2020.", "S"),
    ("Chirkova & Troshin (2021)", "Empirical Study of Transformers for Source Code. ESEC/FSE 2021. arXiv:2010.07987.", "P"),
    ("Cho, Cha, Awasthi, Bhojanapalli, Gupta & Yun (2024)", "Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure. NeurIPS 2024. arXiv:2405.20671.", "P"),
    ("Chowdhery et al. (2022)", "PaLM: Scaling Language Modeling with Pathways. arXiv:2204.02311.", "P"),
    ("CRPE (2025)", "Advancing code completion through rotary position embedding. Cluster Computing 28. Paywalled; read from the abstract and secondary summaries.", "S"),
    ("DeepSeek-AI (2024)", "DeepSeek-V3 Technical Report. arXiv:2412.19437.", "P"),
    ("Dong et al. (2024)", "SpreadsheetLLM: Encoding Spreadsheets for Large Language Models. EMNLP 2024. arXiv:2407.09025. Source of the addressed render's 'address:content' format.", "P"),
    ("Gao, Wettig, He, Dong, Malladi & Chen (2025)", "Metadata Conditioning Accelerates Language Model Pre-training. ICML 2025. arXiv:2501.01956.", "P"),
    ("Golovneva, Wang, Weston & Sukhbaatar (2024)", "Contextual Position Encoding: Learning to Count What's Important. arXiv:2405.18719.", "P"),
    ("Gong, Elhoushi & Cheung (2024)", "AST-T5: Structure-Aware Pretraining for Code Generation and Understanding. ICML 2024. arXiv:2401.03003.", "P"),
    ("Grattafiori, Dubey et al. (2024)", "The Llama 3 Herd of Models. arXiv:2407.21783.", "P"),
    ("Guo et al. (2021)", "GraphCodeBERT: Pre-training Code Representations with Data Flow. ICLR 2021. arXiv:2009.08366.", "P"),
    ("Guo et al. (2022)", "Modeling Hierarchical Syntax Structure with Triplet Position for Source Code Summarization. ACL 2022.", "P"),
    ("He et al. (2024)", "HDT: Hierarchical Document Transformer. COLM 2024. arXiv:2407.08330.", "P"),
    ("He, Sedoc & Rodu (2021)", "Trees in transformers: a theoretical analysis of the Transformer's ability to represent trees. arXiv:2112.11913.", "S"),
    ("Hellendoorn, Sutton, Singh, Maniatis & Bieber (2020)", "Global Relational Models of Source Code (GREAT). ICLR 2020.", "S"),
    ("Hendrycks et al. (2021)", "Measuring Coding Challenge Competence With APPS. NeurIPS 2021 Datasets and Benchmarks. arXiv:2105.09938. Source of Python programs.", "P"),
    ("Heo et al. (2024)", "Rotary Position Embedding for Vision Transformer (RoPE-ViT; includes RoPE-Mixed). ECCV 2024. arXiv:2403.13298.", "P"),
    ("Herzig, Nowak, Müller, Piccinno & Eisenschlos (2020)", "TaPas: Weakly Supervised Table Parsing via Pre-training. ACL 2020. arXiv:2004.02349.", "S"),
    ("Huang et al. (2025)", "Revisiting Multimodal Positional Encoding in Vision-Language Models. arXiv:2510.23095.", "P"),
    ("Jin et al. (2024)", "LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning. ICML 2024. arXiv:2401.01325.", "S"),
    ("Karpathy (2022)", "nanoGPT. Software, https://github.com/karpathy/nanoGPT. The code base this work builds on.", "P"),
    ("Kazemnejad, Padhi, Natesan Ramamurthy, Das & Reddy (2023)", "The Impact of Positional Encoding on Length Generalization in Transformers. NeurIPS 2023. arXiv:2305.19466.", "P"),
    ("Kogkalidis, Bernardy & Garg (2024)", "Algebraic Positional Encodings. NeurIPS 2024. arXiv:2312.16045.", "P"),
    ("LESA (2021)", "Linguistically-Enhanced Self-Attention. EACL 2021. arXiv:2101.11891. Author list unconfirmed.", "S"),
    ("Li, Che, Lyu, Huang & Xu (2026)", "TCDA: Thread-Constrained Discourse-Aware Modeling for Conversational Sentiment Quadruple Analysis (introduces D-RoPE). IJCAI 2026. arXiv:2605.01717.", "P"),
    ("Li, Chen, Li, Dong, Wang & Zhu (2025)", "Head-wise Adaptive Rotary Positional Encoding for Fine-Grained Image Generation (HARoPE). arXiv:2510.10489.", "P"),
    ("Li, Guan, Wu, Yu & Yan (2025)", "2D-TPE: Two-Dimensional Positional Encoding Enhances Table Understanding for Large Language Models. WWW 2025. arXiv:2409.19700.", "P"),
    ("Li, Xu, Cui & Wei (2022)", "MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding. ACL 2022. arXiv:2110.08518.", "P"),
    ("McLeish et al. (2024)", "Transformers Can Do Arithmetic with the Right Embeddings. NeurIPS 2024. arXiv:2405.17399.", "P"),
    ("Murty, Sharma, Andreas & Manning (2023)", "Pushdown Layers: Encoding Recursive Structure in Transformer Language Models. EMNLP 2023. arXiv:2310.19089.", "P"),
    ("Narang et al. (2021)", "Do Transformer Modifications Transfer Across Implementations and Applications? EMNLP 2021. arXiv:2102.11972.", "P"),
    ("Oh & Yoo (2024)", "CSA-Trans: Code Structure Aware Transformer for AST. arXiv:2404.05767.", "S"),
    ("Peng, Li, Wang, Zhao & Jin (2021)", "Integrating Tree Path in Transformer for Code Representation (TPTrans). NeurIPS 2021.", "S"),
    ("Peng, Li, Zhao & Jin (2022)", "Rethinking Positional Encoding in Tree Transformer for Code Representation. EMNLP 2022.", "P"),
    ("Press, Smith & Lewis (2022)", "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi). ICLR 2022. arXiv:2108.12409.", "P"),
    ("Qwen Team (2024)", "Qwen2.5 Technical Report. arXiv:2412.15115.", "P"),
    ("Raffel et al. (2020)", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5, C4). JMLR 21(140). arXiv:1910.10683.", "P"),
    ("Reid et al. (2025)", "Rotary Position Encodings for Graphs (WIRE). arXiv:2509.22259.", "P"),
    ("Geisler, Li, Mankowitz, Cemgil, Günnemann & Paduraru (2023)", "Transformers Meet Directed Graphs. ICML 2023. arXiv:2302.00049.", "P"),
    ("Ostmeier, Axelrod, Varma, Moseley, Chaudhari & Langlotz (2025)", "LieRE: Lie Rotational Positional Encodings. ICML 2025. arXiv:2406.10322.", "P"),
    ("Schenck, Reid, Jacob, Bewley, Ainslie, Rendleman et al. (2025)", "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING. arXiv:2502.02562.", "P"),
    ("Wei, Liu, Zang, Dong, Zhang, Cao et al. (2025)", "VideoRoPE: What Makes for Good Video Rotary Position Embedding? arXiv:2502.05173.", "P"),
    ("Huang, Liu, Song, Hou, Chang, Lin & Bai (2025)", "Revisiting Multimodal Positional Encoding in Vision-Language Models. arXiv:2510.23095.", "P"),
    ("Riaz, Kim & Surdeanu (2026)", "Beyond Sequence Order: Syntax-Informed Positional Embeddings for Transformers (SiPE). arXiv:2608.06111.", "P"),
    ("Rousselet (2021)", "MSc thesis, KTH Royal Institute of Technology, on tree positional encoding for the DOM. Title unconfirmed; its negative result should be re-checked against the thesis.", "S"),
    ("Shaw, Uszkoreit & Vaswani (2018)", "Self-Attention with Relative Position Representations. NAACL 2018. arXiv:1803.02155.", "P"),
    ("Shiv & Quirk (2019)", "Novel Positional Encodings to Enable Tree-Based Transformers. NeurIPS 2019.", "P"),
    ("Su (2023)", "ReRoPE: rectified rotary position embeddings (blog post, kexue.fm), a training-free context extension that caps relative positions.", "S"),
    ("Su, Lu, Pan, Murtadha, Wen & Liu (2024)", "RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568:127063. arXiv:2104.09864.", "P"),
    ("Sun, Deng, Nie & Tang (2019)", "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. ICLR 2019. arXiv:1902.10197.", "P"),
    ("TableLoRA (2025)", "Low-rank adaptation with table-structure encoding. ACL 2025, pp. 22376–22391.", "P"),
    ("Team Cohere (2025)", "Command A: An Enterprise-Ready Large Language Model. arXiv:2504.00698.", "P"),
    ("Tian, Han, Wang & Liao (2025)", "Sheetpedia: A 300K-Spreadsheet Corpus for Spreadsheet Intelligence and LLM Fine-Tuning. NeurIPS 2025 Datasets and Benchmarks. Source of the spreadsheets.", "P"),
    ("Trans+C-Embedding (2024)", "Coreference-aware position encoding. TACL 2024. arXiv:2404.19505.", "P"),
    ("Tree-Enhanced CodeBERTa (2025)", "Seamlessly Integrating Tree-Based Positional Embeddings into Transformer Models for Source Code Representation. arXiv:2507.04003. Authors unconfirmed.", "S"),
    ("Vaswani et al. (2017)", "Attention Is All You Need. NeurIPS 2017. arXiv:1706.03762.", "P"),
    ("Wang et al. (2024)", "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (M-RoPE). arXiv:2409.12191.", "P"),
    ("Wang, Dong, Jia, Li, Fu, Han & Zhang (2021)", "TUTA: Tree-based Transformers for Generally Structured Table Pre-training. KDD 2021. arXiv:2010.12537.", "P"),
    ("Wang, Tu, Wang & Shi (2019)", "Self-Attention with Structural Position Representations. EMNLP 2019. arXiv:1909.00383.", "P"),
    ("Wei et al. (2025)", "VideoRoPE: What Makes for Good Video Rotary Position Embedding? ICML 2025. arXiv:2502.05173.", "P"),
    ("Xu, Li, Cui, Huang, Wei & Zhou (2020)", "LayoutLM: Pre-training of Text and Layout for Document Image Understanding. KDD 2020. arXiv:1912.13318.", "P"),
    ("Xu, Liu, Yin, Zhou & Poovendran (2025)", "KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding. ACL 2025. arXiv:2503.02951. Source of Python programs.", "P"),
    ("Yang, Gupta et al. (2022)", "TableFormer: Robust Transformer Modeling for Table-Text Encoding. ACL 2022. arXiv:2203.00274.", "S"),
    ("Yang, Venkitesh, Talupuru, Lin, Cairuz, Blunsom & Locatelli (2025)", "Rope to Nope and Back Again: A New Hybrid Attention Strategy. arXiv:2501.18795.", "P"),
    ("Ying et al. (2021)", "Do Transformers Really Perform Bad for Graph Representation? (Graphormer). NeurIPS 2021. arXiv:2106.05234.", "P"),
    ("Yoshida, Someya & Oseki (2024)", "Tree-Planted Transformers: Unidirectional Transformer Language Models with Implicit Syntactic Supervision. Findings of ACL 2024. arXiv:2402.12691.", "P"),
    ("Zhang, Li, Jin & Li (2023)", "Implant Global and Local Hierarchy Information to Sequence based Code Representation Models (HiT). ICPC 2023. arXiv:2303.07826.", "P"),
    ("Zhang, Li, Zhang & Jin (2024)", "HiRoPE: Length Extrapolation for Code Models Using Hierarchical Position. ACL 2024. arXiv:2403.19115.", "P"),
    ("Zhao, Ren, Hessel, Cardie, Choi & Deng (2024)", "WildChat: 1M ChatGPT Interaction Logs in the Wild. ICLR 2024. arXiv:2405.01470. WildChat-1M, ODC-BY.", "P"),
    ("Zügner et al. (2021)", "Code Transformer. ICLR 2021. arXiv:2103.11318.", "S"),
]


def sec_refs(D) -> str:
    items = "".join(f"<li><strong>{esc(a)}</strong> {esc(b)} <span class=\"tag\">[{t}]</span></li>" for a, b, t in REFERENCES)
    return md("""**[P]** = checked against the primary source (the paper, its arXiv page or its
anthology page). **[S]** = known to us only through secondary summaries; its bibliographic details
should be treated as unconfirmed, and the text relies on it only where it says so.""") + f'<ul class="refs">{items}</ul>'


# =================================================================================== assembly

CSS = f"""
:root {{ color-scheme: light; --surface:{SURF}; --page:#f9f9f7; --ink:{INK}; --ink2:{INK2};
  --muted:{MUTED}; --grid:{GRID}; --axis:{AXIS}; --accent:{C_BLUE}; }}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
.draft {{ position:sticky; top:0; z-index:5; background:#3a3936; color:#fff; text-align:center;
  font-size:13px; letter-spacing:.04em; padding:5px 8px; }}
main {{ max-width: 980px; margin: 0 auto; padding: 24px 28px 80px; background:var(--surface); }}
h1 {{ font-size:28px; line-height:1.25; margin:18px 0 6px; }}
h2 {{ font-size:22px; margin:48px 0 10px; padding-top:12px; border-top:1px solid var(--grid); }}
h3 {{ font-size:17px; margin:30px 0 8px; }}
h4 {{ font-size:15px; margin:24px 0 6px; }}
h5 {{ font-size:14px; margin:18px 0 4px; color:var(--ink2); }}
p, li {{ max-width: 76ch; }}
.lede {{ color:var(--ink2); margin-top:0; }}
a {{ color:#1c5cab; }}
a.term {{ color:inherit; text-decoration: underline dotted var(--muted); text-underline-offset: 3px; }}
a.term:hover, a.term:focus {{ text-decoration-color: var(--ink); }}
code {{ font: 13.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; background:#f0efec; padding:1px 4px; border-radius:3px; }}
nav.toc {{ background:#f4f3ef; border-radius:8px; padding:12px 18px; margin:18px 0 8px; font-size:14.5px; }}
nav.toc ol {{ margin:4px 0; padding-left:22px; }}
nav.toc ol ol {{ padding-left:18px; }}
.status {{ border:1px solid var(--grid); border-left:4px solid var(--muted); border-radius:6px;
  padding:8px 14px; margin:14px 0; font-size:14.5px; background:#fbfbf9; }}
.status ul {{ margin:4px 0; }}
.tw {{ overflow-x:auto; margin:10px 0 14px; }}
table.data {{ border-collapse:collapse; font-size:14px; min-width:50%; }}
table.data th, table.data td {{ padding:5px 10px; border-bottom:1px solid var(--grid); text-align:left; vertical-align:top; }}
table.data th {{ color:var(--ink2); font-weight:600; border-bottom:1px solid var(--axis); }}
table.data td.n, table.data th.n {{ text-align:right; font-variant-numeric: tabular-nums; white-space:nowrap; }}
figure.fig {{ margin:22px 0 26px; padding:14px 16px 10px; border:1px solid rgba(11,11,11,.10); border-radius:10px; background:var(--surface); }}
figcaption.ft {{ font-weight:600; font-size:15.5px; margin-bottom:6px; }}
.fc {{ font-size:14px; color:var(--ink2); }}
.fc p {{ margin:6px 0; }}
svg.chart {{ width:100%; height:auto; display:block; }}
svg text {{ font-family: system-ui,-apple-system,"Segoe UI",sans-serif; fill:var(--ink2); }}
svg .tk {{ font-size:11px; fill:var(--muted); font-variant-numeric: tabular-nums; }}
svg .al {{ font-size:12px; fill:var(--ink2); }}
svg .dir {{ font-size:11.5px; fill:var(--muted); }}
svg .rl {{ font-size:13px; fill:var(--ink); }}
svg .rs {{ font-size:11px; fill:var(--muted); }}
svg .av {{ font-size:13px; fill:var(--ink); font-weight:600; font-variant-numeric: tabular-nums; }}
svg .lg {{ font-size:12px; fill:var(--ink2); }}
svg .el {{ font-size:12px; fill:var(--ink); paint-order:stroke; stroke:var(--surface); stroke-width:3px; }}
svg .hit, svg .band, svg .cell {{ cursor: default; outline:none; }}
svg .dot.on {{ stroke: var(--ink); fill-opacity:1; }}
svg .cell:hover, svg .cell:focus {{ stroke: var(--ink); stroke-width:1.5; }}
svg .hit:focus-visible, svg .band:focus-visible {{ stroke: var(--ink); stroke-width:1; }}
details.tv {{ margin-top:6px; font-size:13.5px; }}
details.tv summary {{ cursor:pointer; color:var(--ink2); }}
dl.gloss {{ display:grid; grid-template-columns: minmax(150px, 230px) 1fr; gap:6px 18px; font-size:14.5px; }}
dl.gloss dt {{ font-weight:600; }}
dl.gloss dd {{ margin:0; color:var(--ink2); }}
dl.gloss dt:target, dl.gloss dt:target + dd {{ background:#fdf2c4; }}
ul.refs {{ font-size:14px; padding-left:18px; }}
ul.refs li {{ margin-bottom:5px; }}
.tag {{ color:var(--muted); font-size:12px; }}
#tip {{ position:fixed; z-index:20; max-width:340px; background:#fff; color:var(--ink);
  border:1px solid rgba(11,11,11,.14); border-radius:8px; box-shadow:0 6px 24px rgba(0,0,0,.12);
  padding:8px 11px; font-size:13px; line-height:1.4; pointer-events:none; }}
#tip .t1 {{ font-weight:650; font-size:14px; margin-bottom:2px; font-variant-numeric: tabular-nums; }}
#tip div {{ color:var(--ink2); }}
#tip .t1 {{ color:var(--ink); }}
footer {{ margin-top:50px; font-size:12.5px; color:var(--muted); }}
@media print {{ .draft {{ position:static; }} details.tv {{ display:block; }} }}
"""

JS = """
(function () {
  const tip = document.getElementById('tip');
  let current = null;
  function place(x, y) {
    const r = tip.getBoundingClientRect();
    let L = x + 14, T = y + 16;
    if (L + r.width > innerWidth - 8) L = Math.max(8, x - r.width - 14);
    if (T + r.height > innerHeight - 8) T = Math.max(8, y - r.height - 14);
    tip.style.left = L + 'px'; tip.style.top = T + 'px';
  }
  function show(el, x, y) {
    const text = el.getAttribute('data-tip');
    if (!text) return;
    current = el;
    tip.textContent = '';
    text.split('\\n').forEach(function (line, i) {
      const d = document.createElement('div');
      if (i === 0) d.className = 't1';
      d.textContent = line;            // labels are data: never innerHTML
      tip.appendChild(d);
    });
    tip.hidden = false;
    place(x, y);
    if (el.dataset.mark) { const m = document.getElementById(el.dataset.mark); if (m) m.classList.add('on'); }
    if (el.dataset.xh) {
      const h = document.getElementById(el.dataset.xh);
      if (h) { h.setAttribute('x1', el.dataset.x); h.setAttribute('x2', el.dataset.x); h.setAttribute('visibility', 'visible'); }
    }
  }
  function hide(el) {
    if (el !== current) return;
    tip.hidden = true; current = null;
    if (el.dataset.mark) { const m = document.getElementById(el.dataset.mark); if (m) m.classList.remove('on'); }
    if (el.dataset.xh) { const h = document.getElementById(el.dataset.xh); if (h) h.setAttribute('visibility', 'hidden'); }
  }
  document.addEventListener('pointerover', function (e) {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el && el !== current) { if (current) hide(current); show(el, e.clientX, e.clientY); }
  });
  document.addEventListener('pointermove', function (e) { if (current) place(e.clientX, e.clientY); });
  document.addEventListener('pointerout', function (e) {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el && !(e.relatedTarget && el.contains(e.relatedTarget))) hide(el);
  });
  document.addEventListener('focusin', function (e) {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) { if (current) hide(current); const r = el.getBoundingClientRect(); show(el, r.right, r.bottom); }
  });
  document.addEventListener('focusout', function (e) {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) hide(el);
  });
})();
"""


def status_box(D) -> str:
    w = D["webl"]["runs"]
    fin = D["webl"]["final"]
    items = []
    for key, label in (("layer_split", "the larger web run layer split"), ("depth", "the larger web run sequence + depth")):
        if w[key]["last"] < fin:
            items.append(f"{label}: {pct(100 * w[key]['last'] / fin, 0)} of its steps")
    for key, label in (("sh_split", "Spreadsheets layer split"),
                       ("sh_rel", "Spreadsheets sequence + tree + depth"),
                       ("ws_rel", "web pages sequence + tree + depth"),
                       ("py_full", "Python full-width RoPE control"),
                       ("sh_full", "Spreadsheets full-width RoPE control"),
                       ("ws_full", "web pages full-width RoPE control")):
        done, n, running = D["prog"][key]
        if done < n:
            items.append(f"{label}: {done} of {n} seeds finished")
    if not items:
        return ""
    return ('<div class="status"><strong>Still training when this page was built.</strong> Results '
            'below use only finished runs; these are reported as in progress:<ul>'
            + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul></div>")



MOVED_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "appendix.md")


def sec_moved(D) -> str:
    """Render paper/appendix.md -- the material moved out of the body for conference length.

    Kept as markdown rather than regenerated here so that audit_paper_numbers.py can go on
    verifying every number in it against the run logs.
    """
    if not os.path.exists(MOVED_MD):
        return "<p>No moved material.</p>"
    src = re.sub(r"<!--.*?-->", "", open(MOVED_MD).read(), flags=re.S)
    out, buf, tbl = [], [], []

    def flush_buf():
        if buf:
            out.append(md("\n".join(buf)))
            buf.clear()

    def flush_tbl():
        if not tbl:
            return
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in tbl]
        rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
        if rows:
            out.append(table(rows[0], rows[1:]))
        tbl.clear()

    for line in src.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            flush_buf(); tbl.append(s); continue
        flush_tbl()
        if s.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
            if m:
                flush_buf()
                svg = m.group(2).split("/")[-1]
                out.append(f'<figure><img src="figures/{esc(svg)}" alt="{esc(m.group(1))}">'
                           f'</figure>')
            continue
        if s.startswith("#"):
            flush_buf()
            lvl = len(s) - len(s.lstrip("#"))
            out.append(f"<h{min(lvl+1,5)}>{inline(s.lstrip('# ').strip())}</h{min(lvl+1,5)}>")
            continue
        if not s:
            flush_buf()
            continue
        buf.append(s)
    flush_tbl(); flush_buf()
    return "\n".join(out)


SECTIONS = [
    ("read", "How to read the numbers", sec_read, []),
    ("moved", "Moved from the paper body", sec_moved, []),
    ("glossary", "Glossary", sec_glossary, []),
    ("setup", "Setup", sec_setup, []),
    ("mechanism", "Mechanism", sec_mechanism, []),
    ("results", "Results", None, [
        ("r-cheap", "Gain against cost: a few planes reach the full tree", res_cheap),
        ("r-recommended", "Sequence + tree + depth: best or tied on Python, spreadsheets and web pages", res_recommended),
        ("r-curves", "Over training, in every setting", res_curves),
        ("r-headline", "Adding the tree helps in every setting", res_headline),
        ("r-python", "Python: tree position against sequence position", res_python),
        ("r-web", "Web text: adding helps, replacing does not", res_web),
        ("r-within", "The within-node index is a stand-in for sequence position", res_within),
        ("r-splits", "One head or separate layers: it depends on the corpus", res_splits),
        ("r-depth", "Two planes of depth", res_depth),
        ("r-ref", "Reference rotation", res_reference),
        ("r-grid", "Spreadsheets: the grid levels carry the path's gain", res_grid),
        ("r-treegrid", "Trees: depth + rank reduces to depth", res_treegrid),
        ("r-labels", "Code's statement grid, and labels on depth", res_labels),
        ("r-where", "Where the gain lands, per kind of token", res_where),
        ("r-mechanism", "How the structure is used: attention and test-time removal", res_mechanism),
        ("r-sheets", "Spreadsheets, and a prediction recorded in advance", res_sheets),
        ("r-test", "A held-out check: validation and test splits", res_test),
        ("r-fullrope", "Is the 16-plane baseline handicapped?", res_fullrope),
        ("r-controls", "Controls: information or delivery?", res_controls),
        ("r-chat", "Chat (WildChat): the grid carries it, depth adds to it", res_chat),
        ("r-generation", "Generation: depth must be in the text or supplied", res_generation),
        ("r-cells", "One model, many corpora; and two stripped-text workarounds", res_cells),
    ]),
    ("protocol", "Evaluation protocol and pitfalls", sec_protocol, []),
    # Related work, Limitations, Future work and a second reference list were dropped
    # 2026-10-13: each duplicated the paper rather than supporting it, and the appendix's
    # job is the per-seed detail the body has no room for. The generators remain below in
    # case they are wanted again.
]


def build(D) -> str:
    toc, body = [], []
    for i, (sid, title, fn, subs) in enumerate(SECTIONS, 1):
        sub_toc = "".join(f'<li><a href="#{s}">{esc(t)}</a></li>' for s, t, _ in subs)
        toc.append(f'<li><a href="#{sid}">{esc(title)}</a>' + (f"<ol>{sub_toc}</ol>" if subs else "") + "</li>")
        body.append(f'<section id="{sid}"><h2>{i}. {esc(title)}</h2>')
        if fn:
            body.append(fn(D))
        for j, (s, t, f) in enumerate(subs, 1):
            body.append(f'<section id="{s}"><h3>{i}.{j} {esc(t)}</h3>{f(D)}</section>')
        body.append("</section>")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Appendix to “{esc(PAPER_TITLE)}”</title>
<style>{CSS}{IC_CSS}</style></head>
<body>
<div class="draft" role="note">Preprint, 2026-10-08 · <a href="https://github.com/wrodenbusch/depth-perception">code</a> · <a href="https://github.com/wrodenbusch/depth-perception/blob/main/paper/depth-perception.pdf">paper (PDF)</a></div>
<main>
<h1>Appendix to “{esc(PAPER_TITLE)}”</h1>
<p class="lede">Interactive appendix: setup, mechanism, every configuration and every seed</p>
{sec_about(D)}
{status_box(D)}
<nav class="toc" aria-label="Contents"><strong>Contents</strong><ol>{''.join(toc)}</ol></nav>
{''.join(body)}
<footer>Generated from the training logs by the appendix build script; status as of {stamp}.
Every result number on this page is recomputed at build time.</footer>
</main>
<div id="tip" role="tooltip" hidden></div>
<script>{JS}{IC_JS}</script>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--publish", action="store_true", help=f"also copy the page to {PUBLISH_DIR}")
    a = ap.parse_args()
    D = collect()
    page = build(D)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "index.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KiB)")
    if a.publish:
        os.makedirs(PUBLISH_DIR, exist_ok=True)
        shutil.copy2(out, os.path.join(PUBLISH_DIR, "index.html"))
        print(f"published to {PUBLISH_DIR}/index.html")


if __name__ == "__main__":
    main()
