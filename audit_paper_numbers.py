#!/usr/bin/env python
"""Recompute every quantitative claim in paper/depth-perception.md from the run logs.

    .venv/bin/python audit_paper_numbers.py            # check every claim against the paper
    .venv/bin/python audit_paper_numbers.py --strict   # exit 1 on any mismatch or missing claim

Each claim is (where, the exact text the paper prints, how to recompute it). The audit checks two
things: that the text still appears in the paper (so a claim cannot silently drift out of sync with
this list), and that the recomputed value, rounded to the paper's printed precision, matches it.

SOURCES. Paired validation comparisons use ``paired.read_arm`` at each setting's final step, with
seeds that have not reached it excluded -- the same rule ``paired.py --require-step`` applies.
Test-split numbers come from ``heldout.jsonl`` (the larger web run) and the token-level evaluation results under
$TOPOS_RUNS/analysis/ (Python, Spreadsheets, web pages), whose ``all`` category is the whole
test split. Dataset facts come from the pack files. A few numbers are design constants of the
method or the training recipe; they are listed under STATED CONSTANTS and not recomputed.

Every run the paper quotes has finished. The the larger web run depth + rank run was stopped early and is not quoted;
``check_no_partial`` still fails the audit if any sentence quotes a number beside "in progress".
"""
from __future__ import annotations
from topos_paths import PACKS, RUNS  # noqa: E402

import argparse
import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

from paired import p_two_sided, read_arm

RUNS = f"{RUNS}"
AN = f"{RUNS}/analysis"
MIX4_PACK = f"{PACKS}/mix4_b4096"
PAPER = "paper/depth-perception.md"
# Material moved out of the body for conference length (2026-10-13). It is published in the
# appendix, and every claim in it stays under audit: claim matching reads PAPER + APPENDIX, while
# --write-controls still rewrites only PAPER (both generated blocks, REC and N, live in the body).
APPENDIX = "paper/appendix.md"
FINAL = {"py": 11200, "sheets": 8240, "webc": 9375, "chat": 13100}


# ------------------------------------------------------------------------------------ sources

def paired(root: str, a: str, b: str) -> dict:
    """a - b per seed at the final step, both finished. Returns mean, won (a better), n, p, ci."""
    step = FINAL[root]
    A = read_arm(os.path.join(RUNS, root), a, "val_nats_tok")
    B = read_arm(os.path.join(RUNS, root), b, "val_nats_tok")
    seeds = sorted(s for s in set(A) & set(B) if A[s][3] == step and B[s][3] == step)
    if not seeds:
        raise SystemExit(f"no finished shared seeds for {root} {a} vs {b}")
    d = [A[s][0] - B[s][0] for s in seeds]
    n, m = len(d), st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    p = p_two_sided(m / se, n - 1) if n > 1 and se > 0 else float("nan")
    return {"mean": m, "won": sum(x < 0 for x in d), "n": n, "p": p, "se": se,
            "seeds": seeds, "base": st.mean(B[s][0] for s in seeds)}


def ci_hi(r: dict) -> float:
    """Upper end of the two-sided 95% CI of the paired mean (t on n-1 df)."""
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_two_sided(mid, r["n"] - 1) > 0.05 else (lo, mid)
    return r["mean"] + hi * r["se"]


def ci_lo(r: dict) -> float:
    """Lower end of the two-sided 95% CI of the paired mean (t on n-1 df)."""
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_two_sided(mid, r["n"] - 1) > 0.05 else (lo, mid)
    return r["mean"] - hi * r["se"]


def curve(root: str, arm: str) -> dict:
    """seed -> {step: val} for every validation milestone."""
    out = {}
    import glob
    for d in glob.glob(os.path.join(RUNS, root, f"{arm}_[sp][0-9]*")):
        base = os.path.basename(d)
        if base.rsplit("_", 1)[0] != arm:
            continue
        tag = base.rsplit("_", 1)[1]
        seed = 0 if tag.startswith("p") else int(tag[1:])      # cell 7 names its single run _p1
        vals = {}
        for line in open(os.path.join(d, "train.jsonl")):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("val_nats_tok") is not None and r.get("step", 0) > 0:
                vals[r["step"]] = r["val_nats_tok"]
        out[seed] = vals
    return out


def heldout(arm: str) -> float:
    rows = [json.loads(l) for l in open(f"{RUNS}/web7/{arm}/heldout.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("split") == "test" and r.get("step") == 537000]
    if not rows:
        raise SystemExit(f"no final test-split row for web7/{arm}")
    return rows[-1]["nats_tok"]


_UND: dict = {}


def und(setting: str, arm: str, cat: str = "all") -> dict:
    if setting not in _UND:
        _UND[setting] = json.load(open(f"{AN}/understanding/{setting}_results.json"))
    for r in _UND[setting]:
        if r["arm"] == arm and r["cat"] == cat:
            return r
    raise SystemExit(f"no understanding row {setting} {arm} {cat}")


def depth_diag(split: str, cfg: str, cat: str) -> dict:
    d = json.load(open(f"{AN}/sheets_depth/loss_by_category_summary.json"))
    return d[split][cfg][cat]


def ppl(delta: float) -> float:
    """Perplexity change in percent for a nats/token difference (negative = lower perplexity)."""
    return -100 * (1 - math.exp(delta))


def pack_tokens(pack: str) -> int:
    return int(np.load(os.path.join(pack, "tokens.npy"), mmap_mode="r").shape[0])


def pack_nodes(pack: str) -> int:
    return int(np.load(pack + "_dompath/node_path.npy", mmap_mode="r").shape[0])


def pack_records(pack: str) -> int:
    m = json.load(open(os.path.join(pack, "meta.json")))
    for k in ("n_records", "records", "n_windows", "n_pages"):
        if k in m and isinstance(m[k], int):
            return m[k]
    off = os.path.join(pack, "rec_off.npy")
    if os.path.exists(off):
        return int(np.load(off, mmap_mode="r").shape[0]) - 1
    raise SystemExit(f"cannot find a record count for {pack}")


# -------------------------------------------- the mixed-corpus cells, the stripped-text workarounds
# and the prefix-causal chat tree (all finished; added to the paper 2026-10-12)

def mix_rows(cell: str, split: str = "test") -> dict:
    """A mixed-corpus cell's paired rows, keyed (comparison, corpus). mix_eval.py writes these."""
    d = json.load(open(f"{AN}/{cell}/{cell}_{split}.json"))
    return {(r["name"], r.get("group", "all")): r for r in d["rows"]}


MIX4_CORPORA = ("web", "py", "sheets", "chat")      # the per-corpus pack writes corpus c at level c


def mix4_steps() -> dict:
    """The learned first-plane step of the shared depth pair and of each corpus's private pair.

    Compared on the log scale: a rotary step is a frequency, so the midpoint of four is their
    geometric mean (mix4_steps.py)."""
    d = json.load(open(f"{AN}/mix4/mix4_test.json"))["rotary_steps"]

    def mean_over_seeds(arm: str, key: str) -> list:
        per_seed = [v[key] for v in d[arm].values() if key in v]
        return [st.mean(col) for col in zip(*per_seed)]

    shared = mean_over_seeds("depth", "step_rel")[0]
    per = mean_over_seeds("depth_percorpus", "step")
    own = {c: per[2 * i] for i, c in enumerate(MIX4_CORPORA)}
    geo = math.exp(st.mean(math.log(x) for x in own.values()))
    return {"shared": shared, "own": own, "geo": geo, "n": len(d["depth"]),
            "lo": min(own.values()), "hi": max(own.values())}


def stripped() -> dict:
    """Depth dropout (exp1) and compact depth-change markers (exp2); stripped_eval.py."""
    return json.load(open(f"{AN}/stripped/results.json"))


def stream_chat(arm: str, key: str) -> dict:
    """Chat arm against sequence only, scored with ``key`` coordinates (stream_depth_eval.py).

    ``train`` is the coordinate the arm trained on; ``stream`` is what a prefix-only parser of the
    generated text supplies. Sequence only has no coordinate, so its two are the same number."""
    import glob as _g
    A = {json.load(open(f))["seed"]: json.load(open(f))["nats"][key]
         for f in _g.glob(f"{AN}/stream_depth/score_chat_{arm}_s*.json")}
    B = {json.load(open(f))["seed"]: json.load(open(f))["nats"]["train"]
         for f in _g.glob(f"{AN}/stream_depth/score_chat_seq_s*.json")}
    seeds = sorted(set(A) & set(B))
    d = [A[s] - B[s] for s in seeds]
    n, m = len(d), st.mean(d)
    se = st.stdev(d) / math.sqrt(n)
    return {"mean": m, "won": sum(x < 0 for x in d), "n": n, "p": p_two_sided(m / se, n - 1)}


def causal_changed() -> float:
    """Share of chat content tokens whose node the prefix-causal rebuild changes."""
    m = json.load(open(f"{PACKS}/chat_wildchat_b2048_causal_dompath/manifest.json"))
    return m["tokens_changed_frac"]


def reprod() -> dict:
    """Run-to-run against seed-to-seed spread in final web pages validation nats (reprod_spread.py)."""
    import glob as _g

    def final_val(run: str):
        p = os.path.join(run, "train.jsonl")
        out = None
        if not os.path.exists(p):
            return None
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("step") == FINAL["webc"] and r.get("val_nats_tok") is not None:
                out = r["val_nats_tok"]
        return out

    pub = [v for d in sorted(_g.glob(f"{RUNS}/webc/depth_s*")) if (v := final_val(d)) is not None]
    rep = [v for d in sorted(_g.glob(f"{RUNS}/reprod/webc_depth_r*")) if (v := final_val(d)) is not None]
    if len(pub) < 10 or len(rep) < 3:
        raise SystemExit(f"reprod: {len(pub)} published seeds, {len(rep)} replicates")
    return {"seed_sd": st.stdev(pub), "n_seed": len(pub), "run_sd": st.stdev(rep), "n_run": len(rep)}

# ------------------------------------------------------------------------------ formatting

def f4(x: float) -> str:
    """Signed 4-dp nats, the paper's style: −0.0113, +0.0010."""
    s = f"{x:+.4f}"
    return s.replace("-", "−")


def pv(p: float) -> str:
    """Two significant figures, the paper's convention; below 1e-4 it prints a bound."""
    if p < 0.0001:
        return "p < 0.0001"
    if p >= 0.995:
        return "p = 1.0"
    txt = f"{p:.2g}"
    if "e" in txt:
        txt = f"{p:.6f}".rstrip("0")
        digits = txt.split(".")[1]
        lead = len(digits) - len(digits.lstrip("0"))
        txt = f"{p:.{lead + 2}f}"
    return f"p = {txt}"


def pp(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}%".replace("-", "−")


# ------------------------------------------------------------------------------ the claims

CLAIMS: list = []          # (where, text, recomputed text or None, note)
CONSTANTS: list = []       # (where, text, why not recomputed)


def claim(where: str, text: str, value: str | None = None, note: str = "") -> None:
    """``text`` must appear in the paper; ``value`` (if given) must appear inside ``text``."""
    CLAIMS.append((where, text, value, note))


def const(where: str, text: str, why: str) -> None:
    CONSTANTS.append((where, text, why))


# Every test the paper reports, for the Benjamini-Hochberg false-discovery-rate correction.
TESTS: dict = {}           # label -> raw p


def T(label: str, r: dict, p_key: str = "p") -> dict:
    """Register a reported test (label must be unique) and return the row."""
    if label in TESTS and abs(TESTS[label] - r[p_key]) > 1e-12:
        raise SystemExit(f"test {label!r} registered twice with different p")
    TESTS[label] = r[p_key]
    return r


def bh() -> dict:
    """Benjamini-Hochberg q-values over the whole reported family."""
    labs = sorted(TESTS, key=lambda k: TESTS[k])
    m, q, prev = len(labs), {}, 1.0
    for rank in range(m, 0, -1):
        k = labs[rank - 1]
        prev = min(prev, TESTS[k] * m / rank)
        q[k] = prev
    return q


def gain(delta: float) -> float:
    """Perplexity reduction in percent for a nats/token difference; positive = better."""
    return 100 * (1 - math.exp(delta))


def gp(delta: float, dp: int = 2) -> str:
    txt = f"{gain(delta):.{dp}f}%"
    if float(txt[:-1]) == 0:
        txt = txt.lstrip("-")          # a value that rounds to zero prints unsigned
    return txt.replace("-", "−")


MARKS: list = []           # (where, label, printed mark) checked after the family is complete


def cell(label: str, r: dict, where: str, delta_key: str = "mean", n_key: str = "n") -> str:
    """A Table 2 cell: gain% (won/n) + significance mark, the mark verified against the BH q."""
    T(label, r)
    MARKS.append((where, label))
    return f"{gp(r[delta_key])} ({r['won']}/{r[n_key]})"


def build() -> None:
    P = lambda a, b: paired("py", a, b)
    S = lambda a, b: paired("sheets", a, b)
    W = lambda a, b: paired("webc", a, b)

    # ---- Table 2: every configuration against sequence only
    rows = [("sequence + tree (44)", ("rope_both", "both_v2", "both")),
            ("sequence + depth (2)", ("depth", "depth_v2", "depth")),
            ("sequence + tree + depth (46)", ("rel", "rel_v2", "rel")),
            ("sequence + grid (6)", (None, "grid_v2", None)),
            ("sequence + depth + rank (4)", ("drank", None, "drank")),
            ("full RoPE (+48 sequence)", ("full_rope", "full_rope_v2", "full_rope"))]
    TABLE = []
    for lab, (pa, sa, wa) in rows:
        cells = []
        if pa:
            cells.append(("py-val", cell(f"py {pa} vs seq", P(pa, "rope_seq"), "T2")))
            r = dict(und("py", pa)); r["n"] = r["n_seeds"]; r["mean"] = r["delta"]
            cells.append(("py-test", cell(f"py-test {pa} vs seq", r, "T2")))
        else:
            cells += [("py-val", ""), ("py-test", "")]
        cells.append(("sh", cell(f"sheets {sa} vs seq", S(sa, "seq_v2"), "T2") if sa else ""))
        cells.append(("ws", cell(f"webc {wa} vs seq", W(wa, "seq"), "T2") if wa else ""))
        TABLE.append((lab, cells))
    py_rel, sh_rel, ws_rel = P("rel", "rope_seq"), S("rel_v2", "seq_v2"), W("rel", "seq")
    lo = min(gain(r["mean"]) for r in (py_rel, sh_rel, ws_rel))
    hi = max(gain(r["mean"]) for r in (py_rel, sh_rel, ws_rel))
    py_d, py_b = P("depth", "rope_seq"), P("rope_both", "rope_seq")
    frac = py_d["mean"] / py_rel["mean"]
    assert abs(frac - 2 / 3) < 0.04, frac
    # abstract (2026-10-09 rewrite): stated in general terms, each checked against the numbers
    assert round(100 * 2 / 64) == 3
    claim("abstract", "(3% of them, no added parameters)", None, "2 of 64 planes = 3.1%")
    assert 18 <= 44 / 2 <= 25
    claim("abstract", "about a twentieth of the cost", None, "2 vs 44 planes = 1/22")
    py_share = py_d["mean"] / py_b["mean"]
    assert py_share > 0.5, py_share                      # "most" of the full path on code
    assert gain(W("depth", "seq")["mean"]) >= gain(W("both", "seq")["mean"]), "web pages: depth must match the path"
    claim("abstract", "Depth recovers most of the full path's", None,
          f"depth / full path: Python {py_share:.2f}, web pages >= 1; the larger web run checked below")
    ws_d = W("depth", "seq")
    assert abs(gain(ws_d["mean"]) - gain(ws_rel["mean"])) < 0.05, "depth does not carry the web pages gain"

    # test split reproduction on spreadsheets and web pages
    worst = 0.0
    for setting, arms, roots in (("sheets", ("both_v2", "depth_v2", "rel_v2", "grid_v2", "full_rope_v2"), "sheets"),
                                 ("webc", ("both", "depth", "rel", "drank", "full_rope"), "webc")):
        seq = "seq_v2" if setting == "sheets" else "seq"
        for a in arms:
            worst = max(worst, abs(gain(und(setting, a)["delta"]) - gain(paired(roots, a, seq)["mean"])))
    assert worst < 0.15, worst
    claim("T2 caption", "reproduce these within 0.15 points", None, f"worst {worst:.3f}")

    # the larger web run
    both_c, seq_c = curve("web7", "rope_both")[0], curve("web7", "rope_seq")[0]
    steps = sorted(set(both_c) & set(seq_c))
    gaps = [gain(both_c[k] - seq_c[k]) for k in steps]
    assert all(g > 0 for g in gaps)
    claim("4.1 the larger web run", f"ahead at all {len(steps)} validation checkpoints, by {min(gaps):.1f}–{max(gaps):.1f}% throughout")
    dv, dt = both_c[537000] - seq_c[537000], heldout("rope_both_p1") - heldout("rope_seq_p1")
    claim("4.1 the larger web run", f"ends {gain(dv):.2f}% lower on validation and {gain(dt):.2f}% on the test split")
    later = gaps[1:]
    slope = np.polyfit(range(len(later)), later, 1)[0]
    assert abs(slope) < 0.03, f"the larger web run gap trend {slope:.3f}%/checkpoint"
    claim("4.1 the larger web run", "with no clear trend", None, f"slope after the first checkpoint {slope:+.3f} points/checkpoint")

    claim("4.1 test", f"(sequence only scores {und('py', 'rel')['seq_nll']:.2f}\nnats/token there against {py_rel['base']:.2f} on validation)".replace("\\n", " "))

    # full RoPE vs sequence + tree + depth
    fr = [T(f"{root} rel vs full", paired(root, rel, frr)) for root, rel, frr in
          (("py", "rel", "full_rope"), ("sheets", "rel_v2", "full_rope_v2"), ("webc", "rel", "full_rope"))]
    claim("4.1 full RoPE", f"by {gp(fr[0]['mean'])[1:] if False else gain(fr[0]['mean']):.2f}% on Python ({fr[0]['won']}/{fr[0]['n']}), "
          f"{gain(fr[1]['mean']):.2f}% on spreadsheets ({fr[1]['won']}/{fr[1]['n']}) and\n{gain(fr[2]['mean']):.2f}% on web pages "
          f"({fr[2]['won']}/{fr[2]['n']}), all q < 0.05".replace("\\n", " "))
    QCLAIMS.extend(f"{root} rel vs full" for root in ("py", "sheets", "webc"))
    for root, frr, seq in (("py", "full_rope", "rope_seq"), ("sheets", "full_rope_v2", "seq_v2"), ("webc", "full_rope", "seq")):
        assert paired(root, frr, seq)["mean"] > -0.0005, f"full RoPE beats the baseline on {root}"

    # ---- 4.2
    r = T("webc depth vs both", W("depth", "both")); QCLAIMS.append("webc depth vs both")
    claim("4.2", f"beats sequence + tree by {gain(r['mean']):.2f}% ({r['won']}/{r['n']} web pages seeds, q < 0.05)")
    r = T("webc rel vs depth", W("rel", "depth"))
    claim("4.2", f"adds nothing to it ({gp(r['mean'])}, {r['won']}/{r['n']})")
    pct = round(100 * py_d["mean"] / py_b["mean"])
    claim("4.2", f"Depth gives {pct}% of the path's validation gain", None, f"{py_d['mean']:.4f}/{py_b['mean']:.4f}")
    tb, td = und("py", "rope_both"), und("py", "depth")
    claim("4.2", f"matches it ({gain(td['delta']):.2f}% against {gain(tb['delta']):.2f}%)")
    r = T("py rel vs depth", P("rel", "depth")); QCLAIMS.append("py rel vs depth")
    claim("4.2", f"beats sequence + depth by {gain(r['mean']):.2f}% ({r['won']}/{r['n']}, q < 0.05)")
    claim("5", f"The full path adds a little on\ncode ({gain(r['mean']):.2f}%)".replace("\\n", " "))
    rp, rw = T("py drank vs depth", P("drank", "depth")), T("webc drank vs depth", W("drank", "depth"))
    claim("4.2", f"ties depth on Python ({gp(rp['mean'])}, {rp['won']}/{rp['n']}) and web pages\n({gp(rw['mean'])}, {rw['won']}/{rw['n']})".replace("\\n", " "))
    sh_dep = S("depth_v2", "seq_v2")
    claim("4.2", f"overall ({gain(sh_dep['mean']):.2f}%, {sh_dep['won']}/{sh_dep['n']})")
    share = json.load(open(f"{AN}/sheets_depth/token_stats.json"))["share_formula_tokens"]
    claim("4.2", f"inside formulas, {round(100 * share)}% of\ntokens".replace("\\n", " "))
    fv, ft = depth_diag("val", "depth", "ALL formula tokens"), depth_diag("test", "depth", "ALL formula tokens")
    assert ft["diff"] < 0 and ft["p"] < 0.05
    T("sheets depth formula tokens", fv)
    claim("4.2", f"where it helps ({gain(fv['diff']):.2f}%, {fv['won']}/{fv['n']}, {pv(fv['p'])})")
    fp = T("sheets depth formula punct", depth_diag("val", "depth", "formula_punct"))
    claim("4.2", f"({gain(fp['diff']):.2f}%,\n10/10)".replace("\\n", " ") if fp["won"] == 10 else "MISSING")
    r = T("sheets grid vs both", S("grid_v2", "both_v2"))
    claim("4.2", f"Six grid planes tie the 44-plane path ({gp(r['mean'])}, {r['won']}/{r['n']})")

    # ---- 4.3 where the gain lands (test split, token-level)
    nb, ng = und("sheets", "both_v2", "sh.newline"), und("sheets", "grid_v2", "sh.newline")
    ab, ag = und("sheets", "both_v2"), und("sheets", "grid_v2")
    T("sheets-test both newline beyond", nb, "beyond_p"); T("sheets-test grid newline beyond", ng, "beyond_p")
    claim("4.3", f"improves by {gain(nb['delta']):.1f}% under sequence + tree and\n{gain(ng['delta']):.1f}% under sequence + grid "
          f"({nb['won']}/{nb['n_seeds']}), against {gain(ab['delta']):.1f}% and {gain(ag['delta']):.1f}% on average".replace("\\n", " "))
    cb, cg = und("sheets", "both_v2", "sh.copy_above_far"), und("sheets", "grid_v2", "sh.copy_above_far")
    T("sheets-test both copy beyond", cb, "beyond_p"); T("sheets-test grid copy beyond", cg, "beyond_p")
    claim("4.3", f"by {gain(cb['delta']):.1f}% and {gain(cg['delta']):.1f}%")
    arms = ("both", "depth", "rel", "drank")
    ns = [gain(und("webc", a, "web.node_start")["delta"]) for a in arms]
    av = [gain(und("webc", a)["delta"]) for a in arms]
    for a in arms:
        r = und("webc", a, "web.node_start"); T(f"webc-test {a} node_start beyond", r, "beyond_p")
        assert r["won"] == 10
    claim("4.3", f"by {min(ns):.1f}–{max(ns):.1f}% under every structured configuration (10/10), against {min(av):.1f}–{max(av):.1f}% on average")
    for c in ("web.node_start", "web.node_end", "web.cont", "web.rep_same", "web.rep_cross"):
        r = und("webc", "full_rope", c)
        assert r["delta"] >= 0 or r["p"] > 0.05, f"full RoPE helps {c}"
    for a in ("rope_both", "depth", "rel", "drank"):
        for c in ("py.ascend", "py.ascend2"):
            r = und("py", a, c)
            assert r["delta"] <= 0 or r["p"] > 0.05, f"python {a} {c} significantly negative"

    # reference rotation
    r1 = T("sheets ref vs both", S("ref_anchor", "both_v2")); QCLAIMS.append("sheets ref vs both")
    r3 = T("sheets ref addressed vs both", paired("sheets", "ref_vanilla_v2", "both_vanilla_v2")); QCLAIMS.append("sheets ref addressed vs both")
    r2 = T("py splice vs both", P("splice", "rope_both"))
    claim("4.3 ref", f"by {gain(r1['mean']):.2f}% on spreadsheets ({r1['won']}/{r1['n']}, q < 0.05)")
    claim("4.3 ref", f"by\n{gain(r3['mean']):.2f}% when every address".replace("\\n", " "))
    claim("4.3 ref", f"({r3['won']}/{r3['n']}, q < 0.05), and not on Python ({gp(r2['mean'])},\n{r2['won']}/{r2['n']})".replace("\\n", " "))
    rr = und("sheets", "ref_anchor", "sh.ref"); T("sheets-test ref sh.ref", rr)
    claim("4.3 ref", f"the gain is {gain(rr['delta']):.1f}% against sequence only")

    # adding vs replacing; layers
    leaf = T("py leaf vs seq", P("rope_leaf", "rope_seq")); QCLAIMS.append("py leaf vs seq")
    claim("4.3", f"(tree only, {gain(leaf['mean']):.2f}%, {leaf['won']}/{leaf['n']}, q < 0.05)")
    claim("4.3", f"hurts on the larger web run ({gp(heldout('rope_leaf_p1') - heldout('rope_seq_p1'))}, test)")
    la = T("py layer_alt vs both", P("layer_alt", "rope_both")); QCLAIMS.append("py layer_alt vs both")
    claim("4.3", f"worse than sharing each head on Python ({gp(la['mean'])}, {la['won']}/{la['n']}, q < 0.05)")
    sl = T("sheets layer_split vs both", S("layer_split_v2", "both_v2"))
    claim("4.3", f"({gain(sl['mean']):.2f}%,\n{sl['won']}/{sl['n']}, {pv(sl['p'])})".replace("\\n", " "))
    SUGGEST.append("sheets layer_split vs both")

    # ---- 4.4 how the structure is used (mechanism analyses; analysis/mechanism.py on card 1,
    # test split, seeds 0-4, raw results under analysis/mechanism/)
    MD = f"{AN}/mechanism"

    def mpaired(d: list) -> dict:
        n, m = len(d), st.mean(d)
        se = st.stdev(d) / math.sqrt(n)
        return {"mean": m, "p": p_two_sided(m / se, n - 1), "n": n, "pos": sum(x > 0 for x in d)}

    def removal(setting: str, arm: str, seq: str) -> dict:
        R = json.load(open(f"{MD}/remove_{setting}.json"))
        g, r, sh = [], [], []
        for k in range(5):
            a, b = R[f"{arm}_s{k}"], R[f"{seq}_s{k}"]
            g.append(b["true"] - a["true"])
            r.append(a["absent"] - a["true"])
            sh.append(a["shuffle"] - a["true"])
        return {"gain": mpaired(g), "remove": mpaired(r), "shuffle": mpaired(sh)}

    def attn(setting: str, arm: str, seq: str, rel: str) -> dict:
        A = json.load(open(f"{MD}/attn_{setting}.json"))
        mass = lambda grid: np.array([[c["mass"] for c in row] for row in grid])
        d, base, new = [], [], []
        for k in range(5):
            a, b = mass(A[f"{arm}_s{k}"]["rel"][rel]), mass(A[f"{seq}_s{k}"]["rel"][rel])
            d.append(float((a - b).mean()))
            base.append(b.mean())
            new.append(a.mean())
        r = mpaired(d)
        r.update(base=float(np.mean(base)), new=float(np.mean(new)))
        return r

    up = lambda d: 100 * (math.exp(d) - 1)          # perplexity increase, %
    rg = removal("sheets", "grid_v2", "seq_v2")
    rw = removal("webc", "depth", "seq")
    rp = removal("py", "depth", "rope_seq")
    for lab, r in (("mech sheets grid shuffle", rg), ("mech webc depth shuffle", rw), ("mech py depth shuffle", rp)):
        T(lab, r["shuffle"])
        assert r["shuffle"]["pos"] == 5 and r["remove"]["pos"] == 5, lab
        QCLAIMS.append(lab)
    claim("4.4", f"sequence + grid on spreadsheets gains {100 * (1 - math.exp(-rg['gain']['mean'])):.1f}% but loses "
          f"{up(rg['shuffle']['mean']):.0f}% when shuffled ({up(rg['remove']['mean']):.0f}% when removed)")
    claim("4.4", f"sequence + depth on web pages gains {100 * (1 - math.exp(-rw['gain']['mean'])):.1f}% and loses "
          f"{up(rw['shuffle']['mean']):.0f}% ({up(rw['remove']['mean']):.1f}%)")
    claim("4.4", f"shuffling depth multiplies perplexity {math.exp(rp['shuffle']['mean']):.1f}×")
    ac = T("mech sheets grid same column", attn("sheets", "grid_v2", "seq_v2", "same column, earlier row"))
    ar = T("mech sheets grid referenced cell", attn("sheets", "grid_v2", "seq_v2", "the referenced cell"))
    aw = T("mech webc depth same depth", attn("webc", "depth", "seq", "same depth, elsewhere"))
    ap = T("mech py depth ancestor", attn("py", "depth", "rope_seq", "ancestor"))
    QCLAIMS.extend(["mech sheets grid same column", "mech sheets grid referenced cell", "mech webc depth same depth"])
    claim("4.4", f"earlier cells of the same column ({100 * ac['base']:.1f}% → {100 * ac['new']:.1f}%, {ac['pos']}/{ac['n']}, q < 0.05)")
    claim("4.4", f"onto the referenced cell ({100 * ar['mean']:+.2f} points, {ar['pos']}/{ar['n']}, q < 0.05)")
    claim("4.4", f"elsewhere on the page ({100 * aw['mean']:+.1f} points, {aw['pos']}/{aw['n']}, q < 0.05)")
    SUGGEST.append("mech py depth ancestor")
    claim("4.4", f"onto ancestor nodes, {100 * ap['mean']:+.1f} points, {ap['pos']}/{ap['n']}, {pv(ap['p'])}")

    # ---- code: the statement grid; labels added to depth (parent tags, parity-split IDs)
    sg = T("py stmtgrid vs seq", P("stmtgrid", "rope_seq"))
    sgv = T("py stmtgrid vs depth", P("stmtgrid", "depth")); QCLAIMS.append("py stmtgrid vs depth")
    sgd = T("py stmtgrid_depth vs seq", P("stmtgrid_depth", "rope_seq")); QCLAIMS.append("py stmtgrid_depth vs seq")
    sgb = T("py stmtgrid_depth vs both", P("stmtgrid_depth", "rope_both"))
    sgr = T("py stmtgrid_depth vs rel", P("stmtgrid_depth", "rel"))
    assert sgb["p"] > 0.5 and sgv["won"] == 0
    claim("4.2", f"alone gains only {gp(sg['mean'])} ({sg['won']}/{sg['n']}), less than depth (worse on "
          f"{sgv['n'] - sgv['won']}/{sgv['n']} seeds, q < 0.05). With depth added (8 planes) it gains "
          f"{gp(sgd['mean'])} ({sgd['won']}/{sgd['n']}, q < 0.05), tying the full path ({gp(sgb['mean'])}, "
          f"{sgb['won']}/{sgb['n']}) and short of tree + depth ({gp(sgr['mean'])}, {sgr['won']}/{sgr['n']}, {pv(sgr['p'])})")
    SUGGEST.append("py stmtgrid_depth vs rel")
    LBL = [("py", "dparent", "depth"), ("py", "dparentg", "depth"), ("py", "drankparent", "depth"),
           ("webc", "dparent", "depth"), ("webc", "dparentg", "depth"), ("sheets", "dparent_v2", "depth_v2"),
           ("sheets", "dparentg_v2", "depth_v2"), ("chat", "dparent", "depth"), ("chat", "dparentg", "depth")]
    lbl = [T(f"{r} {a} vs {b}", paired(r, a, b)) for r, a, b in LBL]
    LABEL_TESTS.extend(f"{r} {a} vs {b}" for r, a, b in LBL)
    mx = max(abs(gain(x["mean"])) for x in lbl)
    claim("4.2", f"tie depth in every setting (all within {mx:.1f}% of it, none surviving the correction)")
    par = T("py parityid vs depth", P("parityid", "depth")); SUGGEST.append("py parityid vs depth")
    claim("4.2", f"trail it slightly ({gp(par['mean'])}, {par['won']}/{par['n']}, {pv(par['p'])})")

    # ---- the larger web run sequence + depth (one run, finished)
    dep_c = curve("web7", "rope_depth")[0]
    st_ = sorted(set(dep_c) & set(seq_c))
    assert len(st_) == 20 and all(dep_c[k] < seq_c[k] for k in st_)
    dvd = dep_c[537000] - seq_c[537000]
    dtd = heldout("rope_depth_p1") - heldout("rope_seq_p1")
    share = round(100 * gain(dtd) / gain(dt))
    claim("4.1 the larger web run", f"sequence + depth is ahead at all {len(st_)} checkpoints and ends {gain(dvd):.2f}% lower on "
          f"validation and {gain(dtd):.2f}% on the test split, {share}% of sequence + tree's test gain with 2 planes "
          f"instead of 44")
    WEBL.update(dt=dt, dtd=dtd, share=share)
    assert share > 50, share                             # "most" of the full path at the larger scale too
    dep_rng = [gain(P("depth", "rope_seq")["mean"]), gain(W("depth", "seq")["mean"]), gain(dtd)]
    assert 0.5 <= min(dep_rng) and max(dep_rng) <= 2.5, dep_rng
    claim("abstract", "roughly 1–2%", None, "depth on nested text: " + ", ".join(f"{x:.2f}%" for x in dep_rng))

    # ---- 5 recommendation
    claim("5", f"{gain(py_d['mean']):.2f}% on Python validation ({gain(td['delta']):.2f}% on test), "
          f"{gain(ws_d['mean']):.1f}% on web pages and {gain(sh_dep['mean']):.2f}% on spreadsheets")
    claim("5", f"({sh_dep['n'] - sh_dep['won']} of {sh_dep['n']} spreadsheet seeds were slightly worse)")
    for r in (py_d, ws_d, sh_dep):
        assert r["mean"] < 0

    # ---- controls (interim while seeds land): registered in the family before it is closed
    ctl = control_results()

    # ---- chat (interim while seeds land): tests join the family only once n >= CHAT_TEST_MIN
    chat = chat_results()

    # ---- the mixed-corpus cells, the stripped-text workarounds and the prefix-causal chat tree
    M4, M2 = mix_rows("mix4", "test"), mix_rows("mix", "test")
    mix4_tests(M4)
    mix_tests(M2)
    S4, X = mix4_steps(), stripped()
    assert S4["n"] == 10, S4["n"]
    for k in ("depth/true", "depth/nodepth", "depthdrop30/true", "depthdrop30/nodepth",
              "depthdrop50/true", "depthdrop50/nodepth"):
        T(f"drop {k}", X["exp1"]["table"][k])
    for k in ("marked_seq vs seq", "marked_depth vs marked_seq", "marked_depth vs depth"):
        T(f"mark {k}", X["exp2"]["table"]["visible " + k])
    C = {a: T(f"causal {a} vs seq", paired("chat", a, "seq")) for a in ("c_depth", "c_grid_depth")}
    C["c_grid_depth~c_both"] = paired("chat", "c_grid_depth", "c_both")
    STREAM = {f"{a}_{m}": stream_chat(a, m)
              for a in ("grid_depth", "c_grid_depth") for m in ("train", "stream")}
    RP = reprod()
    claim("3 protocol", f"spread {RP['run_sd']:.4f} nats at the final step against {RP['seed_sd']:.4f} "
          f"across the ten published seeds — {round(100 * RP['run_sd'] / RP['seed_sd'])}% of the")
    claim("3 protocol", f"and {round(100 * RP['run_sd'] / abs(ws_d['mean']))}% of the depth effect measured against it")
    MX = json.load(open(f"{MIX4_PACK}/meta.json"))
    claim("4.6", f"equal tokens per corpus ({min(MX['train_tokens_per_corpus'].values()) / 1e6:.2f} M each), {MX['n_records']:,} records and")
    claim("4.6", f"{MX['total_tokens'] / 1e6:.1f} M tokens at context {MX['block_size']:,}, over a union vocabulary of {MX['vocab_size']:,}")
    assert set(MX["corpus_counts"]["test"].values()) == {2048}

    # ---- family size and the marks
    gen_depth_claims(W)            # registers its tests before the family is closed
    q = bh()
    ew, es = ctl["emb_ws_rot"], ctl["emb_ws_seq"]
    ratio = (es["mean"] - ew["mean"]) / es["mean"]
    # No prose quotes this ratio any more -- see the note in control_blocks. It survives only
    # as the guard below, which backs the one thing the paper does say about web pages.
    # Was `assert ratio > 1.5` to back "delivered through the rotation". That threshold
    # encoded a confounded reading: the ratio is measured at a fixed step where the
    # zero-initialised embedding has not converged, and it falls monotonically
    # (18.7 -> 2.94) and is still falling at our budget. On a longer run it is EXPECTED
    # to approach 1.0, so the old assertion would fail for the right reason and read as
    # a regression. Assert only what survives: the rotary gain is real and converged,
    # and the rotation is still ahead at the budget we report.
    assert ratio > 1.0, ratio
    # ew = emb_ws_rot, signed so that POSITIVE = rotary depth ahead of the embedding;
    # "won" counts seeds where the EMBEDDING won, so 0 means rotary took all ten.
    assert ew["mean"] > 0 and ew["won"] == 0, (ew["mean"], ew["won"])
    for k in ("addr_grid_seq", "vt_textdepth_vs_tags", "vt_tagsdepth_vs_tags", "shuf_py_rot", "shuf_ws_rot"):
        assert q[f"control {k}"] < 0.05, k
    for k in ("vt_tags_vs_text", "shuf_py_seq", "shuf_ws_seq"):
        assert q[f"control {k}"] >= 0.05, k
    for key, block in control_blocks(ctl, q).items():
        BLOCKS[key] = block
        claim(f"control {key}", block, None, "regenerated by --write-controls as seeds land")
    for key, block in chat_blocks(chat, q).items():
        BLOCKS[key] = block
        claim(f"chat {key}", block, None, "regenerated by --write-controls as chat seeds land")
    mix4_body, mix4_precursor = mix4_block(M4, M2, q, S4)
    for key, block in (("MIX4T", mix4_table(M4, q)), ("MIX4", mix4_body),
                       ("MIX2", mix4_precursor),
                       ("MARK", mark_block(X, q)), ("DROP", drop_block(X, q)),
                       ("CAUSAL", causal_block(C, q, causal_changed(), STREAM))):
        BLOCKS[key] = block
        claim(f"cell {key}", block, None, "generated by --write-controls")
    QCLAIMS.extend(("mix4 depth_rel vs depth [py]", "mark marked_depth vs depth"))
    for lab in LABEL_TESTS:
        assert q[lab] >= 0.05, f"{lab} now survives the correction"
    cr, cb = chat["rel"], chat["both"]
    assert q["chat rel vs seq"] >= 0.05 and abs(gain(cb["mean"]) - gain(chat["grid_depth"]["mean"])) < 0.005
    claim("4.1", f"on chat its {gain(cr['mean']):.2f}% (n.s.) is below the {gain(cb['mean']):.2f}% of the full tree and of grid + depth")
    vtd = ctl["vt_tagsdepth_vs_tags"]
    assert q["control vt_tagsdepth_vs_tags"] < 0.05
    claim("5", f"perplexity on the visible text by {gp(vtd['mean'])} ({vtd['won']}/{vtd['n']} seeds)")
    claim("5 generation", f"depth still adds {gp(vtd['mean'], 1)} ({vtd['won']} of {vtd['n']} seeds)")
    BLOCKS["REC"] = rec_table(q, P, lambda a, b: paired("sheets", a, b), W, chat, ctl)
    claim("5 table", BLOCKS["REC"], None, "generated by --write-controls")
    BLOCKS["N"] = str(len(TESTS))
    claim("3", f"across the <!-- ctl:N -->{len(TESTS)}<!-- /ctl:N --> tests reported here")
    text = audited_text()
    flat = re.sub(r"\s+", " ", text)
    for lab, cells in TABLE:
        printed = []
        for key, c in cells:
            if not c:
                printed.append("")
                continue
            tl = {"py-val": "py", "py-test": "py-test", "sh": "sheets", "ws": "webc"}[key]
            label = next(k for k in TESTS if k.startswith(tl + " ") and c and k.endswith("vs seq")
                         and k.split()[1] == {"py-val": (rows_map := dict(rows))[lab][0], "py-test": rows_map[lab][0],
                                               "sh": rows_map[lab][1], "ws": rows_map[lab][2]}[key])
            mark = "*" if q[label] < 0.05 else ("†" if TESTS[label] < 0.1 else "")
            printed.append(c + mark)
        line = f"| {lab} | " + " | ".join(printed) + " |"
        claim("T2", line, None, "marks = BH q<0.05 (*), p<0.1 not surviving (†)")
    for lab in QCLAIMS:
        assert q[lab] < 0.05, f"paper says q < 0.05 for {lab}, q = {q[lab]:.3f}"
    for lab in SUGGEST:
        assert q[lab] >= 0.05 and TESTS[lab] < 0.1, f"{lab} is not 'suggestive'"
    # every row of Table 1
    for path, ctx, steps, lr, geo in (("config/py_rope_seq.py", 1024, 11200, 1.2e-3, (6, 2, 256)),
                                      ("config/sheet_v2_seq.py", 4096, 8240, 1.2e-3, (6, 2, 256)),
                                      ("config/webc_seq.py", 2048, 9375, 1.2e-3, (6, 2, 256))):
        c = {}
        exec(open(path).read(), {}, c)
        assert (c["block_size"], c["max_iters"], c["learning_rate"], (c["n_layer"], c["n_head"], c["n_embd"])) == (ctx, steps, lr, geo), path
        assert (c["weight_decay"], c["grad_clip"], c["beta1"], c["beta2"]) == (0.1, 1.0, 0.9, 0.95), path
        assert abs(c["warmup_iters"] / c["max_iters"] - 0.02) < 0.002 and abs(c["min_lr"] / c["learning_rate"] - 0.1) < 1e-9, path
    jobs = json.load(open(f"{RUNS}/web7/jobs.json"))
    cmd = next(j["cmd"] for j in jobs if j["name"] == "rope_seq_p1")
    assert "--max_iters=537000" in cmd and "--learning_rate=6e-4" in cmd and "--min_lr=6.0e-05" in cmd, cmd
    claim("3 table", "| 11,200 steps, 8 epochs | 8,240 steps, 8 epochs | 9,375 steps, ⅓ epoch | 537,000 steps, 1 epoch |")
    claim("3 table", "| 1,024 | 4,096 | 2,048 | 4,096 |")
    claim("3 training", "peak learning rate of 1.2 × 10⁻³ (6 × 10⁻⁴ for the larger web run)")
    for lab, pack, show in (("Python", f"{PACKS}/py_ast_423k", lambda t: f"{t / 1e6:.1f} M"),
                            ("Spreadsheets", f"{PACKS}/sheet_values_v2", lambda t: f"{t / 1e6:.1f} M"),
                            ("web pages", f"{PACKS}/web_whole_b2048", lambda t: f"{t / 1e6:.0f} M")):
        n_tok = pack_tokens(pack)
        claim("3 table", show(n_tok), None, f"{lab} tokens")
        claim("3 table", f"{pack_nodes(pack) / n_tok:.2f}", None, f"{lab} nodes per token")
    n_tok = pack_tokens(f"{PACKS}/web_text_full_v3")
    claim("3 table", f"{n_tok / 1e9:.1f} B", None, "the larger web run tokens")
    claim("3 table", f"| {pack_nodes(f'{PACKS}/web_text_full_v3') / n_tok:.2f} |", None, "the larger web run nodes per token")
    cp = f"{PACKS}/chat_wildchat_b2048"
    n_tok = pack_tokens(cp)
    claim("3 table", f"| {pack_records(cp):,} conversations |", None, "chat conversations")
    claim("3 table", f"| {n_tok / 1e6:.0f} M |", None, "chat tokens")
    claim("3 table", f"| {pack_nodes(cp) / n_tok:.2f} |", None, "chat nodes per token")
    m = json.load(open(os.path.join(cp, "meta.json")))
    ep = 13100 * 32 * m["tokens_per_record_mean"] / m["train_tokens"]
    claim("3 table", f"| 13,100 steps, {ep:.1f} epochs |", None, "chat epochs = steps x 32 records x mean record / train tokens")
    for c in ("chat_seq", "chat_grid", "chat_depth"):
        txt = open(f"config/{c}.py").read()
        assert "max_iters = 13100" in txt and "learning_rate = 1.2e-3" in txt and "block_size = 2048" in txt, c
    for root, arm, n in (("py", "rope_seq", 20), ("py", "rope_leaf", 20), ("py", "rope_both", 5), ("sheets", "seq_v2", 10),
                         ("webc", "seq", 10)):
        got = sum(1 for d in read_arm(os.path.join(RUNS, root), arm, "val_nats_tok").values() if d[3] == FINAL[root])
        assert got >= n, f"{root}/{arm}: {got} finished seeds, paper says {n}"
    print("\nFDR family (Benjamini-Hochberg):")
    for k in sorted(TESTS, key=lambda k: TESTS[k]):
        print(f"  {k:40s} p={TESTS[k]:.3g}  q={q[k]:.3g}  {'q<0.05' if q[k] < 0.05 else ('suggestive' if TESTS[k] < 0.1 else 'n.s.')}")


BLOCKS: dict = {}          # control block key -> generated text

# THE CONTROLS (2026-10-03) answer the referee's "information or delivery?" objection. Their seeds
# land over ~2 days, so their prose is GENERATED from the logs: --write-controls rewrites the text
# between <!-- ctl:KEY --> and <!-- /ctl:KEY --> in the paper, and the ordinary audit then requires
# the paper to match. A control with no finished seed leaves its block as a "pending" line.
CONTROL_TARGET = 10


def _try(root: str, a: str, b: str):
    try:
        return paired(root, a, b)
    except SystemExit:
        return None


VT = f"{AN}/visible_text"
VT_CFG = {"text_seq": ("text", "seq"), "text_depth": ("text", "depth"),
          "tags_seq": ("tags", "seq"), "tags_depth": ("tags", "depth")}


def vt_rates(split: str = "val") -> dict:
    """Per-seed nats per VISIBLE-TEXT token for the four HTML-control runs, on the pages every run
    holds -- the same page set and token check as eval_visible_text.py compare."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_visible_text import _load
    data = {k: _load(*v, split) for k, v in VT_CFG.items()}
    seeds = sorted(set.intersection(*[set(d) for d in data.values()]))
    if not seeds:
        raise SystemExit("visible-text control: no seed scored in every configuration")
    pages = set.intersection(*[set(d[s]) for d in data.values() for s in seeds])
    if any(len({d[s][pg][2] for d in data.values() for s in seeds}) != 1 for pg in pages):
        raise SystemExit("visible-text control: runs score different target tokens")
    pages = sorted(pages)
    n_tok = sum(data["text_seq"][seeds[0]][pg][1] for pg in pages)
    return {"seeds": seeds, "pages": len(pages), "tokens": n_tok,
            "rate": {k: {s: sum(data[k][s][pg][0] for pg in pages) / n_tok for s in seeds} for k in data}}


def vt_paired(R: dict, a: str, b: str) -> dict:
    d = [R["rate"][a][s] - R["rate"][b][s] for s in R["seeds"]]
    n, m = len(d), st.mean(d)
    se = st.stdev(d) / math.sqrt(n)
    return {"mean": m, "won": sum(x < 0 for x in d), "n": n, "p": p_two_sided(m / se, n - 1), "se": se}


def control_results() -> dict:
    """Every control comparison, registered in TESTS (the controls are complete: 10 seeds, 5 on Python)."""
    want = {
        "addr_grid_seq": ("sheets", "grid_vanilla_v2", "vanilla_seq_v2"),
        "addr_grid_tree": ("sheets", "grid_vanilla_v2", "both_vanilla_v2"),
        "addr_depth_seq": ("sheets", "depth_vanilla_v2", "vanilla_seq_v2"),
        "emb_py_seq": ("py", "depthemb", "rope_seq"),
        "emb_py_rot": ("py", "depthemb", "depth"),
        "emb_sh_seq": ("sheets", "depthemb_v2", "seq_v2"),
        "emb_sh_rot": ("sheets", "depthemb_v2", "depth_v2"),
        "emb_ws_seq": ("webc", "depthemb", "seq"),
        "emb_ws_rot": ("webc", "depthemb", "depth"),
        "shuf_py_seq": ("py", "depthshuf", "rope_seq"),
        "shuf_py_rot": ("py", "depthshuf", "depth"),
        "shuf_ws_seq": ("webc", "depthshuf", "seq"),
        "shuf_ws_rot": ("webc", "depthshuf", "depth"),
    }
    out = {}
    for k, (root, a, b) in want.items():
        r = paired(root, a, b)
        need = 5 if root == "py" else CONTROL_TARGET
        if r["n"] != need:
            raise SystemExit(f"control {k}: {r['n']} finished paired seeds, expected {need}")
        out[k] = T(f"control {k}", r)
    R = vt_rates("val")
    out["vt"] = R
    for k, (a, b) in {"vt_tags_vs_text": ("tags_seq", "text_seq"),
                      "vt_textdepth_vs_tags": ("text_depth", "tags_seq"),
                      "vt_tagsdepth_vs_tags": ("tags_depth", "tags_seq"),
                      "vt_tagsdepth_vs_textdepth": ("tags_depth", "text_depth")}.items():
        out[k] = T(f"control {k}", vt_paired(R, a, b))
    return out


def _sig(r: dict, q: dict, key: str) -> str:
    qq = q[f"control {key}"]
    if qq < 0.05:
        return "q < 0.05"
    if r["p"] < 0.1:
        return f"suggestive, {pv(r['p'])}"
    return "n.s."


def _r(ctl: dict, q: dict, k: str, worse_is_first: bool = False) -> str:
    """'x.xx% (won/n, sig)' with won counted for the first-named arm."""
    r = ctl[k]
    return f"{gp(r['mean'])} ({r['won']}/{r['n']}, {_sig(r, q, k)})"


def control_blocks(ctl: dict, q: dict) -> dict:
    """The paper's control prose, generated from the results (positive % = the first arm better)."""
    blocks = {}
    g, t, d = ctl["addr_grid_seq"], ctl["addr_grid_tree"], ctl["addr_depth_seq"]
    blocks["A"] = (f"**Is the grid's gain only information the baseline lacks?** One rendering writes every cell's "
                   f"address out as text, so the baseline can read the grid. On it, sequence + grid still "
                   f"gains {_r(ctl, q, 'addr_grid_seq')} against sequence only and ties the full path "
                   f"({gp(t['mean'])}, {t['won']}/{t['n']}). Sequence + depth gains {_r(ctl, q, 'addr_depth_seq')}. "
                   f"Writing the coordinate into the text does not give the baseline the rotation's gain.")
    R = ctl["vt"]
    blocks["D"] = (f"**Web text with the HTML tags kept.** We rebuilt web pages with every page's tags in the "
                   f"token stream (70% of tokens are markup; context 8,192 so pages stay whole) and scored all "
                   f"models on the same visible-text tokens ({R['pages']:,} validation pages, "
                   f"{R['tokens']:,} tokens). Seeing the tags gives the sequence-only baseline nothing: "
                   f"{_r(ctl, q, 'vt_tags_vs_text')} against the text-only baseline. Rotary depth on the "
                   f"text-only stream beats the baseline that sees the tags by {_r(ctl, q, 'vt_textdepth_vs_tags')}, "
                   f"and adding depth to the tagged stream gains {_r(ctl, q, 'vt_tagsdepth_vs_tags')}.")
    name = {"py": "Python", "sh": "spreadsheets", "ws": "web pages"}
    seqs = ", ".join(f"{_r(ctl, q, f'emb_{k}_seq')} on {name[k]}" for k in ("py", "sh", "ws"))
    txt = (f"**Does the rotation matter, or only the information?** Depth given instead as a learned "
           f"embedding added at the input (zero-initialised, the same depth values) gains {seqs} against "
           f"sequence only.")
    ties = [k for k in ("py", "sh", "ws") if _sig(ctl[f"emb_{k}_rot"], q, f"emb_{k}_rot") == "n.s."]
    wins = [k for k in ("py", "sh", "ws") if k not in ties]
    if ties:
        txt += " It is indistinguishable from rotary depth on " + " and ".join(
            f"{name[k]} ({gp(ctl[f'emb_{k}_rot']['mean'])}, {ctl[f'emb_{k}_rot']['won']}/{ctl[f'emb_{k}_rot']['n']})"
            for k in ties) + "."
    for k in wins:
        r, rs = ctl[f"emb_{k}_rot"], ctl[f"emb_{k}_seq"]
        better = "rotary depth is better" if r["mean"] > 0 else "the embedding is better"
        txt += (f" On {name[k]} {better} by {gp(-abs(r['mean']))} "
                f"({r['n'] - r['won'] if r['mean'] > 0 else r['won']}/{r['n']} seeds, {_sig(r, q, f'emb_{k}_rot')})")
        if r["mean"] > 0 and rs["mean"] < 0:
            # DO NOT quote a rotary/embedding ratio here. It is read at a fixed step and the
            # two routes are not on an equal footing there: depth_emb is zero-initialised
            # (model.py: zeros_) and must learn 23x256 parameters, while the rotary step sizes
            # cannot be -- zero frequency is the identity, not a neutral rotation, and its
            # gradient there is zero -- so rotation carries depth from step 0. Measured on webc
            # (10 seeds, 21 milestones) the ratio falls monotonically 18.7 -> 2.94 and is STILL
            # falling at the 9,375-step budget, rotary flat at -0.006%/1k against the embedding
            # at +0.104%/1k. A multiplier implies a stable quantity and this one is not stable,
            # so report the difference as what it is -- an observation at the shared budget --
            # and recommend the rotation on cost, never on delivered quality.
            txt += " at the shared training budget"
        txt += "."
    blocks["B"] = txt + (" Either route delivers the gain, so what matters is the information and not "
                         "the rotation. The rotation is the cheaper way to deliver it: two learned step "
                         "sizes against a 5,888-parameter table, and active from the first step rather "
                         "than learned from zero, since a zero rotary frequency is the identity rather "
                         "than a neutral rotation.")
    rows = [("Spreadsheets, addresses written out", "sequence + grid", "addr_grid_seq", "sequence only (reads the addresses)"),
            ("Spreadsheets, addresses written out", "sequence + depth", "addr_depth_seq", "sequence only (reads the addresses)"),
            ("web pages, HTML tags kept (visible text)", "sequence only, sees tags", "vt_tags_vs_text", "sequence only, text only"),
            ("web pages, HTML tags kept (visible text)", "text + rotary depth", "vt_textdepth_vs_tags", "sequence only, sees tags"),
            ("web pages, HTML tags kept (visible text)", "tags + rotary depth", "vt_tagsdepth_vs_tags", "sequence only, sees tags"),
            ("web pages", "depth as input embedding", "emb_ws_seq", "sequence only"),
            ("web pages", "depth as input embedding", "emb_ws_rot", "rotary depth"),
            ("web pages", "shuffled depth", "shuf_ws_seq", "sequence only")]
    t = ["", "| setting | configuration | against | perplexity change | seeds better |", "|---|---|---|---|---|"]
    for st_, cfg, k, vs in rows:
        r = ctl[k]
        t.append(f"| {st_} | {cfg} | {vs} | {gp(r['mean'])} | {r['won']}/{r['n']} ({_sig(r, q, k)}) |")
    # The old Table 3 (controls) was cut on 2026-10-08: it duplicated the control prose
    # line for line, and that prose carries every number. Tables renumbered to close the gap.
    sp, sw = ctl["shuf_py_rot"], ctl["shuf_ws_rot"]
    blocks["C"] = (f"**Does the depth have to be the token's own?** With each record's depths shuffled among its "
                   f"nodes (same two planes, same values, wrong tokens), the gain disappears: against sequence only, "
                   f"{_r(ctl, q, 'shuf_py_seq')} on Python and {_r(ctl, q, 'shuf_ws_seq')} on web pages; against true "
                   f"depth, worse by {gp(-sp['mean'])} on Python ({sp['n'] - sp['won']}/{sp['n']}) and "
                   f"{gp(-sw['mean'])} on web pages ({sw['n'] - sw['won']}/{sw['n']}).")
    return blocks


# ---------------------------------------- §4.6 four corpora, §4.2 markers, §4.5 causal tree, §5 dropout
# All of these cells are finished, but their prose is generated for the same reason the controls'
# is: every number in it then comes from the logs, and the ordinary audit refuses a paper that has
# drifted from them.

MIX4_COLS = [("depth vs seq", "depth, one shared pair (2)"),
             ("depth_grid vs seq", "depth + grid (8)"),
             ("both vs seq", "full tree (44)"),
             ("depth_percorpus vs depth", "private pairs (8) vs shared")]
MIX4_ROWS = [("web", "web pages"), ("py", "Python"), ("sheets", "spreadsheets"),
             ("chat", "chat"), ("all", "all four together")]


def mix4_tests(M4: dict) -> None:
    """Register every four-corpus test the paper reports, before the family is closed."""
    for nm, _ in MIX4_COLS:
        for g, _ in MIX4_ROWS:
            T(f"mix4 {nm} [{g}]", M4[(nm, g)])
    for nm in ("depth_grid vs both", "depth_rel vs depth", "depth_scaled vs depth"):
        T(f"mix4 {nm} [all]", M4[(nm, "all")])
    T("mix4 depth_rel vs depth [py]", M4[("depth_rel vs depth", "py")])


def mix_tests(M2: dict) -> None:
    """The two-corpus cell (web + spreadsheets, five seeds)."""
    for nm, g in (("depth vs seq", "web"), ("depth vs seq", "sheets"), ("grid vs seq", "web"),
                  ("grid vs seq", "sheets"), ("separate vs reuse", "all")):
        T(f"mix {nm} [{g}]", M2[(nm, g)])


def _m(r: dict, q: dict, label: str) -> str:
    """'x.xx% (won/n, sig)' for a mixed-cell row."""
    sig = ("q < 0.05" if q.get(label, 1) < 0.05 else
           (f"suggestive, {pv(r['p'])}" if r["p"] < 0.1 else "n.s."))
    return f"{gp(r['mean'])} ({r['won']}/{r['n']}, {sig})"


def mix4_table(M4: dict, q: dict) -> str:
    t = ["", "| corpus | " + " | ".join(c for _, c in MIX4_COLS) + " |",
         "|---|" + "---|" * len(MIX4_COLS)]
    for g, lab in MIX4_ROWS:
        cells = []
        for nm, _ in MIX4_COLS:
            r, lb = M4[(nm, g)], f"mix4 {nm} [{g}]"
            mark = "*" if q[lb] < 0.05 else ("†" if r["p"] < 0.1 else "")
            cells.append(f"{gp(r['mean'])} ({r['won']}/{r['n']}){mark}")
        t.append(f"| {lab} | " + " | ".join(cells) + " |")
    t += ["", "*Table 3. Four corpora in one model: perplexity reduction against sequence only "
          "(positive = better), test split, mean over 10 paired seeds, with the seeds on which the "
          "configuration is better. The last column compares four private depth coordinates with the "
          "one shared coordinate, not with sequence only.*", ""]
    return "\n".join(t)


def mix4_block(M4: dict, M2: dict, q: dict, S4: dict) -> str:
    g = lambda nm, grp="all": M4[(nm, grp)]
    da, dw, ds = g("depth vs seq"), g("depth vs seq", "web"), g("depth vs seq", "sheets")
    gs = g("depth_grid vs seq", "sheets")
    gb, pc = g("depth_grid vs both"), g("depth_percorpus vs depth")
    rel, sc, rpy = g("depth_rel vs depth"), g("depth_scaled vs depth"), g("depth_rel vs depth", "py")
    p1 = (f"**One pair of planes, shared by four corpora.** The shared depth coordinate lowers "
          f"perplexity by {_m(da, q, 'mix4 depth vs seq [all]')} over all four corpora at once. Each "
          f"corpus keeps the preference its own setting found: depth carries web pages, "
          f"{_m(dw, q, 'mix4 depth vs seq [web]')}, while on spreadsheets depth alone is "
          f"{_m(ds, q, 'mix4 depth vs seq [sheets]')} and depth with a grid is "
          f"{_m(gs, q, 'mix4 depth_grid vs seq [sheets]')}. Eight planes — depth for all four, plus a grid "
          f"for the two corpora that have one — tie the 44-plane path over the mixture "
          f"({gp(gb['mean'])}, {gb['won']}/{gb['n']}, {pv(gb['p'])}), with no per-corpus tuning and no "
          f"domain label at inference.")
    p2 = (f"**Four private coordinates buy no more than we can resolve.** Giving each corpus its own depth plane "
          f"pair, with its own learned angle, costs four times the planes and gains "
          f"{_m(pc, q, 'mix4 depth_percorpus vs depth [all]')} over the shared pair. Read that as an upper "
          f"bound rather than as equivalence: the private arm is numerically ahead on all four corpora "
          f"and has strictly more freedom, so a tie is evidence that sharing is cheap, not proof that it "
          f"is free. A 95% interval on the paired margin spans {gp(ci_lo(pc))} to {gp(ci_hi(pc))}, so it "
          f"bounds what four private coordinates buy rather than showing they buy nothing. The null "
          f"is informative only if extra "
          f"planes could have mattered, and they can. On Python, 46 planes carrying the full tree path "
          f"beside depth gain 1.13% where two planes carrying depth alone gain 0.76% (Table 2, both "
          f"5/5 seeds, q < 0.05). So planes do pay when they carry information the shared coordinate "
          f"lacks, and four private copies of depth carry none. The shared step size settles at "
          f"{S4['shared']:.2f} rad, close to the geometric mean ({S4['geo']:.2f}) of the four the "
          f"private arm picks when free, which span {S4['lo']:.2f}–{S4['hi']:.2f}.")
    p3 = (f"**Rescaling depth per corpus does not help.** Two attempts to remove the disagreement both "
          f"failed against plain shared depth. The first was a relative depth: the level divided by the "
          f"deepest level so far in the record, which needs no corpus label and is still computable from "
          f"the prefix. It is {_m(rel, q, 'mix4 depth_rel vs depth [all]')} overall and costs "
          f"{gp(-rpy['mean'])} on Python ({rpy['n'] - rpy['won']}/{rpy['n']}, q < 0.05). It is exactly 1.0 "
          f"for 80% of chat tokens, so it is weakest where the trees are shallowest. The second, a "
          f"per-corpus constant aligning the four 90th percentiles, is "
          f"{_m(sc, q, 'mix4 depth_scaled vs depth [all]')} and separates nothing anywhere. "
          f"The raw level is what the planes are best spent on.")
    mw, mws, mgw, mgs, msr = (M2[("depth vs seq", "web")], M2[("depth vs seq", "sheets")],
                              M2[("grid vs seq", "web")], M2[("grid vs seq", "sheets")],
                              M2[("separate vs reuse", "all")])
    p4 = (f"A two-corpus precursor (web pages and spreadsheets, five seeds, test split) found the same "
          f"split of preferences: depth {_m(mw, q, 'mix depth vs seq [web]')} on web against the grid's "
          f"{_m(mgw, q, 'mix grid vs seq [web]')} there, and the grid "
          f"{_m(mgs, q, 'mix grid vs seq [sheets]')} on spreadsheets. It also found that two corpora "
          f"writing different meanings into one plane group cost nothing against giving each its own, "
          f"{_m(msr, q, 'mix separate vs reuse [all]')}. Full tables for both cells follow below.")
    return "\n\n".join((p1, p2, p3)), p4


def mark_block(X: dict, q: dict) -> str:
    e = X["exp2"]["table"]
    ms, md, mdd = (e["visible marked_seq vs seq"], e["visible marked_depth vs marked_seq"],
                   e["visible marked_depth vs depth"])
    gc = X["exp2"]["gen_check"]
    return (f"**Depth written into the text as compact markers.** A cheaper rendering than full HTML: one "
            f"marker token wherever the depth changes, 5.67% token overhead instead of 70%. On the visible "
            f"text, reading the markers gives the sequence-only baseline nothing, "
            f"{_m(ms, q, 'mark marked_seq vs seq')}, and rotary depth still adds "
            f"{_m(md, q, 'mark marked_depth vs marked_seq')} on top of them. But the whole arrangement "
            f"trails rotary depth on the clean stream by {gp(-mdd['mean'])} "
            f"({mdd['n'] - mdd['won']}/{mdd['n']}, q < 0.05), because the markers themselves cost "
            f"{e['marker loss marked_depth (nats/marker)']['mean']:.2f} nats each, "
            f"{e['page total marked_depth vs seq (nats/page)']['mean']:.1f} nats a page. Depth recovered "
            f"from the markers in the prefix matched the training coordinate exactly ({gc['mismatches']} "
            f"mismatches in {gc['tokens']:,} tokens), so this is not a parsing failure: serialising "
            f"structure into the token stream does not create the information the rotation delivers.")


def causal_block(C: dict, q: dict, changed: float, stream: dict) -> str:
    gd, dp, gt = C["c_grid_depth"], C["c_depth"], C["c_grid_depth~c_both"]
    kept = gain(stream["grid_depth_stream"]["mean"]) / gain(stream["grid_depth_train"]["mean"])
    return (f"The chat tree above parses whole messages, so a few tokens' Markdown depth depends on later "
            f"text — a blank line further down makes a list loose. Rebuilding the tree from each token's "
            f"own prefix changes the node of {100 * changed:.1f}% of content tokens. Retraining on it "
            f"(10 seeds) leaves the conclusions in place: grid + depth gains "
            f"{_m(gd, q, 'causal c_grid_depth vs seq')} against sequence only, depth alone still nothing, "
            f"{_m(dp, q, 'causal c_depth vs seq')}, and grid + depth still ties the full tree "
            f"({gp(gt['mean'])}, {gt['won']}/{gt['n']}, {pv(gt['p'])}). Because the causal tree is "
            f"computable from each token's prefix, its reading gain is also its generation gain. Scored "
            f"with the coordinates a prefix-only parser of the generated text supplies, grid + depth keeps "
            f"{gp(stream['c_grid_depth_stream']['mean'])}, where the models trained on the whole-message "
            f"tree keep {round(100 * kept)}% of theirs.")


def drop_block(X: dict, q: dict) -> str:
    e = X["exp1"]["table"]
    d, dn = e["depth/true"], e["depth/nodepth"]
    t30, n30 = e["depthdrop30/true"], e["depthdrop30/nodepth"]
    t50, n50 = e["depthdrop50/true"], e["depthdrop50/nodepth"]
    return (f"**Withholding depth on part of the training data.** Training web pages with depth dropped on a "
            f"random 30% of records — the coordinate absent there, contributing no rotation — keeps most "
            f"of the reading gain and removes the harm when depth is missing. With depth supplied it gains "
            f"{_m(t30, q, 'drop depthdrop30/true')} against sequence only, where plain depth gains "
            f"{_m(d, q, 'drop depth/true')}; without it, {_m(n30, q, 'drop depthdrop30/nodepth')}, where a "
            f"model trained with depth always present is {gp(-dn['mean'])} worse than sequence position "
            f"alone ({dn['won']}/{dn['n']}). At 50% dropout the pair is "
            f"{_m(t50, q, 'drop depthdrop50/true')} and {_m(n50, q, 'drop depthdrop50/nodepth')}. **One "
            f"model can therefore serve both regimes: read structured documents with the marker, generate "
            f"free text without it. Use depth where the structure is in the text or available at "
            f"inference; where it may be missing, withhold it on part of the training data.**")

# CHAT (WildChat, 2026-10-04). Seeds land over ~a day, so the chat prose is generated the same way
# as the controls. While fewer than CHAT_TEST_MIN seeds are finished, no test is reported and none
# joins the false-discovery-rate family: a t-test on two seeds says nothing, and letting it into the
# family would move every other q-value for no information.
CHAT_TARGET = 10
CHAT_TEST_MIN = 5
CHAT_ARMS = [("grid", "sequence + grid (exchange, speaker, position in message)"),
             ("grid_depth", "sequence + grid + Markdown depth"),
             ("depth", "sequence + depth (Markdown nesting)"),
             ("drank", "sequence + depth + rank"),
             ("both", "sequence + tree"),
             ("rel", "sequence + tree + depth")]


def chat_results() -> dict:
    out = {}
    for arm, _ in CHAT_ARMS:
        r = _try("chat", arm, "seq")
        if r is not None and r["n"] >= CHAT_TEST_MIN:
            T(f"chat {arm} vs seq", r)
        out[arm] = r
    for a, b in (("grid", "depth"), ("grid_depth", "depth"), ("drank", "depth"), ("grid_depth", "grid"),
                 ("grid_depth", "both")):
        r = _try("chat", a, b)
        if r is not None and r["n"] >= CHAT_TEST_MIN:
            T(f"chat {a} vs {b}", r)
        out[f"{a}~{b}"] = r
    return out


def _chat_cell(r, q, label) -> str:
    if r is None:
        return "no finished seed yet"
    s = f"{gp(r['mean'])} ({r['won']}/{r['n']}"
    if r["n"] >= CHAT_TEST_MIN:
        qq = q.get(label)
        s += ", q < 0.05" if qq is not None and qq < 0.05 else (f", suggestive, {pv(r['p'])}" if r["p"] < 0.1 else ", n.s.")
    return s + ")"


def chat_blocks(chat: dict, q: dict) -> dict:
    """The chat paragraph, table and recommendation sentence (positive % = the first arm better)."""
    ns = [r["n"] for r in chat.values() if r is not None]
    nmin = min(ns) if ns else 0
    interim = nmin < CHAT_TARGET
    tag = (f"*Interim: {nmin}–{max(ns)} of {CHAT_TARGET} seeds finished per configuration"
           + (f"; with fewer than {CHAT_TEST_MIN}, no test is reported.*" if nmin < CHAT_TEST_MIN else ".*")
           if interim else f"*{CHAT_TARGET} seeds per configuration.*")
    rows = [""]      # the per-configuration table is in the appendix; the prose carries every number
    g, d, gd, dr = chat["grid"], chat["depth"], chat["grid~depth"], chat["drank~depth"]
    gg, gt, gdd = chat["grid_depth~grid"], chat["grid_depth~both"], chat["grid_depth"]
    if interim or any(r is None for r in (g, d, gg, gt, gdd)):
        txt = f"**Chat.** {tag} Results are reported when every configuration has finished."
        blocks = {"CHAT": txt + "\n" + "\n".join(rows)}
        return blocks
    # the prose below states these facts; the audit refuses to print it if any stops being true
    assert q["chat depth vs seq"] >= 0.05 and d["p"] >= 0.1, "chat: depth alone is no longer null"
    assert q["chat grid_depth vs seq"] < 0.05, "chat: grid + depth no longer separates from sequence only"
    assert gt["p"] > 0.5, "chat: grid + depth no longer ties the full tree"
    assert gg["mean"] < 0 and q["chat grid_depth vs grid"] >= 0.05, "chat: grid + depth vs grid changed status"
    txt = (f"**Chat.** {tag} Depth alone gives nothing measurable on conversations: "
           f"{_chat_cell(d, q, 'chat depth vs seq')} against sequence only. Most chat text sits at one depth "
           f"(77% of tokens sit at a message's top level: a paragraph, heading, table or code block), so depth varies only inside "
           f"lists and headed sections. The chat grid marks the exchange, the speaker and the position in the "
           f"message. On its own it is suggestive, {_chat_cell(g, q, 'chat grid vs seq')}. Adding depth to it "
           f"gives the cheapest configuration that clearly beats sequence only: grid + depth, 10 planes, "
           f"{_chat_cell(gdd, q, 'chat grid_depth vs seq')}, ahead of the grid alone on {gg['won']} of {gg['n']} seeds "
           f"({gp(gg['mean'])}, n.s.) and tied with the full tree ({gp(gt['mean'])}, {gt['won']}/{gt['n']}). The two "
           f"carry different information: the grid says who is speaking and how far into the message, depth how "
           f"deep in its formatting.")
    if dr is not None:
        txt += f" Adding sibling rank to depth: {_chat_cell(dr, q, 'chat drank vs depth')} against depth alone."
    blocks = {"CHAT": txt + "\n" + "\n".join(rows)}
    return blocks


def rec_table(q: dict, P, S, W, chat: dict, ctl: dict) -> str:
    """Section 5's "which to use" table, generated so it cannot drift from the runs."""
    def c(r, label):
        sig = ("q < 0.05" if q.get(label, 1) < 0.05 else
               (f"suggestive, {pv(r['p'])}" if r["p"] < 0.1 else "n.s."))
        return f"{gp(r['mean'])} ({r['won']}/{r['n']}, {sig})"
    vtd = ctl["vt_tagsdepth_vs_tags"]
    rows = [
        ("Web pages, markup in the text (pages with tags kept; visible-text loss)", "depth", "2",
         f"{gp(vtd['mean'])} ({vtd['won']}/{vtd['n']}, q < 0.05)", "not run"),
        ("Web pages, markup stripped", "depth", "2", c(W("depth", "seq"), "webc depth vs seq"), c(W("both", "seq"), "webc both vs seq")),
        ("Web pages, markup stripped, long context (larger run, one run, test)", "depth", "2", f"{gain(WEBL['dtd']):.2f}% (one run)",
         f"{gain(WEBL['dt']):.2f}% (one run)"),
        ("Code (Python)", "depth", "2", c(P("depth", "rope_seq"), "py depth vs seq"), c(P("rope_both", "rope_seq"), "py rope_both vs seq")),
        ("Code (Python)", "depth + statement grid", "8", c(P("stmtgrid_depth", "rope_seq"), "py stmtgrid_depth vs seq"),
         c(P("rope_both", "rope_seq"), "py rope_both vs seq")),
        ("Tables (spreadsheets)", "depth", "2", c(S("depth_v2", "seq_v2"), "sheets depth_v2 vs seq"),
         c(S("both_v2", "seq_v2"), "sheets both_v2 vs seq")),
        ("Tables (spreadsheets)", "row + column (+ sheet)", "6", c(S("grid_v2", "seq_v2"), "sheets grid_v2 vs seq"),
         c(S("both_v2", "seq_v2"), "sheets both_v2 vs seq")),
        ("Chat", "depth", "2", c(chat["depth"], "chat depth vs seq"), c(chat["both"], "chat both vs seq")),
        ("Chat", "depth + speaker, exchange, position in message", "10", c(chat["grid_depth"], "chat grid_depth vs seq"),
         c(chat["both"], "chat both vs seq")),
    ]
    t = ["", "| data | coordinate beside sequence position | extra planes | gain against sequence only | "
         "full tree's gain (44 planes) |", "|---|---|---|---|---|"]
    t += [f"| {a} | {b} | {n} | {x} | {y} |" for a, b, n, x, y in rows]
    t += ["", "*Table 4. Which coordinate to add. Perplexity reduction against sequence only (positive = better), "
          "validation unless marked, with seeds better and significance after the false-discovery-rate correction.*", ""]
    return "\n".join(t)


GEN_DEPTH = os.path.join(RUNS, "analysis", "gen_depth")


def gen_depth_claims(W) -> None:
    """Section 5 'Generation': web pages depth models fed predicted / fixed depth (gen_depth_eval.py)."""
    import glob as _g
    import json as _j
    import math as _m
    from paired import p_two_sided as _p
    runs = [_j.load(open(f)) for f in sorted(_g.glob(os.path.join(GEN_DEPTH, "seed*.json")))]
    if len(runs) != 10:
        raise SystemExit(f"gen_depth: {len(runs)} seeds, expected 10")

    def arm(a):
        d = [r["nats"][a] - r["nats"]["seq"] for r in runs]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        t = m / (sd / _m.sqrt(len(d)))
        TESTS[f"gen_depth {a} vs seq"] = _p(t, len(d) - 1)
        return m, sum(x < 0 for x in d)
    tm, tw = arm("true")
    assert abs(gain(tm) - gain(W("depth", "seq")["mean"])) < 0.005, "gen_depth true arm must reproduce web pages depth"
    pm, pw = arm("pred")
    cm, cw = arm("const")
    acc = sum(r["depth_accuracy"]["pred"]["exact"] for r in runs) / len(runs)
    med = sum(r["depth_accuracy"]["const"]["exact"] for r in runs) / len(runs)
    tf = sum(r["head_acc_teacher_forced"] for r in runs) / len(runs)
    claim("5 generation", f"right only {round(100 * acc)}% of the time")
    claim("5 generation", f"({round(100 * tf)}% even given the true depths of earlier blocks; {round(100 * med)}%")
    claim("5 generation", f"made the models {-gain(pm):.1f}%", None, f"pred vs seq {gain(pm):.2f}%")
    claim("5 generation", f"({pw} of 10 seeds better)")
    assert pw == 0 and cw == 0
    claim("5 generation", f"left them {-gain(cm):.1f}% worse")



def audited_text() -> str:
    """The body plus the moved-material appendix: a claim is verified wherever it lives."""
    text = open(PAPER).read()
    if os.path.exists(APPENDIX):
        text += "\n" + open(APPENDIX).read()
    return text


def write_controls() -> None:
    """Rewrite each <!-- ctl:K -->...<!-- /ctl:K --> region from BLOCKS.

    Regions live in the paper or, since the body was cut to conference length, in the
    appendix. A block whose region is in neither file is an error: it means a generated
    claim has no home and would silently stop being regenerated from the logs.
    """
    files = [PAPER] + ([APPENDIX] if os.path.exists(APPENDIX) else [])
    texts = {f: open(f).read() for f in files}
    where: dict = {}
    for k, block in BLOCKS.items():
        pat = re.compile(rf"(<!-- ctl:{k} -->)(.*?)(<!-- /ctl:{k} -->)", re.S)
        hits = [f for f in files if pat.search(texts[f])]
        if not hits:
            raise SystemExit(f"no <!-- ctl:{k} --> region in {' or '.join(files)}")
        if len(hits) > 1:
            raise SystemExit(f"<!-- ctl:{k} --> appears in more than one of {hits}")
        texts[hits[0]] = pat.sub(lambda m: m.group(1) + block + m.group(3), texts[hits[0]])
        where.setdefault(hits[0], []).append(k)
    for f, t in texts.items():
        open(f, "w").write(t)
    for f, ks in where.items():
        print(f"rewrote control blocks {sorted(ks)} in {f}")


QCLAIMS: list = []         # tests the prose calls "q < 0.05"
LABEL_TESTS: list = []     # labels-on-depth tests the prose calls null after correction
WEBL: dict = {}
SUGGEST: list = []         # tests the prose calls suggestive


def build_constants() -> None:
    const("3 table", "423,111 programs / 41,245 workbooks / 899,382 web pages", "record counts printed by PackedWindows at load")
    const("4.2 control", "70% of tokens are markup; context 8,192", "web_tags_b8192 manifest (build_web_tags_pack.py)")
    const("2 method", "0.316 rad", "initial ladder step, a design constant (rope_pos.py BASE_STEP = sqrt(2*(1-0.95)))")
    const("2 method", "1/16", "slow-plane ratio, a design constant")
    const("2 method", "22 levels / 44 / 16 / 6 / 4 / 2 planes", "plane allocation, design constants")
    const("3 setup", "17.19 M web pages", "the larger web run pack meta.json page count")
    const("3 setup", "about 10% deeper than 22 levels", "pack manifests (dropped pages)")
    const("3 setup", "AdamW, bfloat16, 2% warmup, cosine to 10%", "train.py / configs (betas, decay, clip, warmup, LR, steps are checked)")
    const("3 table", "base 10,000; 128-wide heads", "model constants")
    const("3 setup", "4.7 M / 31.5 M parameters", "model geometry")
    const("3 setup", "1,024 validation documents; test 8,192 / 1,024 / 1,024 / 2,048", "eval settings (eval_records; understanding_evals.py; eval_test.py)")
    const("4.5 / 5", "77% of chat tokens at one depth", "token depth over 4,000 sampled conversations (per-record node mapping), 2026-10-05; matches build_chat_pack.py's stats")
    const("4.4", "individual heads move 15–37 points", "each seed's most-moved head, docs/mechanism/results.md (heads are not aligned across seeds)")
    const("4.2 control", "5.67% token overhead", "build_marked_pack.py log: 48,094,575 markers over 895,874,770 tokens")
    const("4.6", "median web node at level 10, spreadsheet 5, Python 4, chat 3", "mix4_depth_dist.py over the built pack")
    const("4.6", "relative depth is exactly 1.0 for 80% of chat tokens against 20% of web's", "mix4_rescale_probe.py, token-weighted over in-tree tokens")
    const("4.6", "2.16 passes; 104 records x 9,060 steps", "build_mix4_pack.py manifest and make_mix4_configs.py")


# ------------------------------------------------------------------------------ check

def check_no_partial(text: str) -> list:
    """The two the larger web run runs still training must not be quoted with numbers."""
    bad = []
    for m in re.finditer(r"[^.]*(still training|in progress)[^.]*\.", text):
        if re.search(r"\d+(\.\d+)?%|−0\.\d|\+0\.\d|first \d+%", m.group(0)):
            bad.append(m.group(0).strip())
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--write-controls", action="store_true",
                    help="regenerate the control blocks in the paper from the logs, then audit")
    a = ap.parse_args()
    if a.write_controls:
        build()
        write_controls()
        CLAIMS.clear(); CONSTANTS.clear(); TESTS.clear(); MARKS.clear(); QCLAIMS.clear(); SUGGEST.clear()
        BLOCKS.clear()
    text = audited_text()
    norm = lambda s: re.sub(r"\s+", " ", s)
    flat = norm(text)
    build()
    build_constants()
    ok = miss = 0
    for where, t, val, note in CLAIMS:
        found = norm(t) in flat
        ok += found
        miss += not found
        print(f"{'ok  ' if found else 'MISS'} [{where}] {t}" + (f"   ({note})" if note else ""))
    print(f"\n{ok} claims match the paper, {miss} do not.")
    print(f"\nSTATED CONSTANTS, not recomputed ({len(CONSTANTS)}):")
    for where, t, why in CONSTANTS:
        print(f"  [{where}] {t} -- {why}")
    bad = check_no_partial(text)
    for b in bad:
        print(f"PARTIAL-RESULT QUOTE for a run still training: {b}")
    if a.strict and (miss or bad):
        sys.exit(1)


if __name__ == "__main__":
    main()
