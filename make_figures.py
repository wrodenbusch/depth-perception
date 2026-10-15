#!/usr/bin/env python
"""The paper's figures, regenerated from the run logs. Nothing here is a typed-in number.

    .venv/bin/python make_figures.py            # writes docs/figures/fig*.{svg,png}

The mechanism schematics (fig_rotation, fig_planes, fig_path_vs_depth, fig_reference) are drawings, not data;
everything else is read from the logs.

WHY FROM THE LOGS. Every figure is a picture of a comparison the paper also states as a number, so
a figure drawn from a hand-copied table would be a second, unchecked copy of that number. Each value
plotted here is read from ``train.jsonl`` (validation milestones) or ``heldout.jsonl`` (the test
split) by the same reader ``paired.py`` uses, so a figure and the tool that produced the paper's
table cannot disagree.

RAGGED RUNS ARE REFUSED, as in ``paired.py`` and for the same reason: a run still training sits in
the middle of its schedule, and pairing it with a finished one reports how far along each run is,
on a scale far larger than the effects. ``pairs()`` exits if the runs of a contrast are not all at
one final step. The main web cell's curves pair each arm with sequence-only *at the same step*, so
an arm still running (layer-split) simply ends early rather than being compared off-step.

DESIGN, per the dataviz method. Light mode only, static: these are print figures for a markdown
paper, which is the documented exception to the hover layer -- every plotted value is also in the
paper's tables, which serve as the table view. Colours are the reference palette's first three
categorical slots, validated all-pairs (worst CVD dE 9.2, normal-vision 24.0); aqua sits below 3:1
contrast on the surface, so every series also carries a distinct marker and a direct label. Text is
in ink tokens, never series colour. One axis per panel; zero is drawn as the baseline that matters.
The paper figures and the per-token and learning-curve figures are in % perplexity reduction, where
POSITIVE means structure better; the older per-seed strip figures are in nats/token (negative = better).
"""
from __future__ import annotations
from topos_paths import RUNS  # noqa: E402

import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines  # noqa: E402,F401
import matplotlib.patches  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401

from paired import p_two_sided, read_arm  # noqa: E402

RUNS = f"{RUNS}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "figures")

# reference palette (light) -- see the dataviz skill's palette.md
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "svg.fonttype": "none",
})


# ---------------------------------------------------------------- data

def pairs(root: str, a: str, b: str) -> list[tuple[int, float]]:
    """(seed, a - b) at each run's final milestone, for seeds both arms have. Refuses ragged runs."""
    A = read_arm(os.path.join(RUNS, root), a, "val_nats_tok")
    B = read_arm(os.path.join(RUNS, root), b, "val_nats_tok")
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit(f"{root}: no shared seeds for {a} / {b}")
    last = {A[s][3] for s in shared} | {B[s][3] for s in shared}
    if len(last) > 1:
        raise SystemExit(f"REFUSED {root} {a} vs {b}: runs stopped at different steps {sorted(last)}")
    return [(s, A[s][0] - B[s][0]) for s in shared]


def pairs_finished(root: str, a: str, b: str, step: int) -> list[tuple[int, float]]:
    """Like ``pairs`` but for a setting whose seeds are still landing (chat): only seeds where BOTH
    runs reached ``step`` -- the paired.py --require-step rule -- so a running seed is never paired."""
    A = read_arm(os.path.join(RUNS, root), a, "val_nats_tok")
    B = read_arm(os.path.join(RUNS, root), b, "val_nats_tok")
    shared = sorted(s for s in set(A) & set(B) if A[s][3] == step and B[s][3] == step)
    return [(s, A[s][0] - B[s][0]) for s in shared]


def t_crit(df: int) -> float:
    """Two-sided 95% critical value, by bisection on paired.py's verified t distribution."""
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_two_sided(mid, df) > 0.05 else (lo, mid)
    return (lo + hi) / 2


def summary(d: list[float]):
    """mean, 95% CI half-width (None at n=1), p (None at n=1), seeds where structure is better."""
    n, m = len(d), st.mean(d)
    won = sum(1 for x in d if x < 0)
    if n < 2:
        return m, None, None, won
    se = st.stdev(d) / math.sqrt(n)
    return m, t_crit(n - 1) * se, p_two_sided(m / se, n - 1), won


def web7_curve(arm: str) -> dict[int, float]:
    """step -> val nats/token for a main-web-cell run. Later lines win (a resumed run repeats steps)."""
    out = {}
    for line in open(os.path.join(RUNS, "web7", arm, "train.jsonl")):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = r.get("val_nats_tok")
        if v is not None and math.isfinite(v) and r.get("step", 0) > 0:
            out[r["step"]] = v
    return out


def web7_test(arm: str):
    """(step, test nats/token) from heldout.jsonl, or None if the run has not been scored on test."""
    p = os.path.join(RUNS, "web7", arm, "heldout.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(x) for x in open(p) if x.strip()]
    rows = [r for r in rows if r.get("split") == "test"]
    return (rows[-1]["step"], rows[-1]["nats_tok"]) if rows else None


# ---------------------------------------------------------------- drawing helpers

def save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("svg", {}), ("png", {"dpi": 150})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.svg / .png")


def zero_line(ax, vertical: bool = True) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color=INK2, linewidth=1.0, zorder=1)


def strip_row(ax, y: float, diffs: list[float], color: str, marker: str = "o") -> None:
    """Seed dots jittered deterministically around row y, then the mean and its 95% CI in ink."""
    n = len(diffs)
    for i, x in enumerate(diffs):
        jit = ((i % 5) - 2) * 0.045 if n > 1 else 0.0
        ax.plot(x, y + jit, marker, ms=5.5, color=color, alpha=0.85, mec=SURFACE, mew=1.2, zorder=3)
    m, ci, _, _ = summary(diffs)
    if ci is not None:
        ax.plot([m - ci, m + ci], [y - 0.28, y - 0.28], color=INK, linewidth=1.6, zorder=4)
    ax.plot(m, y - 0.28, "|", ms=9, mew=2, color=INK, zorder=5)


def stat_text(diffs: list[float], who: str = "") -> str:
    """Mean, seeds won and p. ``who`` names the side "won" refers to, so a count is never ambiguous."""
    m, _, p, won = summary(diffs)
    n = len(diffs)
    if p is None:
        return f"{m:+.4f}   1 seed"
    ps = "p<0.0001" if p < 1e-4 else f"p={p:.2g}" if p < 0.01 else f"p={p:.2f}"
    return f"{m:+.4f}   {who + ' ' if who else ''}{won}/{n} seeds   {ps}"


# ---------------------------------------------------------------- figures

def ppl_pct(diff: float) -> float:
    """Reduction in perplexity, %, for a paired nats/token difference. Positive = structure helps."""
    return 100 * (1 - math.exp(diff))


def fig1_headlines() -> None:
    """The main claim in a unit a cold reader knows: % reduction in perplexity, per corpus.

    Perplexity rather than nats/token because the corpora sit at very different baseline losses
    (code ~0.35, prose ~3.3 nats), so the same raw difference means very different things on each;
    as a ratio of perplexities the effect is comparable across rows. The CI is computed on the
    nats/token differences (where the paired t applies) and mapped through the same monotone
    transform, so it is exact, not approximated.
    """
    t_ = web7_test("rope_both_p1"), web7_test("rope_seq_p1")
    s7 = web7_curve("rope_both_p1"), web7_curve("rope_seq_p1")
    last = max(set(s7[0]) & set(s7[1]))
    web_main = ([t_[0][1] - t_[1][1]] if all(t_) and t_[0][0] == t_[1][0]
                else [s7[0][last] - s7[1][last]])
    # The headline configuration is sequence + tree + depth. Web-L has no run of it (one seed costs
    # ~180 GPU-hours), so its row is sequence + tree and says so in its label.
    rows = [("Python code", [d for _, d in pairs("py", "rel", "rope_seq")]),
            ("Spreadsheets", [d for _, d in pairs("sheets", "rel_v2", "seq_v2")]),
            ("Web pages", [d for _, d in pairs("webc", "rel", "seq")])]
    # Web-L is a DIFFERENT configuration (sequence + tree; no sequence + tree + depth run there) on a
    # DIFFERENT split (held-out test, one run), so it gets its own panel rather than a row the
    # title's claim would wrongly cover (referee, 2026-10-03).
    webl = ("Larger web run, 1 seed,\nheld-out test split", web_main)
    allpts = [ppl_pct(x) for _, d in rows + [webl] for x in d]
    hi, lo = max(allpts), min(0.0, min(allpts))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1.05], hspace=0.62))

    def draw(axx, rws, col):
        for y, (label, d) in enumerate(rws):
            n = len(d)
            pts = [ppl_pct(x) for x in d]
            m, ci, _, won = summary(d)
            mp = ppl_pct(m)
            axx.plot([0, mp], [y, y], color=col, linewidth=7, solid_capstyle="butt", alpha=0.35, zorder=2)
            for i, x in enumerate(pts):
                jit = ((i % 5) - 2) * 0.07 if n > 1 else 0.0
                axx.plot(x, y + jit, "o", ms=4.5, color=col, alpha=0.6, mec=SURFACE, mew=0.8, zorder=3)
            if ci is not None:
                axx.plot([ppl_pct(m + ci), ppl_pct(m - ci)], [y, y], color=INK, linewidth=1.5, zorder=4)
            axx.plot(mp, y, "o", ms=8, color=INK, mec=SURFACE, mew=1.2, zorder=5)
            lab = f"{mp:.2f}%  " + (f"{won}/{n} seeds" if n > 1 else "1 run")
            axx.text(1.02, y, lab, va="center", ha="left", color=INK2, fontsize=10,
                     transform=axx.get_yaxis_transform())
        axx.axvline(0, color=INK2, linewidth=1.0, zorder=1)
        axx.set_yticks(range(len(rws)))
        axx.set_yticklabels([r[0] for r in rws], fontsize=10)
        axx.tick_params(axis="y", length=0)
        axx.set_ylim(len(rws) - 0.5, -0.6)
        axx.spines["left"].set_visible(False)
        axx.grid(axis="x")
        axx.set_axisbelow(True)

    draw(ax, rows, V7)
    draw(ax2, [webl], S1)
    ax2.set_xlim(lo - 0.15, hi * 1.06)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax2.tick_params(axis="x", labelsize=10)
    ax2.set_xlabel("perplexity reduction against sequence only (right = better)", fontsize=10)
    ax.set_title("Sequence + tree + depth, in the three settings where it was run",
                 loc="left", color=INK, fontsize=11.5, pad=18)
    ax.annotate("dark dot = mean, line = 95% CI, faint dots = seeds (validation split)",
                xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=9)
    ax2.set_title("Separately: sequence + tree on the larger web run (a different configuration and split)",
                  loc="left", color=INK, fontsize=10.5, pad=6)
    for label, d in rows + [webl]:
        print(f"  fig1 {label!r}: {ppl_pct(st.mean(d)):.3f}%  n={len(d)}")
    save(fig, "fig1_headlines")


def fig_configs() -> None:
    """Every structured configuration against sequence only, per setting, in % perplexity.

    The paper's argument for sequence + tree + depth is that each simpler configuration fails
    somewhere: the path alone loses to depth on web text, depth alone barely helps on spreadsheets.
    That is a comparison ACROSS configurations within a setting, so the three sit side by side on
    one row per setting, colour = configuration in every row. Perplexity % so rows are comparable.
    """
    sets = [("Python code", "py", {"both": "rope_both", "depth": "depth", "rel": "rel", "drank": "drank"},
             "rope_seq"),
            ("Spreadsheets", "sheets", {"both": "both_v2", "depth": "depth_v2", "rel": "rel_v2",
                                        "grid": "grid_v2"}, "seq_v2"),
            ("Web pages", "webc", {"both": "both", "depth": "depth", "rel": "rel",
                                                        "drank": "drank"}, "seq"),
            ("Chat (interim)", "chat", {"both": "both", "depth": "depth", "rel": "rel", "drank": "drank",
                                        "grid": "grid"}, "seq")]
    # "depth + rank" (depth and the node's sibling rank, code and web) is NOT the spreadsheet grid
    # (row, column) -- the referee read the shared orange triangle as one configuration. It is drawn
    # as depth's hue with a hollow marker: it adds a rank to depth and measures like depth. No fifth
    # hue: one next to orange fails the all-pairs normal-vision floor (validated 2026-10-02).
    order = [("both", "sequence + tree", S1, "o", True), ("depth", "sequence + depth", S3, "s", True),
             ("drank", "sequence + depth + rank", S3, "v", False),
             ("rel", "sequence + tree + depth", V7, "D", True),
             ("grid", "sequence + grid (spreadsheets: row, column; chat: exchange, speaker, position)", S2, "^", True)]
    fig, ax = plt.subplots(figsize=(7.4, 5.9))
    hi, lo = 0.0, 0.0
    for yi, (label, root, arms, seq) in enumerate(sets):
        present = [o for o in order if o[0] in arms]
        for k, (key, name, col, mk, filled) in enumerate(present):
            d = [v for _, v in (pairs_finished(root, arms[key], seq, FINAL[root]) if root == "chat"
                                else pairs(root, arms[key], seq))]
            if not d:
                continue
            y = yi + (k - (len(present) - 1) / 2) * 0.2
            m, ci, _, won = summary(d)
            pts = [ppl_pct(x) for x in d]
            hi, lo = max(hi, max(pts)), min(lo, min(pts))
            for i, x in enumerate(pts):
                ax.plot(x, y + ((i % 5) - 2) * 0.018, mk, ms=3.6, color=col, alpha=0.45,
                        mec="none" if filled else col, mfc=col if filled else "none", zorder=3)
            if ci is not None and len(d) >= 5:       # no CI below 5 seeds (chat, interim): t on 1 df is meaningless
                ax.plot([ppl_pct(m + ci), ppl_pct(m - ci)], [y, y], color=INK, linewidth=1.3, zorder=4)
            ax.plot(ppl_pct(m), y, mk, ms=8.5, color=col, mfc=col if filled else SURFACE,
                    mec=INK if filled else col, mew=0.9 if filled else 1.8, zorder=5)
            ax.text(1.02, y, f"{ppl_pct(m):.2f}%  {won}/{len(d)}", va="center", ha="left",
                    color=INK2, fontsize=9.5, transform=ax.get_yaxis_transform())
            print(f"  configs {root:6s} {key:5s} {ppl_pct(m):.3f}%  {won}/{len(d)}")
    ax.axvline(0, color=INK2, linewidth=1.0, zorder=1)
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([s_[0] for s_ in sets], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(len(sets) - 0.38, -0.62)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(lo - 0.1, hi * 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("perplexity reduction against sequence only (right = better)", fontsize=10)
    handles = [matplotlib.lines.Line2D([], [], color=c, marker=mk, linestyle="none", ms=8,
                                       mfc=c if f else SURFACE, mec=INK if f else c, mew=0.9 if f else 1.8)
               for _, _, c, mk, f in order]
    ax.legend(handles, [n for _, n, _, _, _ in order], loc="upper left", bbox_to_anchor=(-0.02, -0.13),
              ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2)
    ax.set_title("Which coordinate pays depends on the corpus",
                 loc="left", color=INK, fontsize=11.5, pad=18)
    ax.annotate("large marker = mean, line = 95% CI (5+ seeds), small markers = seeds; right: mean, seeds better",
                xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=9)
    save(fig, "fig_configs")


def fig2_web_curves() -> None:
    """Main web cell: each arm's gap to sequence-only at every milestone both have reached."""
    seq = web7_curve("rope_seq_p1")
    arms = [("rope_both_p1", "sequence + tree", S1, "o"),
            ("rope_layer_split_p1", "layer-split (5 seq / 5 tree)", S3, "s"),
            ("rope_leaf_p1", "tree + token (sequence replaced)", S2, "^")]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    zero_line(ax, vertical=False)
    ax.text(4, 0.0006, "sequence only", color=INK2, fontsize=8.5, va="bottom")
    ends = []
    for arm, label, c, mk in arms:
        cur = web7_curve(arm)
        steps = sorted(set(cur) & set(seq))
        g = [cur[s] - seq[s] for s in steps]
        x = [s / 1000 for s in steps]
        ax.plot(x, g, "-", color=c, linewidth=2, zorder=3)
        ax.plot(x, g, mk, color=c, ms=4.5, mec=SURFACE, mew=1.0, zorder=4)
        ends.append((x[-1], g[-1], label, steps[-1]))
        tt, ts = web7_test(arm), web7_test("rope_seq_p1")
        if tt and ts and tt[0] == ts[0]:
            ax.plot(tt[0] / 1000 + 12, tt[1] - ts[1], "D", ms=6, mfc=SURFACE, mec=c, mew=1.8, zorder=5)
    for x, gv, label, step in ends:
        note = "" if step == max(seq) else f"  (still running, step {step:,})"
        ax.text(x + 22, gv, label + note, color=INK2, fontsize=8.5, va="center")
    ax.plot([], [], "D", ms=6, mfc=SURFACE, mec=INK2, mew=1.5, label="held-out test split, final")
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(seq) / 1000 * 1.38)
    ax.set_xticks([t for t in range(0, max(seq) // 1000 + 1, 100)])
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("gap to sequence-only, val nats/token\n(negative = better)")
    ax.set_title("Main web cell: gap to sequence-only at every milestone (one seed per arm)",
                 loc="left", color=INK, fontsize=11)
    save(fig, "fig2_web_curves")


def fig3_depth() -> None:
    """Depth against sequence-only and against the full path, per seed, one panel per corpus."""
    panels = [("Python AST", "py", "depth", "rope_seq", "rope_both"),
              ("Spreadsheets", "sheets", "depth_v2", "seq_v2", "both_v2"),
              ("Web pages", "webc", "depth", "seq", "both")]
    data = [(t, pairs(r, d, s), pairs(r, d, p)) for t, r, d, s, p in panels]
    allv = [v for _, a, b in data for _, v in a + b]
    lo, hi = min(allv), max(allv)
    span = hi - lo
    fig, axes = plt.subplots(len(data), 1, figsize=(8.2, 5.0), sharex=True)
    for ax, (title, vs_seq, vs_path) in zip(axes, data):
        for y, d, c, mk in ((0, [v for _, v in vs_seq], S1, "o"), (1, [v for _, v in vs_path], S2, "s")):
            strip_row(ax, y, d, c, mk)
            ax.text(hi + 0.06 * span, y, stat_text(d, "depth better on"), va="center",
                    color=INK2, fontsize=8.5)
        zero_line(ax)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["depth − sequence only", "depth − full tree path"])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(1.6, -0.7)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5, pad=2)
    axes[-1].set_xlim(lo - 0.05 * span, hi + 0.55 * span)
    axes[-1].set_xlabel("paired difference in validation nats/token (negative = depth better)")
    fig.suptitle("Two planes of depth, against sequence-only and against the 44-plane path",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_depth")


def fig4_sheets() -> None:
    """Spreadsheet grid position vs sequence, per seed, on both renderings."""
    series = [("cell values only", pairs("sheets", "both_v2", "seq_v2"), S1, "o"),
              ("addresses spelled out", pairs("sheets", "both_vanilla_v2", "vanilla_seq_v2"), S2, "s")]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    zero_line(ax, vertical=False)
    for k, (label, pr, c, mk) in enumerate(series):
        off = -0.12 if k == 0 else 0.12
        xs = [s + off for s, _ in pr]
        ys = [v for _, v in pr]
        ax.plot(xs, ys, mk, ms=7, color=c, mec=SURFACE, mew=1.5, zorder=3, label=label)
        m = st.mean(ys)
        ax.plot([-0.5, max(s for s, _ in pr) + 0.5], [m, m], "-", color=c, linewidth=1.2, alpha=0.6, zorder=2)
        ax.text(max(s for s, _ in pr) + 0.7, m, f"{label}\n{stat_text(ys, 'grid better on')}", color=INK2,
                fontsize=8.5, va="center")
    ax.set_xticks(range(0, 10))
    ax.set_xlim(-0.7, 13.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("grid + sequence − sequence\nval nats/token (negative = better)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("Spreadsheets: grid position beats sequence position on every seed",
                 loc="left", color=INK, fontsize=11)
    save(fig, "fig3_sheets")


# ---------------------------------------------------------------- loss curves

V7 = "#4a3aa7"                     # palette slot 7 (violet); with S1-S3 validated all-pairs, see below
FINAL = {"py": 11200, "sheets": 8240, "webc": 9375, "web7": 537000, "chat": 13100}

# Colour follows the CONFIGURATION, identically in every panel. Sequence only is the baseline every
# gap is measured against, so it is drawn in neutral ink rather than taking a categorical slot; no
# position is muted and dashed for the same reason (it is a floor, not a contender). The four
# categorical series -- blue, orange, aqua, violet -- validate ALL-PAIRS in light mode (worst CVD dE
# 9.2, normal-vision 16.3); aqua sits below 3:1 on the surface, so every series also carries its own
# marker and the legend names it.
CONFIGS = {
    "seq":   ("sequence only", INK2, "o", "-"),
    "both":  ("sequence + tree", S1, "o", "-"),
    "depth": ("sequence + depth", S3, "s", "-"),
    "rel":   ("sequence + tree + depth", V7, "D", "-"),
    "grid":  ("sequence + grid", S2, "^", "-"),
    # depth + the node's sibling rank (code, web): depth's hue, its own marker and dash -- it is
    # not the spreadsheet grid, and measures like depth
    "drank": ("sequence + depth + rank", S3, "v", (0, (4, 2))),
    # tree only is a side point (replacing sequence position), so it is drawn neutral, like the
    # full-RoPE control: orange belongs to "grid", and a fifth hue fails the all-pairs floors
    "leaf":  ("tree only (no sequence index)", MUTED, "^", (0, (1, 1.5))),
    "none":  ("no position", MUTED, "v", "--"),
}
# setting -> (title, {config: arm directory prefix}). Web-L's no-position and the other tree-only
# variant diverged and are absent by construction; Web-L depth appears once it has a milestone.
CURVE_SETTINGS = [
    ("py", "Python (5 seeds; tree only 20)", {"seq": "rope_seq", "both": "rope_both", "depth": "depth", "rel": "rel",
                      "drank": "drank", "leaf": "rope_leaf", "none": "none"}),
    ("sheets", "Spreadsheets (10 seeds)", {"seq": "seq_v2", "both": "both_v2", "depth": "depth_v2",
                                "rel": "rel_v2", "grid": "grid_v2"}),
    ("webc", "Web pages (10 seeds)", {"seq": "seq", "both": "both", "depth": "depth", "rel": "rel",
                                               "drank": "drank"}),
    ("web7", "Larger web run (1 seed)", {"seq": "rope_seq", "both": "rope_both",
                                          "depth": "rope_depth", "leaf": "rope_leaf"}),
]


def run_curves(root: str, prefix: str) -> dict[int, dict[int, float]]:
    """seed -> {step: val nats/token} for every run of an arm. Step 0 (untrained) is dropped; a
    resumed run repeats steps, and the later line wins. ``<arm>_p1`` (Web-L) is seed 0."""
    import glob
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, root, f"{prefix}_[sp][0-9]*"))):
        base, tail = os.path.basename(d).rsplit("_", 1)
        if base != prefix or not tail[1:].isdigit():
            continue
        seed = 0 if tail.startswith("p") else int(tail[1:])
        log = os.path.join(d, "train.jsonl")
        if not os.path.exists(log):
            continue
        cur = {}
        for line in open(log):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("val_nats_tok")
            if v is not None and math.isfinite(v) and r.get("step", 0) > 0:
                cur[r["step"]] = v
        if cur:
            out[seed] = cur
    return out


def _gap_series(root: str, arms: dict, fin: int):
    """Per configuration: (x %, mean gap nats, CI half-width nats, n, live) against sequence only.

    NO RAGGED COMPARISONS. A finished configuration uses only seeds that reached the setting's
    final step, paired with the same seeds of sequence only, at milestones both have. A
    configuration still training is drawn only at milestones its paired sequence-only seeds also
    have, so it ends early and is marked in progress. One with no milestone is left out.
    """
    import functools
    tc = functools.lru_cache(maxsize=None)(t_crit)
    base = {s: c for s, c in run_curves(root, arms["seq"]).items() if max(c) == fin}
    if not base:
        raise SystemExit(f"{root}: no finished sequence-only run")
    out = {}
    for key, prefix in arms.items():
        if key == "seq":
            continue
        runs = run_curves(root, prefix)
        done = {s: c for s, c in runs.items() if max(c) == fin}
        live = not done
        use = {s: c for s, c in (runs if live else done).items() if s in base}
        if not use:
            continue
        steps = sorted(set.intersection(*(set(c) for c in use.values()),
                                        *(set(base[s]) for s in use)))
        if not steps or (live and len(steps) < 2):
            continue
        n = len(use)
        gaps = [[use[s][t] - base[s][t] for s in use] for t in steps]
        gm = [st.mean(g) for g in gaps]
        ci = [tc(n - 1) * st.stdev(g) / math.sqrt(n) if n > 1 else 0.0 for g in gaps]
        out[key] = ([100 * t / fin for t in steps], gm, ci, n, live)
        print(f"  curves {root:6s} {key:9s} n={n} last {steps[-1]} gap {gm[-1]:+.4f}"
              + ("  IN PROGRESS" if live else ""))
    return base, out


def _spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Nudge label y positions apart (keeping their order) so direct labels never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for _ in range(50):
        moved = False
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < min_gap:
                mid = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = mid - min_gap / 2, mid + min_gap / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


#: Curve figures add the full-RoPE control. It is a CONTROL (sequence position on every plane), so
#: like sequence only and no position it is drawn in neutral ink with its own dash pattern rather
#: than taking a categorical slot; the four categorical series keep their validated all-pairs set.
CURVE_STYLE = dict(CONFIGS, full=("full RoPE (control)", INK2, "x", (0, (5, 2))))
CURVE_ARMS = {
    "py": {"full": "full_rope"},
    "sheets": {"full": "full_rope_v2"},
    "webc": {"full": "full_rope"},
    "web7": {},
}
SHORT = {"rel": "seq + tree + depth", "both": "seq + tree", "depth": "seq + depth", "grid": "seq + grid",
         "drank": "seq + depth + rank",
         "leaf": "tree only", "none": "no position", "full": "full RoPE"}


def fig_loss_curves() -> None:
    """Each configuration's gap to sequence only over training, as % perplexity, one LARGE panel
    per setting.

    WHY THIS FORM (2026-10-02, after "the val loss chart is too small to see any difference"). The
    effects are ~1% of perplexity while the loss falls by several nats over a run, so raw-loss
    panels show lines on top of each other at any size. The quantity the paper reports is the
    paired gap, so the paper figure plots only that, in perplexity % -- 100*(1-e^gap), positive =
    better than sequence only -- with a 2x2 grid instead of four thin columns. The raw curves move
    to a companion (``fig_loss_raw``) in the appendix.

    THE BAND IS THE 95% CI OF THE PAIRED MEAN, not the seed range. Each panel has its own y-axis:
    the settings are different corpora and their gaps are not meant to be read against each other.
    The y-range is set by the finished configurations from 25% of training on; earlier
    milestones and far-off configurations (no position on Python) are clipped and say so.
    Direct labels sit at each line's end, nudged apart, instead of a legend.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), gridspec_kw={"hspace": 0.34, "wspace": 0.66})
    order = ["rel", "grid", "both", "depth", "drank", "full", "leaf", "none"]
    # spreadsheets: the first milestones swing far off scale (two seeds diverge early and recover),
    # which the referee read as a crash; that panel starts at 25% and says so
    start = {"sheets": 25}
    for ax, (root, title, arms) in zip(axes.flat, CURVE_SETTINGS):
        fin = FINAL[root]
        arms = dict(arms, **CURVE_ARMS[root])
        _, series = _gap_series(root, arms, fin)
        # A configuration still training is not drawn in the PAPER figure (2026-10-03): the paper
        # quotes only finished runs, and a partial curve is a partial result. This includes the
        # single-seed Web-L depth and tree-grid runs; the appendix shows them, marked in progress.
        series = {k: v for k, v in series.items() if not v[4]}
        ax.axhline(0, color=INK2, linewidth=1.1, zorder=1)
        rng, offscale, ends = [], [], []
        for key in order:
            if key not in series:
                continue
            x, gm, ci, n, live = series[key]
            keep = [i for i, xx in enumerate(x) if xx >= start.get(root, 10)]
            x, gm, ci = [x[i] for i in keep], [gm[i] for i in keep], [ci[i] for i in keep]
            if len(x) < 2:      # a lone point is not a curve (Web-L depth at its first milestones)
                continue
            pct = [ppl_pct(g) for g in gm]
            lo = [ppl_pct(g + c) for g, c in zip(gm, ci)]
            hi = [ppl_pct(g - c) for g, c in zip(gm, ci)]
            label, color, mk, ls = CURVE_STYLE[key]
            head = key == "rel"
            lw = 3.0 if head else 1.7
            alpha = 0.55 if live else 1.0
            if key == "none":
                offscale.append(f"no position: {pct[-1]:+.1f}%, off scale")
                continue
            if n > 1 and not live and head:   # one band only: with five, they bury the lines
                ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
            # Same reason as the compact figure: sequence + depth must draw above the head line,
            # or on web pages (2.00% against 1.98%) it disappears underneath it.
            ax.plot(x, pct, linestyle=ls, color=color, linewidth=lw, alpha=alpha,
                    zorder=4.5 if key == "depth" else (4 if head else 3))
            if not live:
                rng += [v for xx, v in zip(x, pct) if xx >= 25]
            tag = SHORT[key] + (f" (in progress, {x[-1]:.0f}%)" if live else "")
            ends.append((x[-1], pct[-1], tag, color, head))
        g0, g1 = min(rng + [0.0]), max(rng + [0.0])
        pad = 0.18 * (g1 - g0)
        ax.set_ylim(g0 - pad, g1 + pad)
        ylo, yhi = ax.get_ylim()
        ys = _spread_labels([min(max(e[1], ylo), yhi) for e in ends], 0.115 * (yhi - ylo))
        for (xe, ye, tag, color, head), yl in zip(ends, ys):
            ax.plot([xe], [min(max(ye, ylo), yhi)], "o", color=color, ms=5 if head else 4,
                    mec=SURFACE, mew=0.8, zorder=5)
            ax.annotate(tag, (xe, min(max(ye, ylo), yhi)), xytext=(103, yl), textcoords="data",
                        va="center", ha="left", fontsize=8.5, color=INK if head else INK2,
                        fontweight="bold" if head else "normal",
                        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6,
                                        shrinkA=0, shrinkB=2), annotation_clip=False)
        ax.text(9, 0, "sequence only", color=INK2, fontsize=8, va="bottom")
        for k, msg in enumerate(offscale):
            ax.text(0.98, 0.03 + 0.07 * k, msg, transform=ax.transAxes, ha="right", va="bottom",
                    color=INK2, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
        ax.set_title(title + ("  (from 25% of training)" if root in start else ""), loc="left",
                     color=INK, fontsize=11)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlim(8, 101)
        ax.set_xticks([10, 25, 50, 75, 100])
        ax.set_xlabel("% of training")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}%"
                                                                     if v else "0"))
        ax.set_ylabel("perplexity reduction vs sequence only\n(above 0 = better)")
    fig.suptitle("How much each configuration beats sequence only, over the course of training",
                 x=0.06, y=0.985, ha="left", color=INK, fontsize=13)
    fig.text(0.06, 0.945, "line = mean over seeds of the paired gap, as % perplexity; band = its 95% "
             "confidence interval (bold line only). Each panel has its own scale; plotted from 10% of training.",
             color=MUTED, fontsize=9)
    save(fig, "fig_loss_curves")


def fig_loss_curves_compact() -> None:
    """Compact version of ``fig_loss_curves`` for the paper body; the full 2x2 stays in the appendix.

    ("just make them smaller", 2026-10-03.) Drawn at the printed text width (6.7 in) so the type
    prints at its nominal size, >= 7 pt. Only the key configurations, one row of four panels. In
    panels this small, end labels for four or five lines collide, so identity is carried by one
    shared legend above the row plus distinct dashes and markers -- the dataviz rule (legend for
    >= 2 series) rather than direct labels. Same data path, pairing and CI as the full figure.
    """
    keys = ["rel", "grid", "both", "depth", "full"]
    start = {"sheets": 25}
    fig, axes = plt.subplots(1, 4, figsize=(6.7, 1.95), gridspec_kw={"wspace": 0.42})
    for ax, (root, title, arms) in zip(axes, CURVE_SETTINGS):
        fin = FINAL[root]
        arms = {k: v for k, v in dict(arms, **CURVE_ARMS[root]).items() if k == "seq" or k in keys}
        _, series = _gap_series(root, arms, fin)
        series = {k: v for k, v in series.items() if not v[4]}
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
        rng = []
        for key in keys:
            if key not in series:
                continue
            x, gm, ci, n, _ = series[key]
            keep = [i for i, xx in enumerate(x) if xx >= start.get(root, 10)]
            x, gm, ci = [x[i] for i in keep], [gm[i] for i in keep], [ci[i] for i in keep]
            if len(x) < 2:
                continue
            pct = [ppl_pct(g) for g in gm]
            label, color, mk, ls = CURVE_STYLE[key]
            head = key == "rel"
            if head and n > 1:
                ax.fill_between(x, [ppl_pct(g + c) for g, c in zip(gm, ci)],
                                [ppl_pct(g - c) for g, c in zip(gm, ci)],
                                color=color, alpha=0.16, linewidth=0, zorder=2)
            # Draw sequence + depth ABOVE sequence + tree + depth. On web pages the two are
            # 2.00% and 1.98%, so the thick head line painted over the thin depth line and the
            # panel looked as though depth had not been run. Thin over thick leaves the head
            # visible as a halo, which is what two coincident curves should look like.
            zo = 4.5 if key == "depth" else (4 if head else 3)
            ax.plot(x, pct, linestyle=ls, color=color, linewidth=2.0 if head else 1.2, zorder=zo)
            ax.plot([x[-1]], [pct[-1]], mk, color=color, ms=3.5, mec=SURFACE, mew=0.5,
                    zorder=5.5 if key == "depth" else 5)
            rng += [v for xx, v in zip(x, pct) if xx >= 25]
        g0, g1 = min(rng + [0.0]), max(rng + [0.0])
        pad = 0.15 * (g1 - g0)
        ax.set_ylim(g0 - pad, g1 + pad)
        short = title.split(" (")[0].replace(", short pages", "").replace(", long run", "")
        n_lab = title[title.find("(") + 1:title.find(")")].split(";")[0]
        ax.set_title(f"{short} ({n_lab})", loc="left", color=INK, fontsize=7.5, pad=3)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlim(start.get(root, 10) - 2, 101)
        ax.set_xticks([25, 50, 75, 100] if root in start else [10, 50, 100])
        ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}" if v else "0"))
        ax.set_xlabel("% of training", fontsize=7, labelpad=1)
    axes[0].set_ylabel("perplexity reduction, %\n(above 0 = better)", fontsize=7, labelpad=2)
    handles = [matplotlib.lines.Line2D([], [], color=CURVE_STYLE[k][1], linestyle=CURVE_STYLE[k][3],
                                       linewidth=2.0 if k == "rel" else 1.2, marker=CURVE_STYLE[k][2], ms=3.5)
               for k in keys]
    fig.legend(handles, [SHORT[k] for k in keys], loc="lower center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(keys), frameon=False, fontsize=7, labelcolor=INK2, handlelength=2.4,
               columnspacing=1.2, handletextpad=0.4)
    save(fig, "fig_loss_curves_compact")


def fig_understanding_compact() -> None:
    """Compact ``fig_understanding`` for the paper body: the most telling token kinds per setting.

    One row of three panels at the printed width (6.7 in). Mean and 95% CI per configuration; seeds
    are omitted at this size (the full figure in the appendix has them). Same data and units.
    """
    pick = {
        "sheets": (["all", "sh.newline", "sh.ref", "sh.copy_above_far"],
                   ["both_v2", "grid_v2", "depth_v2", "ref_anchor"]),
        "webc": (["all", "web.node_start", "web.rep_cross"], ["both", "depth", "rel"]),
        "py": (["all", "py.ident_reuse", "py.ascend"], ["rope_both", "depth", "rel"]),
    }
    relabel = {"all tokens (the average)": "all tokens", "copy from the cell above, ≥256 tokens back":
               "copy from far above", "cell reference in a formula": "cell reference",
               "first token of a new text block": "block start", "word last seen in another branch":
               "word from\nanother branch", "a subtree has just closed": "after a\nsubtree closes",
               "reusing a variable": "reusing a\nvariable"}
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 1.85), gridspec_kw={"wspace": 0.62,
                                                                     "width_ratios": [1.15, 1, 1]})
    seen = {}
    for ax, (setting, title, cats, arms) in zip(axes, UND_PANELS):
        keep_c, keep_a = pick[setting]
        cats = [c for c in cats if c[0] in keep_c]
        arms = [a for a in arms if a[0] in keep_a]
        R = {(r["arm"], r["cat"]): r for r in json.load(open(os.path.join(UND, f"{setting}_results.json")))}
        lo, hi = 0.0, 0.0
        for yi, (cat, lab) in enumerate(cats):
            for k, (arm, name, col, mk) in enumerate(arms):
                r = R.get((arm, cat))
                if r is None:
                    continue
                m, ci, _, _ = summary(r["diffs"])
                y = yi + (k - (len(arms) - 1) / 2) * 0.2
                if ci is not None:
                    a, b = ppl_pct(m + ci), ppl_pct(m - ci)
                    ax.plot([a, b], [y, y], color=col, linewidth=1.0, zorder=3)
                    lo, hi = min(lo, a, b), max(hi, a, b)
                ax.plot(ppl_pct(m), y, mk, ms=4, color=col, mec=SURFACE, mew=0.5, zorder=4)
                seen[name] = (col, mk)
        ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
        ax.axhspan(-0.5, 0.5, color=GRID, alpha=0.45, zorder=0)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([relabel.get(c[1], c[1]) for c in cats], fontsize=6.8, linespacing=0.95)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1.5)
        ax.set_ylim(len(cats) - 0.5, -0.6)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        span = hi - lo
        ax.set_xlim(lo - 0.06 * span, hi + 0.06 * span)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_title(title.replace(", short pages", ""), loc="left", color=INK, fontsize=7.5, pad=3)
    axes[1].set_xlabel("perplexity reduction on that kind of token vs sequence only (right = better)",
                       fontsize=7, labelpad=2)
    handles = [matplotlib.lines.Line2D([], [], color=c, marker=m, linestyle="none", ms=4.5)
               for c, m in seen.values()]
    fig.legend(handles, list(seen), loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=len(seen),
               frameon=False, fontsize=7, labelcolor=INK2, columnspacing=1.0, handletextpad=0.2)
    save(fig, "fig_understanding_compact")


def fig_loss_raw(log: bool = False) -> None:
    """Raw validation loss, for readers who want the curves themselves rather than the gap.

    Two renderings were made and compared by eye (2026-10-02):
      * ``log=False``: the LAST HALF of training on linear axes with a tight y-range;
      * ``log=True``: the whole run on log-log axes (log step, log loss). On a log y-axis equal
        vertical gaps are equal RATIOS, so a ~1% perplexity gap is the same height early and late,
        and a log x-axis gives the late phase -- where the runs separate -- most of the width.
    VERDICT: the late-window linear version. On log-log the curves still lie on top of each other
    -- the loss spans a factor of 2-3 over the run, so a 1% gap is a sliver at any height -- while
    the late window on a tight linear axis separates every configuration by eye. It earned a paper
    slot; the log-log rendering is kept callable but is not produced by ``main``.
    Three configurations only -- sequence only, full RoPE, and the best structured configuration
    (sequence + tree + depth; sequence + tree on Web-L, which has no path + depth run) -- because
    on a raw axis more lines than that are unreadable at a ~0.01 separation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), gridspec_kw={"hspace": 0.38, "wspace": 0.72})
    for ax, (root, title, arms) in zip(axes.flat, CURVE_SETTINGS):
        fin = FINAL[root]
        arms = dict(arms, **CURVE_ARMS[root])
        best = "rel" if "rel" in arms else "both"
        base = {s: c for s, c in run_curves(root, arms["seq"]).items() if max(c) == fin}
        vals, ends = [], []
        for key in ("seq", "full", best):
            if key not in arms:
                continue
            runs = {s: c for s, c in run_curves(root, arms[key]).items() if max(c) == fin}
            use = base if key == "seq" else {s: c for s, c in runs.items() if s in base}
            if not use:
                continue
            steps = sorted(set.intersection(*(set(c) for c in use.values()),
                                            *(set(base[s]) for s in use)))
            steps = [t for t in steps if (t >= 0.05 * fin if log else t >= fin / 2)]
            if not steps:
                continue
            y = [st.mean(use[s][t] for s in use) for t in steps]
            x = steps if log else [100 * t / fin for t in steps]
            vals += [v for t, v in zip(steps, y) if t >= fin / 2] if not log else y
            if key == "seq":
                color, ls, lw, name = INK2, "-", 1.7, "sequence only"
            else:
                _, color, _, ls = CURVE_STYLE[key]
                lw, name = (2.8, SHORT[key]) if key == best else (1.7, SHORT[key])
            ax.plot(x, y, linestyle=ls, color=color, linewidth=lw, zorder=3)
            ends.append((x[-1], y[-1], name + (f" (n={len(use)})" if key != "seq" else ""),
                         INK if key == best else INK2, key == best))
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.set_xlabel("training step (log scale)")
        else:
            lo, hi = min(vals), max(vals)
            ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
            ax.set_xlim(50, 100)
            ax.set_xticks([50, 75, 100])
            ax.set_xlabel("% of training")
        ylo, yhi = ax.get_ylim()
        span = (math.log(yhi) - math.log(ylo)) if log else (yhi - ylo)
        raw = [math.log(e[1]) if log else e[1] for e in ends]
        ys = _spread_labels(raw, 0.08 * span)
        x_lab = None
        for (xe, ye, tag, color, head), yl in zip(ends, ys):
            x_lab = xe * 1.06 if log else xe + 1.5          # just right of the line ends
            ax.annotate(tag, (xe, ye), xytext=(x_lab, math.exp(yl) if log else yl),
                        textcoords="data", va="center", ha="left", fontsize=8.5, color=color,
                        fontweight="bold" if head else "normal", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6,
                                        shrinkA=0, shrinkB=1))
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="y", which="both" if log else "major")
        ax.set_axisbelow(True)
        ax.set_ylabel("validation loss (nats/token" + (", log scale)" if log else ")"))
    what = ("whole run, log-log axes: equal vertical gaps are equal % perplexity" if log
            else "second half of training, tight linear y-axis")
    fig.suptitle(f"Raw validation loss, mean over seeds ({what})", x=0.06, y=0.99, ha="left",
                 color=INK, fontsize=12)
    save(fig, "fig_loss_raw_log" if log else "fig_loss_raw")


# ---------------------------------------------------------------- mechanism schematics

PALE = "#f0efec"
COORD = {  # what rotates a plane -> (colour, legend label)
    "seq": (INK2, "sequence index"),
    "tree": (S1, "tree path (2 planes per level)"),
    "depth": (S3, "depth"),
    "node": (S2, "within-node index"),
    "off": (PALE, "not rotated"),
}


def fig_planes() -> None:
    """Which of a head's 64 rotary planes each configuration rotates, and by what."""
    def row(*spans):
        out = []
        for kind, n in spans:
            out += [kind] * n
        assert len(out) == 64, len(out)
        return out
    rows = [
        ("standard RoPE", row(("seq", 64)), "64 rotate"),
        ("sequence only (our baseline)", row(("seq", 16), ("off", 48)), "16"),
        ("sequence + tree", row(("seq", 16), ("tree", 44), ("off", 4)), "60"),
        ("sequence + depth", row(("seq", 16), ("off", 44), ("depth", 2), ("off", 2)), "18"),
        ("sequence + tree + depth", row(("seq", 16), ("tree", 44), ("depth", 2), ("off", 2)), "62"),
        ("sequence + grid (spreadsheets)", row(("seq", 16), ("tree", 6), ("off", 42)), "22"),
        ("sequence + depth + rank (code, web)", row(("seq", 16), ("depth", 2), ("tree", 2), ("off", 44)), "20"),
        ("tree only", row(("off", 16), ("tree", 44), ("node", 4)), "48"),
        ("reference rotation", row(("seq", 16), ("ref", 44), ("off", 4)), "60"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    w, h, gap = 1.0, 0.72, 0.12
    for y, (name, kinds, note) in enumerate(rows):
        for i, k in enumerate(kinds):
            x = i * w + (0.25 if i >= 16 else 0) + (0.25 if i >= 60 else 0)
            # reference rotation rotates the SAME planes as the tree path, by different values
            # (the referent's coordinates), so it keeps the tree hue and adds a hatch as the
            # secondary encoding rather than spending a new, unvalidated hue
            base = "tree" if k == "ref" else k
            ax.add_patch(matplotlib.patches.Rectangle(
                (x + gap / 2, y - h / 2), w - gap, h, facecolor=COORD[base][0],
                edgecolor=AXIS if k == "off" else (SURFACE if k == "ref" else "none"),
                linewidth=0.5 if k == "off" else 0.0, hatch="////" if k == "ref" else None,
                alpha=(0.55 if (base == "tree" and ((i - 16) // 2) % 2) else 1.0)))
        if name.startswith("sequence + depth + rank"):
            ax.text(16.25 + 4.6, y, "← depth, then the node's rank among its siblings",
                    ha="left", va="center", color=INK2, fontsize=11)
        if name.startswith("sequence + grid"):
            ax.text(16.25 + 6.6, y, "← sheet, row, column: the first three tree levels", ha="left",
                    va="center", color=INK2, fontsize=11)
        if "ref" in kinds:
            ax.text(16.25 + 22, y + h / 2 + 0.08, "tree path: referent's coordinates on referring tokens",
                    ha="center", va="top", color=INK2, fontsize=10.5)
        ax.text(-1.0, y, name, ha="right", va="center", color=INK, fontsize=12)
        ax.text(65.5, y, note, ha="left", va="center", color=INK2, fontsize=11.5)
    # group captions over the top row's spans
    for x0, x1, lab in ((0, 16, "planes 1–16"), (16.25, 60.25, "planes 17–60: one pair per tree level, 22 levels"),
                        (60.5, 64.5, "61–64")):
        ax.annotate("", xy=(x0 + 0.1, -0.75), xytext=(x1 - 0.1, -0.75),
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
        ax.text((x0 + x1) / 2, -0.95, lab, ha="center", va="bottom", color=MUTED, fontsize=10.5)
    ax.text(65.5, -0.95, "planes\nrotated", ha="left", va="bottom", color=MUTED, fontsize=10.5)
    handles = [matplotlib.patches.Patch(facecolor=c, edgecolor=AXIS if k == "off" else "none", label=l)
               for k, (c, l) in COORD.items()]
    handles.append(matplotlib.patches.Patch(facecolor=S1, edgecolor=SURFACE, hatch="////",
                                            label="tree path, referent's coordinates"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=3, frameon=False,
              fontsize=11.5, labelcolor=INK2, handlelength=1.2)
    ax.set_xlim(-0.5, 69)
    ax.set_ylim(len(rows) - 0.15, -1.5)
    ax.axis("off")
    fig.suptitle("One attention head's 64 rotary planes, and what rotates each, per configuration",
                 x=0.01, ha="left", color=INK, fontsize=13)
    save(fig, "fig_planes")


def _dial(ax, cx, cy, ang_deg, r=0.36, hand=INK, face=SURFACE, ring=AXIS, lw=2.0):
    ax.add_patch(matplotlib.patches.Circle((cx, cy), r, facecolor=face, edgecolor=ring, linewidth=1.0))
    a = math.radians(90 - ang_deg)
    ax.plot([cx, cx + 0.82 * r * math.cos(a)], [cy, cy + 0.82 * r * math.sin(a)], color=hand,
            linewidth=lw, solid_capstyle="round")
    ax.plot(cx, cy, "o", ms=2.5, color=hand)


def fig_path_vs_depth() -> None:
    """Three tokens on a small page: what the tree path puts in the score, and what sequence + depth
    can only infer -- and only when the tree's ancestors are themselves tokens in the stream.

    Schematic, not data. Angles are exaggerated (one sibling = 55 degrees) so equal and unequal
    hands are visible; the model learns its own steps. A and B share a section and differ only in
    paragraph; C sits in the next section at the same depth. The right-hand panel is sequence +
    depth AS RUN (16 sequence planes + 2 depth planes), not depth alone: it sees order and depth.
    """
    fig = plt.figure(figsize=(12.5, 6.4))
    # --- left: the tree
    axt = fig.add_axes([0.0, 0.33, 0.27, 0.58])
    axt.axis("off")
    axt.set_xlim(0, 10)
    axt.set_ylim(0, 10)
    nodes = {"body": (5, 9.2), "s0": (2.6, 6.5), "s1": (7.4, 6.5),
             "p00": (1.3, 3.8), "p01": (3.9, 3.8), "p10": (6.1, 3.8), "p11": (8.7, 3.8)}
    names = {"body": "<body>", "s0": "<section>\nrank 0", "s1": "<section>\nrank 1",
             "p00": "<p>\nrank 0", "p01": "<p>\nrank 1", "p10": "<p>\nrank 0", "p11": "<p>\nrank 1"}
    for u, v in (("body", "s0"), ("body", "s1"), ("s0", "p00"), ("s0", "p01"), ("s1", "p10"), ("s1", "p11")):
        (x0, y0), (x1, y1) = nodes[u], nodes[v]
        axt.plot([x0, x1], [y0 - 0.55, y1 + 0.6], color=AXIS, linewidth=1.2, zorder=1)
    for k, (x, y) in nodes.items():
        axt.text(x, y, names[k], ha="center", va="center", fontsize=8, color=INK,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=AXIS, linewidth=0.8))
    tokcol = {"A": S1, "B": S1, "C": S2}
    for tok, nk in (("A", "p00"), ("B", "p01"), ("C", "p10")):
        x, y = nodes[nk]
        axt.text(x, y - 1.6, tok, ha="center", va="center", fontsize=11, color=SURFACE, weight="bold",
                 bbox=dict(boxstyle="circle,pad=0.3", facecolor=tokcol[tok], edgecolor="none"))
    axt.text(0.2, 0.3, "A and B: same section, different paragraph\nC: next section, same depth (3)",
             fontsize=8, color=INK2, va="bottom")

    # --- (i) what each configuration puts into one attention score
    axd = fig.add_axes([0.29, 0.33, 0.70, 0.58])
    axd.axis("off")
    axd.set_xlim(-1.3, 9.6)
    axd.set_ylim(-1.0, 4.1)
    axd.set_aspect("equal", adjustable="datalim")
    step, theta = 55, (0, 20)
    ranks = {"A": (0, 0), "B": (0, 1), "C": (1, 0)}
    pos = {"A": 2, "B": 4, "C": 7}          # positions in the tagged stream below
    rowy = {"A": 2.6, "B": 1.5, "C": 0.4}
    # columns: [path] seq, level1, level2 | [seq+depth] seq, depth
    cx = {"pseq": 0.3, "l1": 1.6, "l2": 3.05, "dseq": 6.1, "dep": 7.35}

    def seqbox(x, y, tok):
        axd.text(x, y, f"position\n{pos[tok]}", ha="center", va="center", fontsize=8, color=INK,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor=AXIS, linewidth=0.8))

    for tok, y in rowy.items():
        axd.text(-0.85, y, tok, ha="center", va="center", fontsize=11, color=SURFACE, weight="bold",
                 bbox=dict(boxstyle="circle,pad=0.3", facecolor=tokcol[tok], edgecolor="none"))
        seqbox(cx["pseq"], y, tok)
        seqbox(cx["dseq"], y, tok)
        for lv, key in ((0, "l1"), (1, "l2")):
            same = ranks[tok][lv] == ranks["A"][lv]
            _dial(axd, cx[key], y, theta[lv] + ranks[tok][lv] * step,
                  ring=AXIS if same else INK, face=SURFACE if same else PALE)
            if tok != "A":
                axd.text(cx[key] + 0.42, y - 0.32, "= A" if same else "≠ A", fontsize=8.5,
                         color=INK2 if same else INK, weight="normal" if same else "bold")
        _dial(axd, cx["dep"], y, 110)
        if tok != "A":
            axd.text(cx["dep"] + 0.42, y - 0.32, "= A", fontsize=8.5, color=INK2)
    for key, txt in (("pseq", "sequence"), ("l1", "level 1:\nsection"),
                     ("l2", "level 2:\nparagraph"), ("dseq", "sequence"), ("dep", "depth")):
        axd.text(cx[key], 3.2, txt, ha="center", va="bottom", fontsize=8.5, color=INK)
    axd.text((cx["pseq"] + cx["l2"]) / 2, 3.95, "sequence + tree (16 + 44 planes)", ha="center",
             va="bottom", fontsize=9.5, color=INK, weight="bold")
    axd.text((cx["dseq"] + cx["dep"]) / 2, 3.95, "sequence + depth (16 + 2 planes)", ha="center",
             va="bottom", fontsize=9.5, color=INK, weight="bold")
    axd.text((cx["pseq"] + cx["l2"]) / 2, -0.3,
             "the branch is in the score: B matches A on\n\"which section\", C does not",
             ha="center", va="top", fontsize=8.5, color=INK2)
    axd.text((cx["dseq"] + cx["dep"]) / 2, -0.3,
             "the score sees the same depth and an order:\nB and C differ from A only in distance",
             ha="center", va="top", fontsize=8.5, color=INK2)
    axd.plot([4.7, 4.7], [-0.2, 4.3], color=GRID, linewidth=1.2)

    # --- (ii) the inference sequence + depth needs, and when the stream allows it
    axs = fig.add_axes([0.02, 0.02, 0.96, 0.27])
    axs.axis("off")
    axs.set_xlim(0, 100)
    axs.set_ylim(0, 10)
    axs.text(0, 9.6, "Sequence + depth can still find the branch, but only by inference across tokens: "
             "a token's parent is the nearest earlier token one level shallower. That needs the "
             "ancestors to be tokens in the stream.", fontsize=8.5, color=INK, va="top")

    def stream(y, items, label, verdict):
        axs.text(0, y, label, fontsize=8.5, color=INK, va="center")
        x = 29
        for txt, d, col in items:
            w = 6.0 if len(txt) > 3 else (3.6 if len(txt) > 1 else 2.6)
            if col:
                axs.text(x + w / 2, y, txt, ha="center", va="center", fontsize=8.5, color=SURFACE,
                         weight="bold", bbox=dict(boxstyle="circle,pad=0.25", facecolor=col, edgecolor="none"))
            else:
                axs.text(x + w / 2, y, txt, ha="center", va="center", fontsize=8, color=INK,
                         bbox=dict(boxstyle="round,pad=0.25", facecolor=PALE, edgecolor=AXIS, linewidth=0.6))
            axs.text(x + w / 2, y - 1.55, f"d{d}", ha="center", va="center", fontsize=7, color=MUTED)
            x += w + 0.4
        axs.text(x + 1.0, y, verdict, fontsize=8.5, color=INK2, va="center")

    stream(5.6, [("<section>", 1, None), ("<p>", 2, None), ("A", 3, S1), ("<p>", 2, None), ("B", 3, S1),
                 ("<section>", 1, None), ("<p>", 2, None), ("C", 3, S2)],
           "ancestors are tokens\n(code: every parent recovered)",
           "a <section> at depth 1 lies between\nB and C, so C is in a new section")
    stream(1.7, [("A", 3, S1), ("B", 3, S1), ("C", 3, S2)],
           "only the text is in the stream\n(web text, as trained)",
           "nothing between the tokens to infer from:\nthe branch is not recoverable")
    fig.suptitle("The tree path puts the branch in the attention score; sequence + depth must infer it,"
                 " and can only when ancestors are in the stream",
                 x=0.01, y=0.995, ha="left", color=INK, fontsize=11.5)
    save(fig, "fig_path_vs_depth")


def fig_reference() -> None:
    """A formula's reference, rotated to where the referenced cell is. Schematic, not data.

    Sheet1 holds A1 = 10, A2 = 20, A3 = "=A1+A2". The token "A1" inside A3's formula belongs to
    cell A3, so under sequence + tree it carries A3's coordinates and differs from the cell it
    names on the row level. Under reference rotation it carries A1's coordinates on the tree-path
    planes, so it agrees with its referent on every level, exactly as two tokens of one cell do.
    Angles exaggerated (one rank = 55 degrees).
    """
    fig = plt.figure(figsize=(11.5, 4.6))
    # --- left: the sheet as a grid, with the reference drawn as an arrow
    axg = fig.add_axes([0.01, 0.06, 0.30, 0.80])
    axg.axis("off")
    axg.set_xlim(0, 10)
    axg.set_ylim(0, 10)
    axg.text(0.3, 9.3, "Sheet1", fontsize=9, color=INK, weight="bold")
    cw, ch, x0, ytop = 4.6, 1.5, 2.6, 8.4
    axg.text(x0 + cw / 2, ytop + 0.25, "A", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    vals = ("10", "20", None)
    for r in range(3):
        y = ytop - (r + 1) * ch
        axg.add_patch(matplotlib.patches.Rectangle((x0, y), cw, ch, facecolor=SURFACE,
                                                   edgecolor=AXIS, linewidth=1.0))
        axg.text(x0 - 0.3, y + ch / 2, str(r + 1), ha="right", va="center", fontsize=8.5, color=MUTED)
        if vals[r]:
            axg.text(x0 + cw / 2, y + ch / 2, vals[r], ha="center", va="center", fontsize=10, color=INK)
    y3 = ytop - 3 * ch + ch / 2
    axg.text(x0 + 0.5, y3, "=", ha="left", va="center", fontsize=10, color=INK)
    axg.text(x0 + 1.45, y3, "A1", ha="center", va="center", fontsize=10, color=INK, weight="bold",
             bbox=dict(boxstyle="round,pad=0.15", facecolor=PALE, edgecolor=S2, linewidth=1.4))
    axg.text(x0 + 2.05, y3, "+A2", ha="left", va="center", fontsize=10, color=INK)
    y1 = ytop - ch / 2
    axg.annotate("", xy=(x0 + cw + 0.1, y1), xytext=(x0 + cw + 0.1, y3),
                 arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.2,
                                 connectionstyle="arc3,rad=0.55", mutation_scale=11))
    axg.text(x0 + cw + 1.25, (y1 + y3) / 2, "refers\nto", ha="left", va="center", fontsize=8, color=INK2)
    for lab, y, c in (("R", y1, S1), ("F", y3, S2)):
        axg.text(x0 - 1.4, y, lab, ha="center", va="center", fontsize=10, color=SURFACE, weight="bold",
                 bbox=dict(boxstyle="circle,pad=0.3", facecolor=c, edgecolor="none"))
    axg.text(0.3, 1.0, "R: the cell A1 itself (sheet 1, row 1, column A)\n"
             "F: the token \"A1\" inside A3's formula", fontsize=8, color=INK2, va="bottom")
    # --- right: dials. columns = sheet, row, column ; rows = R, F (seq + tree), F (reference)
    axd = fig.add_axes([0.34, 0.06, 0.65, 0.80])
    axd.axis("off")
    axd.set_xlim(-2.6, 5.4)
    axd.set_ylim(-0.9, 3.9)
    axd.set_aspect("equal", adjustable="datalim")
    step, theta = 55, (0, 20, 40)
    rows = (("R", "the cell A1", (0, 0, 0), S1, 2.6),
            ("F", "under sequence + tree:\ncarries its own cell, A3", (0, 2, 0), S2, 1.5),
            ("F", "under reference rotation:\ncarries the cell it names", (0, 0, 0), S2, 0.4))
    colx = [0.9, 2.3, 3.7]
    for tok, desc, ranks, c, y in rows:
        axd.text(-2.35, y, tok, ha="center", va="center", fontsize=11, color=SURFACE, weight="bold",
                 bbox=dict(boxstyle="circle,pad=0.3", facecolor=c, edgecolor="none"))
        axd.text(-2.0, y, desc, ha="left", va="center", fontsize=8, color=INK2)
        for lv in range(3):
            same = ranks[lv] == rows[0][2][lv]
            _dial(axd, colx[lv], y, theta[lv] + ranks[lv] * step,
                  ring=AXIS if same else INK, face=SURFACE if same else PALE)
            if tok == "F":
                axd.text(colx[lv] + 0.45, y - 0.3, "= R" if same else "≠ R", fontsize=8.5,
                         color=INK2 if same else INK, weight="normal" if same else "bold")
    for x, txt in zip(colx, ("sheet", "row", "column")):
        axd.text(x, 3.15, txt, ha="center", va="bottom", fontsize=8.5, color=INK)
    axd.text(colx[1], -0.3, "with reference rotation the reference and its referent agree on every\n"
             "tree-path plane, as two tokens of the same cell do",
             ha="center", va="top", fontsize=8.5, color=INK2)
    fig.suptitle("Reference rotation puts a formula's reference where the referenced cell is",
                 x=0.01, y=0.99, ha="left", color=INK, fontsize=11.5)
    save(fig, "fig_reference")


def fig_rotation() -> None:
    """One rotary plane: each token's vector is turned by its position, so the angle BETWEEN two
    tokens depends only on how far apart they are, not where they sit."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    deg = 22
    for ax, (m, n) in zip(axes, ((2, 5), (9, 12))):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Circle((0, 0), 1, facecolor="none", edgecolor=GRID, linewidth=1))
        for pos, c, nm in ((m, S1, "query"), (n, S2, "key")):
            a = math.radians(pos * deg)
            ax.annotate("", xy=(math.cos(a), math.sin(a)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=c, linewidth=2.2, mutation_scale=12))
            ax.text(1.18 * math.cos(a), 1.18 * math.sin(a), f"{nm}\nposition {pos}", ha="center",
                    va="center", fontsize=8, color=INK2)
        a0, a1 = math.radians(m * deg), math.radians(n * deg)
        arc = matplotlib.patches.Arc((0, 0), 0.7, 0.7, theta1=math.degrees(a0), theta2=math.degrees(a1),
                                     color=INK, linewidth=1.2)
        ax.add_patch(arc)
        mid = (a0 + a1) / 2
        ax.text(0.62 * math.cos(mid), 0.62 * math.sin(mid), "3 steps", fontsize=8, color=INK,
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                                                    edgecolor="none"))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.45, 1.5)
    fig.suptitle("Rotary position, one plane: positions 2→5 and 9→12 meet at the same angle",
                 x=0.02, ha="left", color=INK, fontsize=10.5)
    save(fig, "fig_rotation")


UND = f"{RUNS}/analysis/understanding"
# setting -> (title, [(category, plain label)], [(arm, label, colour, marker)]). Categories are the
# understanding_evals.py test categories; "all" is the average every other figure reports.
UND_PANELS = [
    ("sheets", "Spreadsheets (10 seeds)",
     [("all", "all tokens (the average)"), ("sh.newline", "row ending"),
      ("sh.ref", "cell reference in a formula"),
      ("sh.copy_above_far", "copy from the cell above, ≥256 tokens back"),
      ("sh.formula", "any formula token"), ("sh.value", "plain cell value")],
     [("both_v2", "sequence + tree", S1, "o"), ("grid_v2", "sequence + grid", S2, "^"),
      ("depth_v2", "sequence + depth", S3, "s"), ("ref_anchor", "reference rotation", INK, "P")]),
    ("webc", "Web pages (10 seeds)",
     [("all", "all tokens (the average)"), ("web.node_start", "first token of a new text block"),
      ("web.node_end", "end of a text block"), ("web.rep_cross", "word last seen in another branch"),
      ("web.cont", "inside a text block")],
     [("both", "sequence + tree", S1, "o"), ("depth", "sequence + depth", S3, "s"),
      ("rel", "sequence + tree + depth", V7, "D"), ("drank", "sequence + depth + rank", S3, "v")]),
    ("py", "Python (5 seeds)",
     [("all", "all tokens (the average)"), ("py.ascend", "a subtree has just closed"),
      ("py.descend", "opening a child node"), ("py.ident_reuse", "reusing a variable"),
      ("py.ident_new", "binding a new variable")],
     [("rope_both", "sequence + tree", S1, "o"), ("depth", "sequence + depth", S3, "s"),
      ("rel", "sequence + tree + depth", V7, "D"), ("drank", "sequence + depth + rank", S3, "v")]),
]


def fig_understanding() -> None:
    """Where the gain lands: perplexity change per kind of token, against sequence only, test split.

    The headline numbers average over every token, and most tokens do not depend on structure.
    This splits the same paired comparison by what is being predicted (understanding_evals.py).
    UNITS (referee, 2026-10-03): perplexity reduction on that category, 100*(1 - e^delta), the same
    unit as every other figure -- an earlier version used % of the category's loss, which is a
    different scale (a 0.2-nat category moved by 0.02 nats was "10%" there and 2% here). Each panel
    has its own x-scale. Faint markers are seeds; the line is the 95% CI of the paired mean.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 11.0), gridspec_kw=dict(hspace=0.42))
    for ax, (setting, title, cats, arms) in zip(axes, UND_PANELS):
        rows = json.load(open(os.path.join(UND, f"{setting}_results.json")))
        R = {(r["arm"], r["cat"]): r for r in rows}
        lo, hi = 0.0, 0.0
        for yi, (cat, lab) in enumerate(cats):
            for k, (arm, name, col, mk) in enumerate(arms):
                r = R.get((arm, cat))
                if r is None:
                    continue
                d = r["diffs"]
                y = yi + (k - (len(arms) - 1) / 2) * 0.19
                pts = [ppl_pct(x) for x in d]
                m, ci, _, _ = summary(d)
                v = ppl_pct(m)
                hollow = mk == "v"
                for i, x in enumerate(pts):
                    ax.plot(x, y + ((i % 5) - 2) * 0.015, mk, ms=3.0, color=col, alpha=0.4,
                            mfc="none" if hollow else col, mec=col if hollow else "none", zorder=2)
                if ci is not None:
                    ax.plot([ppl_pct(m + ci), ppl_pct(m - ci)], [y, y], color=col, linewidth=1.3, zorder=3)
                ax.plot(v, y, mk, ms=7, color=col, mfc=SURFACE if hollow else col,
                        mec=col if hollow else SURFACE, mew=1.6 if hollow else 0.8, zorder=4)
                lo, hi = min(lo, min(pts)), max(hi, max(pts))
                print(f"  understanding {setting:6s} {cat:18s} {arm:12s} {v:+.2f}%  {r['won']}/{r['n_seeds']}")
        ax.axvline(0, color=INK2, linewidth=1.0, zorder=1)
        ax.axhspan(-0.5, 0.5, color=GRID, alpha=0.45, zorder=0)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([c[1] for c in cats], fontsize=10)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_ylim(len(cats) - 0.5, -0.6)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        span = hi - lo
        ax.set_xlim(lo - 0.04 * span, hi + 0.04 * span)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        handles = [matplotlib.lines.Line2D([], [], color=c, marker=m, linestyle="none", ms=7,
                                           mfc=SURFACE if m == "v" else c, mew=1.6 if m == "v" else 0.8)
                   for _, _, c, m in arms]
        ax.legend(handles, [n for _, n, _, _ in arms], loc="upper center", bbox_to_anchor=(0.45, -0.07),
                  ncol=len(arms), frameon=False, fontsize=9, labelcolor=INK2, columnspacing=1.0,
                  handletextpad=0.3)
    axes[-1].set_xlabel("perplexity reduction on that kind of token, against sequence only "
                        "(right = better)", fontsize=10, labelpad=26)
    fig.suptitle("Where the gain lands: the tokens that depend on structure gain most",
                 x=0.01, y=0.962, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.935, "held-out test split; large marker = mean, line = 95% CI, faint markers = "
             "seeds; shaded row = all tokens. Each panel has its own scale.", color=MUTED, fontsize=9)
    save(fig, "fig_understanding")


# ---------------------------------------------------------------- gain against cost

# (key, label, colour, marker, filled). One hue per FAMILY: depth-family green, grid-family orange,
# full tree blue, tree + depth violet, full RoPE grey -- the same four validated hues as every other
# figure; variants inside a family are told apart by marker, hollow for the "adds a label to depth"
# variants that measure like depth.
PVG_STYLE = {
    "depth": ("sequence + depth", S3, "s", True),
    "drank": ("+ sibling rank", S3, "v", False),
    "dparent": ("+ parent tag", S3, "P", False),
    "parity": ("+ parent/child IDs", S3, "X", False),
    "grid": ("sequence + grid", S2, "^", True),
    "gdepth": ("grid + depth", S2, "D", True),
    "both": ("full tree (44 planes)", S1, "o", True),
    "rel": ("tree + depth (46)", V7, "D", True),
    "full": ("full RoPE (+48 sequence)", MUTED, "x", True),
}
# per setting: (title, root, baseline, [(style key, arm, extra planes)])
PVG_SETS = [
    ("Python (5 seeds)", "py", "rope_seq",
     [("depth", "depth", 2), ("drank", "drank", 4), ("grid", "stmtgrid", 6), ("gdepth", "stmtgrid_depth", 8),
      ("both", "rope_both", 44), ("rel", "rel", 46), ("full", "full_rope", 48)]),
    ("Spreadsheets (10)", "sheets", "seq_v2",
     [("depth", "depth_v2", 2), ("grid", "grid_v2", 6), ("both", "both_v2", 44), ("rel", "rel_v2", 46),
      ("full", "full_rope_v2", 48)]),
    ("Web (10)", "webc", "seq",
     [("depth", "depth", 2), ("drank", "drank", 4), ("both", "both", 44), ("rel", "rel", 46), ("full", "full_rope", 48)]),
    ("Chat (10)", "chat", "seq",
     [("depth", "depth", 2), ("drank", "drank", 4), ("grid", "grid", 8), ("gdepth", "grid_depth", 10),
      ("both", "both", 44), ("rel", "rel", 46)]),
]
# the appendix version adds every label-on-depth variant we tried
PVG_EXTRA = {
    "py": [("dparent", "dparentg", 4), ("parity", "parityid", 9)],
    "sheets": [("dparent", "dparentg_v2", 4)],
    "webc": [("dparent", "dparentg", 4)],
    "chat": [("dparent", "dparentg", 4)],
}
# jitter so markers at equal plane counts do not sit on top of each other (log axis: multiplicative)
PVG_JIT = {"drank": 0.93, "dparent": 1.08, "parity": 1.0, "grid": 1.0, "gdepth": 1.0, "full": 1.06}


def fig_planes_vs_gain(full: bool = False) -> None:
    """Perplexity reduction against the number of extra rotary planes it costs, per setting.

    The paper's practical claim in one picture: a cheap coordinate (2-10 planes) gets most or all of
    the 44-plane tree's gain, and WHICH cheap coordinate depends on the data's shape -- depth for
    nested text, a grid for tables and chat, depth + a statement grid for code. Mean over seeds of
    the paired gap, 95% CI from the paired t; Web-L is one run on its test split, so no interval.
    ``full`` adds every label-on-depth variant (appendix).
    """
    cols = len(PVG_SETS) + 1
    fig, axes = plt.subplots(1, cols, figsize=(6.9 if not full else 11.5, 2.15 if not full else 3.3),
                             gridspec_kw={"wspace": 0.38})
    used = []
    for ax, (title, root, base, arms) in zip(axes, PVG_SETS):
        arms = arms + (PVG_EXTRA.get(root, []) if full else [])
        lo = hi = 0.0
        for key, arm, planes in arms:
            d = [v for _, v in (pairs_finished(root, arm, base, FINAL[root]))]
            if len(d) < 2:
                continue
            m, ci, _, _ = summary(d)
            lab, col, mk, filled = PVG_STYLE[key]
            x = planes * PVG_JIT.get(key, 1.0)
            y, ylo, yhi = ppl_pct(m), ppl_pct(m + ci), ppl_pct(m - ci)
            lo, hi = min(lo, ylo), max(hi, yhi)
            ax.plot([x, x], [ylo, yhi], color=col, linewidth=1.1, zorder=3, alpha=0.9)
            line_mk = mk in "x+"           # line markers have no face: draw them in their colour
            ax.plot(x, y, mk, ms=5 if not full else 7, color=col, mfc=col if filled else SURFACE,
                    mec=col if (line_mk or not filled) else SURFACE, mew=1.3 if (line_mk or not filled) else 0.6,
                    zorder=4)
            used.append(key)
            print(f"  planes-vs-gain {root:6s} {arm:16s} {planes:3d} planes  {y:+.2f}%  [{ylo:+.2f}, {yhi:+.2f}]  n={len(d)}")
        _pvg_axes(ax, title, lo, hi, full)
    # Web-L: one run each, test split
    ax = axes[-1]
    seq = web7_test("rope_seq_p1")[1]
    lo = hi = 0.0
    for key, arm, planes in (("depth", "rope_depth_p1", 2), ("both", "rope_both_p1", 44)):
        t = web7_test(arm)
        if t is None:
            continue
        y = ppl_pct(t[1] - seq)
        lab, col, mk, filled = PVG_STYLE[key]
        ax.plot(planes, y, mk, ms=5 if not full else 7, color=col, mec=SURFACE, mew=0.6, zorder=4)
        lo, hi = min(lo, y), max(hi, y)
        print(f"  planes-vs-gain web7   {arm:16s} {planes:3d} planes  {y:+.2f}%  (1 run, test)")
    _pvg_axes(ax, "Larger run (1 run, test)", lo, hi, full)
    axes[0].set_ylabel("perplexity reduction, %\n(above 0 = better)", fontsize=7 if not full else 9, labelpad=2)
    keys = [k for k in PVG_STYLE if k in used]
    handles = [matplotlib.lines.Line2D([], [], color=PVG_STYLE[k][1], marker=PVG_STYLE[k][2], linestyle="none",
                                       ms=5 if not full else 7, mfc=PVG_STYLE[k][1] if PVG_STYLE[k][3] else SURFACE,
                                       mec=PVG_STYLE[k][1] if (PVG_STYLE[k][2] in "x+" or not PVG_STYLE[k][3]) else SURFACE,
                                       mew=1.3 if (PVG_STYLE[k][2] in "x+" or not PVG_STYLE[k][3]) else 0.6)
               for k in keys]
    labels = [PVG_STYLE[k][0] for k in keys]
    labels = ["seq + statement grid (code) /\nseq + grid (tables, chat)" if k == "grid" else
              ("statement grid + depth (code) /\ngrid + depth (chat)" if k == "gdepth" else l)
              for k, l in zip(keys, labels)]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4 if not full else 5,
               frameon=False, fontsize=6.5 if not full else 9, labelcolor=INK2, handletextpad=0.3,
               columnspacing=1.0)
    save(fig, "fig_planes_vs_gain_full" if full else "fig_planes_vs_gain")


def _pvg_axes(ax, title, lo, hi, full) -> None:
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([2, 8, 44])
    ax.set_xticklabels(["2", "8", "44"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.6, 62)
    pad = 0.12 * (hi - lo or 1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}" if v else "0"))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fs = 6.5 if not full else 8.5
    ax.tick_params(labelsize=fs, length=2, pad=1.5)
    ax.set_title(title, loc="left", color=INK, fontsize=7.5 if not full else 10, pad=3)
    ax.set_xlabel("extra planes (log)", fontsize=7 if not full else 9, labelpad=1)


def main() -> None:
    fig1_headlines()
    fig_configs()
    fig_planes_vs_gain()
    fig_planes_vs_gain(full=True)
    fig2_web_curves()
    fig3_depth()
    fig4_sheets()
    fig_loss_curves()
    fig_loss_curves_compact()
    fig_loss_raw()
    fig_planes()
    fig_understanding()
    fig_understanding_compact()
    fig_path_vs_depth()
    fig_reference()
    fig_rotation()


if __name__ == "__main__":
    main()
