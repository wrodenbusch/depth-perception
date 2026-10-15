#!/usr/bin/env python
"""Tables and figures for the mechanism analyses (analysis/mechanism.py). CPU only, except
the worked example, which recomputes one record's attention on the card given by
CUDA_VISIBLE_DEVICES.

    .venv/bin/python analysis/mechanism_report.py            # tables + heatmap figure
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python analysis/mechanism_report.py --example
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "analysis"))
import mechanism as M  # noqa: E402

OUT = M.OUT
FIGS = os.path.join(HERE, "docs", "figures")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, RED, MID, ORANGE = "#2a78d6", "#e34948", "#f0efec", "#eb6834"


def ppl(d):
    """Loss change (nats/token) as % perplexity change; positive = worse."""
    return 100 * (math.exp(d) - 1)


def n_better(gain):
    return sum(x > 0 for x in gain)


def remove_table():
    lines = []
    summary = {}
    for setting, cfg in M.SETTINGS.items():
        p = os.path.join(OUT, f"remove_{setting}.json")
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        lines.append(f"\n**{M.SETTING_NAME[setting]}** (test split, first {cfg['remove_records']} "
                     f"records, seeds {M.SEEDS})\n")
        lines.append("| configuration | gain over sequence only | cost of removing structure | "
                     "cost of shuffling structure | removed, vs sequence only |")
        lines.append("|---|---|---|---|---|")
        for arm in cfg["arms"]:
            gain, rem, shuf, vs_seq = [], [], [], []
            for s in M.SEEDS:
                a, b = R.get(f"{arm}_s{s}"), R.get(f"{cfg['seq']}_s{s}")
                if not a or not b:
                    continue
                gain.append(b["true"] - a["true"])
                rem.append(a["absent"] - a["true"])
                shuf.append(a["shuffle"] - a["true"])
                vs_seq.append(a["absent"] - b["true"])
            if not gain:
                continue
            g, r, sh, v = (M.paired(x) for x in (gain, rem, shuf, vs_seq))
            summary[(setting, arm)] = dict(gain=g, remove=r, shuffle=sh, vs_seq=v)
            f = lambda d: (f"{ppl(d['mean']):+.1f}% ({d.get('pos', '-')}/{d['n']} seeds" +
                           (f", p = {d['p']:.2g})" if 'p' in d else ")"))
            gtxt = f"{-ppl(-g['mean']):.2f}% better ({n_better(gain)}/{g['n']})"
            lines.append(f"| {M.PRETTY[arm]} | {gtxt} | {f(r)} | {f(sh)} | {f(v)} |")
    return "\n".join(lines), summary


def attn_summary():
    """setting -> arm -> rel -> (L,H) arrays of mass diff vs seq per seed, plus seq lift."""
    out = {}
    for setting, cfg in M.SETTINGS.items():
        p = os.path.join(OUT, f"attn_{setting}.json")
        if not os.path.exists(p):
            continue
        A = json.load(open(p))
        out[setting] = {}
        for arm in cfg["attn_arms"]:
            per_rel = {}
            for s in M.SEEDS:
                a, b = A.get(f"{arm}_s{s}"), A.get(f"{cfg['seq']}_s{s}")
                if not a or not b:
                    continue
                for rel, grid in a["rel"].items():
                    if rel not in b["rel"]:
                        continue
                    da = np.array([[c["mass"] for c in row] for row in grid])
                    db = np.array([[c["mass"] for c in row] for row in b["rel"][rel]])
                    ch = np.array([[c["chance"] for c in row] for row in grid])
                    per_rel.setdefault(rel, dict(diff=[], seq=[], struct=[], chance=[]))
                    per_rel[rel]["diff"].append(da - db)
                    per_rel[rel]["seq"].append(db)
                    per_rel[rel]["struct"].append(da)
                    per_rel[rel]["chance"].append(ch)
            split = []
            for s in M.SEEDS:
                a = A.get(f"{arm}_s{s}")
                if a:
                    split.append([[(c["dstruct"] / c["tot"]) if c["tot"] else np.nan for c in row]
                                  for row in a["split"]])
            out[setting][arm] = dict(rel={k: {kk: np.array(vv) for kk, vv in v.items()}
                                          for k, v in per_rel.items()},
                                     split=np.array(split, dtype=float))
    return out


def attn_table(S):
    lines = []
    for setting, arms in S.items():
        for arm, d in arms.items():
            lines.append(f"\n**{M.SETTING_NAME[setting]}, {M.PRETTY[arm]} vs sequence only** "
                         f"(mean over all heads; seeds = {d['split'].shape[0]})\n")
            lines.append("| keys in relation | share under uniform attention | sequence only | "
                         "structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for rel, v in d["rel"].items():
                per_seed = v["diff"].mean(axis=(1, 2))
                t = M.paired(list(per_seed))
                # heads are not aligned across seeds, so take each seed's most-moved head
                top = [v["diff"][s].ravel()[np.argmax(np.abs(v["diff"][s].ravel()))]
                       for s in range(v["diff"].shape[0])]
                lines.append(
                    f"| {rel} | {100 * v['chance'].mean():.1f}% | {100 * v['seq'].mean():.1f}% | "
                    f"{100 * v['struct'].mean():.1f}% | {100 * t['mean']:+.1f} | {t.get('pos', '-')}/{t['n']} | "
                    f"{t.get('p', float('nan')):.2g} | {100 * float(np.median(top)):+.1f} |")
            sp = d["split"].ravel()
            lines.append(f"\nSwitching the structure rotation on changes a head's attention scores by "
                         f"{np.nanpercentile(sp, 10):.2f}-{np.nanpercentile(sp, 90):.2f} (10th-90th "
                         f"percentile, median {np.nanmedian(sp):.2f}) of the spread of those scores, "
                         f"over all {np.isfinite(sp).sum()} heads x seeds (std of the change / std of "
                         f"the score, on 20,000 sampled query-key pairs per record).")
    return "\n".join(lines)


def stripplot(S, path):
    """Per relation: change in attention mass for EVERY head of every seed (dots) + mean.

    Heads are not aligned across seeds (head 2 of layer 4 in seed 0 has no counterpart in seed
    1), so a head-indexed average would mix unrelated heads. Each dot is one head of one seed;
    the bar is the mean over all of them with its seed-level 95% CI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    VIOLET = "#4a3aa7"
    panels = [("sheets", [("grid_v2", BLUE), ("rel_v2", VIOLET)]),
              ("webc", [("depth", BLUE), ("rel", VIOLET)]),
              ("py", [("depth", BLUE), ("rel", VIOLET)])]
    panels = [(s, [(a, c) for a, c in arms if a in S.get(s, {})]) for s, arms in panels if s in S]
    heights = [len(S[s][arms[0][0]]["rel"]) for s, arms in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 0.9 + 0.55 * sum(heights)),
                             gridspec_kw=dict(height_ratios=heights))
    rng = np.random.default_rng(0)
    for ax, (s, arms) in zip(axes, panels):
        rels = list(S[s][arms[0][0]]["rel"])
        for k, (a, color) in enumerate(arms):
            off = -0.17 + 0.34 * k
            for i, r in enumerate(rels):
                d = 100 * S[s][a]["rel"][r]["diff"]  # (seeds, L, H)
                y = i + off
                ax.plot(d.ravel(), y + rng.uniform(-0.08, 0.08, d.size), "o", ms=3.2, color=color,
                        alpha=0.45, mec="none")
                per_seed = d.mean(axis=(1, 2))
                m = per_seed.mean()
                se = per_seed.std(ddof=1) / math.sqrt(len(per_seed)) if len(per_seed) > 1 else 0
                ax.plot([m - 2.78 * se, m + 2.78 * se], [y, y], color=INK, lw=1.6)
                ax.plot(m, y, "|", ms=10, mew=2.2, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(rels)), rels)
        ax.invert_yaxis()
        ax.set_title(M.SETTING_NAME[s], loc="left", fontsize=10, color=INK)
        for spn in ("top", "right", "left"):
            ax.spines[spn].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(1.0, 1.02, "blue: " + M.PRETTY[arms[0][0]] + "   violet: " + M.PRETTY[arms[1][0]],
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_xlabel("change in attention mass vs sequence only, same seed (pp; right = more)",
                      fontsize=8)
    fig.suptitle("Where the structure moves attention. Each dot is one head of one seed; the black\n"
                 "bar is the mean over heads with a 95% CI over 5 seeds. Each panel has its own scale.",
                 x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=170)
    fig.savefig(path + ".svg")
    plt.close(fig)


def grid_map_all(path, setting="sheets", arms=("seq_v2", "grid_v2"), seeds=(0, 1, 2, 3, 4),
                 n_rec=8, R=4, C=3):
    """The AVERAGE head, in grid coordinates: attention mass from every cell token to keys at
    (row offset, column offset), averaged over all 12 heads, 5 seeds and the test spreadsheets.

    Averaging over every head avoids picking one: a sequence-only model also grows a head that
    attends down its column (row lengths repeat, so a fixed sequence offset often lands there),
    so a single hand-chosen head would overstate the difference. The average head shows how much
    of the network's attention the grid coordinates move, which is the claim being made.
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cfg = M.SETTINGS[setting]
    dev = torch.device("cuda")
    from topos_data import PackedWindows
    base = PackedWindows(cfg["base_pack"], need_tree=False, sidecar="path")
    lo, _ = base.splits["test"]
    maps, per_seed = {}, {}
    for a in arms:
        tot = np.zeros((R + 1, 2 * C + 1))
        per_seed[a] = []
        for seed in seeds:
            model, margs, pack = M.load_model(M.run_dir(setting, a, seed), cfg["final"], dev)
            pw = M.windows(pack, margs)
            acc = np.zeros((R + 1, 2 * C + 1))
            nq = 0
            for r in range(lo, lo + n_rec):
                X, Y, POS = pw.get_batch_records([r], dev)
                _, _, BP = base.get_batch_records([r], dev)
                bp = BP["path"][0]
                T = X.shape[1]
                cell = (bp >= 0).sum(-1) >= 3
                same_sheet = bp[:, 0][:, None] == bp[:, 0][None]
                drow = bp[:, 1][:, None] - bp[:, 1][None]
                dcol = bp[:, 2][None] - bp[:, 2][:, None]
                ok = cell[:, None] & cell[None] & same_sheet
                qs = torch.nonzero(cell).flatten()
                store, hooks = M.capture(model)
                with torch.no_grad():
                    model(X, Y, pos=POS)
                    for hk in hooks:
                        hk.remove()
                    mean_att = torch.zeros(T, T, device=dev)
                    nheads = 0
                    for li, (x, rot) in enumerate(store):
                        q, k = M.qk(model.transformer.h[li].attn, x, rot)
                        att = (q[0] @ k[0].transpose(-2, -1)) / math.sqrt(q.shape[-1])
                        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1),
                                              float("-inf")).softmax(-1)
                        mean_att += att.sum(0)
                        nheads += att.shape[0]
                    mean_att /= nheads
                    for dr in range(R + 1):
                        for dc in range(-C, C + 1):
                            m = ok & (drow == dr) & (dcol == dc)
                            acc[dr, dc + C] += float((mean_att * m)[qs].sum())
                nq += len(qs)
            acc /= max(nq, 1)
            per_seed[a].append(acc)
            tot += acc
            del model
            torch.cuda.empty_cache()
        maps[a] = tot / len(seeds)
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    vmax = max(m.max() for m in maps.values()) * 100
    fig, axes = plt.subplots(1, len(arms), figsize=(7.2, 2.7), sharey=True)
    for ax, a in zip(axes, arms):
        m = 100 * maps[a]
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j] >= 0.5:
                    ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if m[i, j] > vmax * 0.55 else INK)
        ax.set_xticks(range(2 * C + 1), [f"{c:+d}" if c else "0" for c in range(-C, C + 1)])
        ax.set_yticks(range(R + 1), ["same row"] + [f"{r} row{'s' if r > 1 else ''} up" for r in range(1, R + 1)])
        ax.set_xlabel("column offset from the query's cell")
        ax.set_title(M.PRETTY[a], loc="left", fontsize=9.5, color=INK)
        for spn in ax.spines.values():
            spn.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("% of attention (average head)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Where the average attention head looks, in grid coordinates (spreadsheets; all 12 heads,\n"
                 f"{len(seeds)} seeds, every cell token of {n_rec} test spreadsheets; same row, offset 0 = "
                 f"the query's own cell)", x=0.02, y=1.10, ha="left", fontsize=10, color=INK)
    fig.savefig(path + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(path + ".svg", bbox_inches="tight")
    plt.close(fig)
    col = {a: [100 * p[1:, C].sum() for p in per_seed[a]] for a in arms}
    row_left = {a: [100 * p[0, :C].sum() for p in per_seed[a]] for a in arms}
    diffs = [g - s for s, g in zip(col[arms[0]], col[arms[1]])]
    info = dict(maps={a: (100 * maps[a]).round(2).tolist() for a in arms},
                same_column_rows_above=col, same_row_left=row_left,
                column_change=M.paired(diffs))
    json.dump(info, open(os.path.join(OUT, "grid_map_sheets.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in info.items() if k != "maps"}, indent=1, default=float))


def steps_table():
    rows = json.load(open(os.path.join(OUT, "steps.json")))
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["setting"], r["arm"])].append(r)
    lines = ["| setting | configuration | learned steps, fast / slow plane (init 0.316 / 0.020 rad) |",
             "|---|---|---|"]
    for (s, a), v in g.items():
        parts = []
        if "rope.step" in v[0]:
            S = np.array([r["rope.step"] for r in v])
            names = ["sheet", "row", "column"] if s == "sheets" else [f"level {k + 1}" for k in range(3)]
            for k in range(min(3, S.shape[1])):
                parts.append(f"{names[k]} {S[:, k, 0].mean():.2f} / {S[:, k, 1].mean():.3f}")
        if "rope.step_rel" in v[0]:
            R = np.array([r["rope.step_rel"] for r in v])
            parts.append(f"depth {R[:, 0].mean():.2f} / {R[:, 1].mean():.2f}")
        lines.append(f"| {M.SETTING_NAME[s]} | {M.PRETTY[a]} | {'; '.join(parts)} (n={len(v)}) |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", action="store_true")
    a = ap.parse_args()
    if a.example:
        grid_map_all(os.path.join(FIGS, "fig_mechanism_example"))
        return
    rt, _ = remove_table()
    S = attn_summary()
    at = attn_table(S)
    stripplot(S, os.path.join(FIGS, "fig_mechanism_attention"))
    md = ("# Mechanism analyses\n\n## 1. Removing the structure at test time\n" + rt +
          "\n\n## 2-3. Attention on structurally related keys, and how much the rotation moves scores\n" + at +
          "\n\n## 4. Learned rotation steps (shared by every head and layer)\n\n" + steps_table() + "\n")
    open(os.path.join(OUT, "results.md"), "w").write(md)
    open(os.path.join(HERE, "docs", "mechanism", "results.md"), "w").write(md) if os.path.isdir(
        os.path.join(HERE, "docs", "mechanism")) else None
    print(md)


if __name__ == "__main__":
    main()
