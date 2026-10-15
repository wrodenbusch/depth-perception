#!/usr/bin/env python
"""Paired per-seed comparison of two arms. The project's standard read of a result.

    .venv/bin/python paired.py --root $TOPOS_RUNS/py --a depth --b toplevel
    .venv/bin/python paired.py --root $TOPOS_RUNS/sheets --a ref_anchor --b both_v2 --at best

WHY PAIRED. Seed-to-seed spread on these corpora (0.004-0.012 nats/token) is comparable to the
effects being measured, so two arm MEANS with error bars cannot separate them. Training the two
arms from the SAME seed makes the difference per seed the unit of observation, and the sign of
that difference carries more than its size: an effect that is negative on 5 of 5 seeds is worth
more than a larger mean that is negative on 3.

WHY THE t-DISTRIBUTION IS INTEGRATED HERE. An earlier hand-rolled approximation in a throwaway
snippet returned p > 1 and negative p, and its numbers reached a status update before anyone
noticed. This integrates the t pdf by Simpson's rule instead, and ``--self-test`` checks it
against published critical values. Do not replace it with an approximation that is not tested.

REPORTS BOTH `final` AND `best` on purpose. A run's lowest validation milestone and its last
one are different estimators, and the project has seen two checkpoints indistinguishable on one
held-out set differ markedly on another. Where they disagree, that disagreement is the finding.
"""
from __future__ import annotations
from topos_paths import RUNS  # noqa: E402

import argparse
import glob
import json
import math
import os
import statistics as st


def t_pdf(x: float, df: float) -> float:
    return (math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
            * (1 + x * x / df) ** (-(df + 1) / 2))


def t_sf(t: float, df: float, n: int = 20001) -> float:
    """P(T > t) by Simpson's rule out to a far tail, plus an analytic remainder bound.

    The pdf decays like |x|^-(df+1), so the tail beyond ``hi`` is bounded by integrating that
    power law -- included rather than dropped so the result is never silently short.
    """
    t = float(t)
    if t < 0:
        return 1.0 - t_sf(-t, df, n)
    hi = max(t + 60.0, t * 20.0, 200.0)
    n = n if n % 2 == 1 else n + 1
    h = (hi - t) / (n - 1)
    s = 0.0
    for i in range(n):
        x = t + i * h
        w = 1 if i in (0, n - 1) else (4 if i % 2 else 2)
        s += w * t_pdf(x, df)
    area = s * h / 3
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * df ** ((df + 1) / 2)
    area += c * hi ** (-df) / df                      # tail bound beyond hi
    return min(max(area, 0.0), 1.0)


def p_two_sided(t: float, df: float) -> float:
    return min(1.0, 2 * t_sf(abs(t), df))


def self_test() -> None:
    """Published two-sided 5% critical values: p must come back 0.0500."""
    for df, tc in ((1, 12.706), (2, 4.303), (4, 2.776), (9, 2.262), (30, 2.042), (100, 1.984)):
        p = p_two_sided(tc, df)
        assert abs(p - 0.05) < 5e-4, f"df={df} t={tc} -> p={p:.5f}, expected 0.0500"
        print(f"  df {df:>3}  t {tc:6.3f}  p {p:.4f}  ok")
    print("  t-distribution verified against published critical values")


def read_arm(root: str, arm: str, metric: str, at_step: int = 0):
    """seed -> (final milestone value, best milestone value, n milestones, last step).

    ``at_step`` truncates each run to its milestones at or below that step, which is how two
    runs of unequal length are made comparable.
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(root, f"{arm}_s[0-9]*"))):
        base = os.path.basename(d)
        seed = base.rsplit("_s", 1)[1]
        if base.rsplit("_s", 1)[0] != arm or not seed.isdigit():
            continue                                   # `depth` must not absorb `depth_ext`
        log = os.path.join(d, "train.jsonl")
        if not os.path.exists(log):
            continue
        vals = []
        for line in open(log):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                               # a torn last line on a live run
            v = r.get(metric)
            if v is not None and math.isfinite(v):
                step = r.get("step", 0)
                if not at_step or step <= at_step:
                    vals.append((step, v))
        if vals:
            out[int(seed)] = (vals[-1][1], min(v for _, v in vals), len(vals), vals[-1][0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--root", default=f"{RUNS}/py")
    ap.add_argument("--a", help="the treatment arm")
    ap.add_argument("--b", help="the arm it is measured against")
    ap.add_argument("--metric", default="val_nats_tok")
    ap.add_argument("--at", choices=("final", "best", "both"), default="both")
    ap.add_argument("--at-step", type=int, default=0,
                    help="compare every run at its last milestone <= this step. Required when "
                         "the runs have not all reached the same step.")
    ap.add_argument("--require-step", type=int, default=0,
                    help="EXCLUDE any seed whose run has not reached this step. The usual way to "
                         "read a comparison while more seeds are still training: it reports the "
                         "finished n rather than mixing finished and partial runs.")
    ap.add_argument("--allow-ragged", action="store_true",
                    help="compare runs stopped at different steps anyway. Almost always wrong.")
    ap.add_argument("--self-test", action="store_true")
    g = ap.parse_args()
    if g.self_test:
        self_test()
        return
    if not (g.a and g.b):
        ap.error("--a and --b are required (or --self-test)")

    A = read_arm(g.root, g.a, g.metric, g.at_step)
    B = read_arm(g.root, g.b, g.metric, g.at_step)
    if not A:
        raise SystemExit(f"no runs matched {g.a}_s* under {g.root}")
    if not B:
        raise SystemExit(f"no runs matched {g.b}_s* under {g.root}")
    if g.require_step:
        drop = {s for s, d in list(A.items()) + list(B.items()) if d[3] < g.require_step}
        if drop:
            print(f"EXCLUDED seeds not yet at step {g.require_step}: {sorted(drop)}")
        A = {s: d for s, d in A.items() if s not in drop}
        B = {s: d for s, d in B.items() if s not in drop}
        if not (A and B):
            raise SystemExit(f"no seed of both arms has reached step {g.require_step}")
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit(f"no shared seeds: {g.a} has {sorted(A)}, {g.b} has {sorted(B)}")
    only = (set(A) ^ set(B))
    if only:
        print(f"NOTE: seeds present in only one arm are EXCLUDED: {sorted(only)}\n")

    # RAGGED RUNS (2026-09-30). A run still training has a last milestone in the MIDDLE of its
    # schedule, and its loss there is far worse than any finished run's. Pairing the two reports
    # a difference dominated by how far along each run is, on a scale ~200x the real effect: this
    # guard was added after such a comparison printed a mean of +1.03 with a spread of 3.24 where
    # the effects being measured are ~0.005. The numbers looked like a result, not like a bug.
    steps = {(arm, s): d[3] for arm, D in ((g.a, A), (g.b, B)) for s, d in D.items()}
    last = {steps[(arm, s)] for arm in (g.a, g.b) for s in shared}
    if len(last) > 1 and not (g.at_step or g.allow_ragged):
        common = min(last)
        print("REFUSED: these runs are at different steps, so their final milestones are not "
              "comparable.\n  last step per run:")
        for arm in (g.a, g.b):
            for s in shared:
                print(f"    {arm}_s{s}: {steps[(arm, s)]}")
        raise SystemExit(f"\n  Re-run with --at-step {common} to compare every run at the "
                         f"milestone they all reached, or --allow-ragged to override.")
    if g.at_step:
        print(f"(every run truncated to its last milestone at or below step {g.at_step})\n")

    for which, idx in (("final", 0), ("best", 1)):
        if g.at != "both" and g.at != which:
            continue
        print(f"=== {g.a} vs {g.b}   ({g.metric}, at each run's {which} milestone) ===")
        diffs = []
        for s in shared:
            a, b = A[s][idx], B[s][idx]
            diffs.append(a - b)
            print(f"  seed {s}: {g.a} {a:.4f}   {g.b} {b:.4f}   diff {a - b:+.4f}"
                  f"{'  <-- ' + g.a + ' better' if a < b else ''}")
        n = len(diffs)
        mean = st.mean(diffs)
        won = sum(1 for d in diffs if d < 0)
        if n < 2:
            print(f"  mean {mean:+.4f} on n=1 -- no spread, no t\n")
            continue
        sd = st.stdev(diffs)
        se = sd / math.sqrt(n)
        t = mean / se if se else float("inf")
        p = p_two_sided(t, n - 1) if se else 0.0
        print(f"  mean paired diff {mean:+.4f}   sd {sd:.4f}   se {se:.4f}")
        print(f"  t = {t:+.2f} on {n - 1} df   p = {p:.4f}   "
              f"{g.a} better on {won}/{n} seeds")
        verdict = ("SEPARATED" if p < 0.05 and won in (0, n) else
                   "separated on the mean but MIXED SIGN" if p < 0.05 else
                   "not separable")
        print(f"  -> {verdict}\n")


if __name__ == "__main__":
    main()
