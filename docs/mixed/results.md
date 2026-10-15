# The mixed cell: web text + spreadsheets in one model

Five arms, five paired seeds, one model trained on both corpora at once. Scored overall and per
corpus on the held-out **test** split (validation agrees everywhere within 0.001 nats; both are in
`$TOPOS_RUNS/analysis/mix/`). Perplexity reduction is `100 (1 - e^Δ)`, positive = better.

## The cell

| | |
|---|---|
| pack | `$TOPOS_PACKS/mix_web_sheets_b4096`, 74,186 records, 75.8M tokens |
| balance | 32,997 train records from each corpus — web capped to the spreadsheet count. 50/50 by record, 48.6/51.4 by token |
| val / test | 2,048 records from **each** corpus's own eval range, per split |
| budget | 32 records × 9,160 steps ≈ 300M tokens = 4.44 passes over 67.6M unique train tokens |
| model | 6 layers / 2 heads / 256 embd, 4.7M non-embedding, block 4,096, LR 1.2e-3 |

The learning rate is the web cell's, inherited unchanged so the five arms match. It was not
re-smoked for this mix; that is an assumption, not a measurement.

## Arms

| arm | planes | web records carry | spreadsheet records carry |
|---|---|---|---|
| `seq` | 16 seq | — | — |
| `depth` | +2 | depth (DOM) | depth (cell/formula tree) |
| `reuse` | +6 | depth, in level 0 of one shared group | sheet, row, column |
| `separate` | +8 | depth, on its own level 3 | sheet, row, column on levels 0–2 |
| `grid` | +6 | nothing (all ABSENT) | sheet, row, column |

`reuse` and `separate` carry the **same coordinates**; they differ only in whether the two corpora
share a plane group. That pair is the experiment.

## Result 1 — one model reproduces each corpus's preference

Per corpus, against the sequence-only baseline **in the same model**:

| arm | web (test) | spreadsheets (test) |
|---|---|---|
| `depth` | **+2.86%** (5/5, p = 0.0014) | +0.79% (5/5, p = 0.021) |
| `grid` | −0.26% (2/5, n.s.) | **+1.35%** (5/5, p = 0.011) |
| `reuse` | +1.52% (5/5, p = 0.018) | +1.34% (5/5, p = 0.0068) |
| `separate` | +1.59% (5/5, p = 0.0044) | +1.32% (5/5, p = 0.0014) |

The single-corpus ordering survives. Depth is the best web coordinate and the grid does nothing
there (−0.26%, n.s.); the grid is the best spreadsheet coordinate. Head to head inside the mixed
model, depth beats the grid on web by **3.22%** (0/5 seeds for the grid, p = 0.0042), and the grid
beats depth on spreadsheets by 0.56% (4/5, p = 0.14, suggestive).

Qualitatively this matches the single-corpus cells, which found (at 10 seeds, their own packs and
budgets — **not** comparable in nats to the table above):

| | depth vs seq | grid vs seq |
|---|---|---|
| web pages alone | −0.0202, 10/10 | — |
| Spreadsheets alone | −0.0025, 7/10, n.s. | −0.0110, 10/10 |

One difference worth naming: depth is **significantly positive on spreadsheets here** (+0.79%,
5/5) where it was not in the single-corpus cell (7/10, p = 0.36). The mixed model has web pages
teaching it what a depth coordinate is for, and the spreadsheet half appears to benefit from that
— but this is a cross-cell comparison and the cells differ in budget, context and data mix, so it
is a hypothesis, not a result.

## Result 2 — sharing a plane group between corpora costs nothing measurable

`separate` (8 planes, disjoint groups) against `reuse` (6 planes, one overloaded group):

| | Δ nats | perplexity | seeds | p |
|---|---|---|---|---|
| both corpora | −0.0002 | +0.02% | 3/5 | 0.93 |
| web only | −0.0006 | +0.06% | 2/5 | 0.82 |
| spreadsheets only | +0.0002 | −0.02% | 3/5 | 0.88 |

A dead heat, on every split and both corpora. Giving each corpus its own plane group buys nothing
over letting them share one, even though in `reuse` nothing marks which meaning a rotation
carries and the two numeric ranges collide by design (web depth 1–22, sheet index 0–11). The model
evidently disambiguates from content.

The two extra planes `separate` spends are therefore not the point either way; the useful reading
is that **an overloaded group is safe**, which matters for a model that must hold many structure
types without a plane budget that grows with them.

## Result 3 — but the union of coordinates loses to depth alone, on the mixed average

| arm | both corpora (test) |
|---|---|
| `depth` (2 planes) | **+1.81%** (5/5, p = 0.0025) |
| `separate` (8 planes) | +1.45% (5/5, p = 0.0022) |
| `reuse` (6 planes) | +1.43% (5/5, p = 0.011) |
| `grid` (6 planes) | +0.57% (3/5, n.s.) |

Depth alone is the best arm overall, beating `separate` by 0.36% (2/5, p = 0.33) and `reuse` by
0.38% (2/5, p = 0.37) — neither separable, but both in the same direction, and depth does it with
a quarter of the planes.

**The mixed average is dominated by web**, which is why this does not contradict Result 1: web
records cost 3.65 nats/token against the spreadsheets' 0.84, so web supplies ~81% of the mixed
loss. An arm that is 2.86% better on web and 0.79% better on spreadsheets (`depth`) outscores one
that is 1.59% and 1.32% (`separate`) on the weighted average, even though the second is better on
the easier corpus. Any mixed-corpus number of this kind is a statement about the mix, not about
the method.

What `separate` and `reuse` give up is web performance: 1.59% against depth's 2.86%, 0/5 seeds
(p = 0.049). Both arms put web's depth in a group that spreadsheets also use, and on web that is
measurably worse than depth on its own dedicated pair — the only place in this cell where sharing
appears to cost something, and it does not show up in the `reuse` vs `separate` contrast because
*both* share.

## What this says

1. **One model holds both.** Each corpus keeps the coordinate preference it had alone; nothing
   about training on two structure types at once destroys either signal.
2. **Overloading a plane group is free.** Two corpora can write different meanings into the same
   planes with no measurable penalty against giving each its own.
3. **Depth remains the cheapest thing that works**, and on a web-dominated mixture it is also the
   best — but that ranking is a property of the mixture's weighting, and the per-corpus tables are
   the honest result.

## Caveats

- Five seeds; several contrasts are suggestive rather than separated.
- The learning rate was inherited from the web cell and not re-smoked for this mix.
- The mix is 50/50 by record, which is a choice. A web-weighted mix would favour depth further and
  a spreadsheet-weighted one the grid, by the weighting argument above.
- Cross-cell nats/token comparisons with the published single-corpus cells are not made; only the
  arm ordering is carried across, and it is stated qualitatively.

Raw per-seed scores: `$TOPOS_RUNS/analysis/mix/mix_{val,test}.json`.
