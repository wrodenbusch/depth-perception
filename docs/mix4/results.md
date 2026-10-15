# The four-corpus cell: web + Python + spreadsheets + chat in one model

Seven arms, **ten** paired seeds, one model trained on four corpora at once. The two-corpus cell
(`docs/mixed/results.md`) showed that one model can hold two corpora wanting different coordinates,
and that overloading one plane group with two meanings costs nothing. This is the harder version:
**can one shared depth coordinate serve four domains whose trees are not the same shape?**

Perplexity reduction is `100 (1 - e^Δ)`, positive = better. Numbers below are the **test** split
unless marked; validation agrees on 76 of the 80 comparisons and the four sign flips are all far
from significance.

## The cell

| | |
|---|---|
| pack | `$TOPOS_PACKS/mix4_b4096`, 452,506 records, 150.5M tokens |
| balance | **equal tokens per corpus**: 34.70M each, within 0.005%, capped by the smallest train pool (spreadsheets) |
| val / test | 2,048 records from **each** corpus's own eval range, per split — 8,192 each |
| budget | 104 records × 9,060 steps ≈ 300M tokens = 2.16 passes over 138.8M unique train tokens |
| model | 6 layers / 2 heads / 256 embd, 4.7M non-embedding, block 4,096, LR 1.2e-3 |
| vocabulary | **83,072** = GPT-2's 50,304 ∪ Python's 32,768 shifted by +50,304 |
| chat tree | the **prefix-causal** Markdown tree (`causal_chat_pack.py`), so chat's coordinate is one a generator can supply |

The learning rate is the web cell's, inherited unchanged so the seven arms match. It was not
re-smoked for this mix; that is an assumption, not a measurement.

### Why the balance is by token

Record length differs by a factor of ten across these corpora, so balancing by record and balancing
by token are not nearly the same thing:

| corpus | train records | share of records | mean tokens/record | share of tokens |
|---|---|---|---|---|
| web | 34,866 | 8.0% | 995.2 | 25.00% |
| Python | 319,987 | 73.4% | 108.4 | 25.00% |
| spreadsheets | 32,997 | 7.6% | 1,051.6 | 25.00% |
| chat | 48,272 | 11.1% | 718.9 | 25.00% |

An AST-compact Python file is a ninth the size of a spreadsheet. Balancing by record would have
left Python contributing about 2.5% of the loss it is meant to contribute a quarter of.

### The union vocabulary

Three of the four corpora are GPT-2. Python is not — it is a 32,768-token AST-compact BPE whose
ids collide with GPT-2's. Every Python id is shifted by +50,304, so the union is 83,072, which is
exactly 1,298 × 64 and needs no padding. **The two halves share no token**, so whatever transfers
between Python and the rest transfers through the shared transformer body and the shared
coordinate, not through shared words. That makes the depth question cleaner, not muddier.

### The four depth distributions

This is what the headline arm is betting on (`mix4_depth_dist.py`, counting present levels exactly
as `rope_depth` does):

| corpus | mean depth | median | 90th pct | deepest |
|---|---|---|---|---|
| web (DOM) | 10.25 | 10 | 15 | 22 |
| spreadsheets (cell/formula) | 6.07 | 5 | 11 | 22 |
| Python (AST) | 4.42 | 4 | 7 | 22 |
| chat (Markdown) | 3.09 | 3 | 4 | 12 |

A typical web node sits more than three times deeper than a typical chat node, and the four
distributions have to share **one** plane pair with **one** learned step size.

## Arms

| arm | planes | what each corpus carries |
|---|---|---|
| `seq` | 16 seq | nothing — the baseline |
| `depth` | +2 | depth, **one shared plane pair for all four**. The headline arm. |
| `depth_percorpus` | +8 | depth, on four **disjoint** pairs — corpus *c* writes level *c*, the rest ABSENT |
| `depth_grid` | +8 | depth for all four on level 3; grid on levels 0–2 for the two corpora that have one |
| `depth_rel` | +2 | depth ÷ the deepest level so far in the record — a relative depth, no corpus label, prefix-computable |
| `depth_scaled` | +2 | depth × a per-corpus constant that aligns the four 90th percentiles |
| `both` | +44 | the full 22-level path, the expensive reference |

`depth` and `depth_percorpus` carry the **same coordinate**; they differ only in whether the four
corpora share a plane group. That pair is the experiment. `depth_rel` and `depth_scaled` ask
whether the disagreement between the four depth scales is worth correcting.

Two asymmetries, both stated because both favour the arm this cell is trying to beat:

- `depth` runs under `rope_depth`, which has one learned `step_rel` for all four corpora.
  `depth_percorpus` runs under `rope_both`, where **each level has its own learnable theta and
  step**. The per-corpus arm therefore has strictly more freedom. If sharing ties with it, that is
  evidence sharing is free, not an artefact of a hobbled comparison.
- The chat grid here is `[exchange, speaker]` only, weaker than the published chat cell's, which
  also uses `tok_within` on the token planes. Rotating those would also rotate web's
  within-text-node index — a known sequence surrogate — so the two could not share the pack.

## The headline, before the tables

**One depth coordinate, shared by four corpora that disagree about depth by a factor of 2.5, works
— and four private coordinates costing four times the planes buy nothing measurable over it.**
The shared coordinate settles almost exactly at the geometric mean of what the four would each have
chosen alone. Adding a grid for the two corpora that have one, eight planes in all, **ties the
44-plane path over the whole mixture**.

At ten seeds this is no longer only directional: **26 of the 80 comparisons survive
Benjamini–Hochberg at q < 0.05**, including the headline `depth vs seq` on web, on Python and over
all four corpora together. What ten seeds did *not* turn into a result is spreadsheets' and chat's
response to depth alone, and the sharing contrast, which remain bounded rather than resolved.

## Result 1 — one shared depth coordinate helps web, Python and the mixture

`depth` (2 shared planes) against `seq`, per corpus, test split, paired over 10 seeds:

| corpus | perplexity | 95% CI | seeds | p |
|---|---|---|---|---|
| web | **+1.48%** | [+0.99, +1.96] | 10/10 | 7.48e-05 |
| Python | **+0.86%** | [+0.25, +1.47] | 8/10 | 0.011 |
| spreadsheets | +0.40% | [-0.11, +0.91] | 8/10 | 0.108 |
| chat | +0.40% | [-0.03, +0.83] | 8/10 | 0.0656 |
| *every corpus* | *+0.80%* | *[+0.40, +1.19]* | *10/10* | *0.00145* |

Web and Python gain on every seed or nearly so. Spreadsheets and chat each land near +0.4% with an
interval that just includes zero — the direction is consistent (8/10 seeds each) but neither
separates. That split is the point of scoring per corpus: in the single-corpus cells depth is the
best web coordinate and spreadsheets prefer their grid, and the four-way mixture reproduces each
corpus's own preference rather than averaging them away.

## Result 2 — eight planes tie the 44-plane path, and the grid is why

`depth_grid` (depth for all four, plus a grid on levels 0–2 for web and spreadsheets) against `seq`:

| corpus | perplexity | 95% CI | seeds | p |
|---|---|---|---|---|
| web | **+1.41%** | [+0.78, +2.05] | 9/10 | 0.000747 |
| Python | **+0.97%** | [+0.58, +1.36] | 9/10 | 0.000348 |
| spreadsheets | **+1.51%** | [+1.05, +1.96] | 10/10 | 3.93e-05 |
| chat | +0.39% | [-0.25, +1.04] | 7/10 | 0.203 |
| *every corpus* | *+1.18%* | *[+0.70, +1.66]* | *9/10* | *0.000384* |

Against the full path:

| corpus | perplexity | 95% CI | seeds | p |
|---|---|---|---|---|
| web | +0.14% | [-0.51, +0.79] | 5/10 | 0.638 |
| Python | -0.39% | [-0.97, +0.20] | 3/10 | 0.169 |
| spreadsheets | +0.01% | [-0.50, +0.51] | 5/10 | 0.98 |
| chat | -0.16% | [-0.96, +0.64] | 4/10 | 0.666 |
| *every corpus* | *-0.00%* | *[-0.60, +0.59]* | *4/10* | *0.995* |

A dead heat over the mixture (-0.00%,
4/10, p = 0.99)
and on every corpus separately. Where the grid earns its planes is spreadsheets, against shared
depth alone:

| corpus | perplexity | 95% CI | seeds | p |
|---|---|---|---|---|
| web | -0.07% | [-0.82, +0.68] | 6/10 | 0.845 |
| Python | +0.11% | [-0.51, +0.73] | 6/10 | 0.698 |
| spreadsheets | **+1.11%** | [+0.68, +1.54] | 9/10 | 0.000242 |
| chat | -0.01% | [-0.87, +0.85] | 7/10 | 0.988 |
| *every corpus* | *+0.39%* | *[-0.20, +0.98]* | *9/10* | *0.172* |

Depth alone, by contrast, does **not** tie the full path over the mixture
(-0.39%,
1/10, p = 0.033),
and the deficit is entirely spreadsheets (-1.12%,
1/10) and Python
(-0.50%, 2/10);
on web two planes of depth are if anything ahead of all 44
(+0.21%, 6/10).

## Result 3 — sharing one plane pair costs nothing measurable; the bound is now tighter

`depth_percorpus` gives each corpus its **own** depth plane pair: 8 planes, four disjoint groups,
each with its own learnable theta and step. `depth` gives all four **one** pair with **one** step.
Same coordinate, four times the planes, strictly more freedom.

| corpus | perplexity | 95% CI | seeds | p |
|---|---|---|---|---|
| web | +0.66% | [-0.21, +1.53] | 7/10 | 0.12 |
| Python | +0.18% | [-0.51, +0.87] | 5/10 | 0.571 |
| spreadsheets | +0.27% | [-0.22, +0.75] | 7/10 | 0.251 |
| chat | +0.44% | [-0.48, +1.35] | 7/10 | 0.311 |
| *every corpus* | *+0.44%* | *[-0.26, +1.14]* | *7/10* | *0.186* |

Every interval spans zero. **Read as an upper bound, not as proof of equivalence** — and note that
the private arm is numerically ahead on all four corpora, so the honest statement is that private
planes are not better than one shared pair *by more than about 0.8–1.5% depending on corpus*,
rather than that they are no better at all. Ten seeds tightened the bound from the five-seed
version (web, for instance, from ±2.9% to ±1.5%) without changing the verdict.

The asymmetry runs the right way for the claim: the per-corpus arm has more parameters and more
freedom, so a tie is evidence that sharing is cheap rather than an artefact of a hobbled control.

## Result 4 — the shared step lands at the midpoint of what the four would have chosen

The per-corpus arm's four steps are what each corpus picks when free; the shared arm's single step
is what it picks when it cannot. Mean over 10 seeds, first plane of each pair:

| corpus | own step | median tree depth |
|---|---|---|
| web | 1.566 | 10 |
| chat | 1.073 | 3 |
| spreadsheets | 0.728 | 5 |
| Python | 0.635 | 4 |
| **shared, all four** | **0.915** | — |

The four span **2.47×**. Their geometric mean — the right
midpoint for a frequency, where halving and doubling are the symmetric moves — is 0.939, and the
shared coordinate sits at 0.915, **2.5% away from it**. The model lands on a
compromise rather than latching onto one corpus's preference and letting the others wear it, and
Result 3 says that compromise is not measurably costly.

What this does **not** show is any tidy relation between a corpus's tree depth and the step it
wants. Web is deepest and wants the fastest rotation, but chat is shallowest and wants the second
fastest, which breaks the monotone story. Four points, one of them out of order; no claim.

## Result 5 — rescaling depth per corpus makes it worse, not better

If the problem were that four depth scales disagree, normalising them should help. Two ways of
doing it, both against plain shared `depth`:

**Relative depth** (`depth_rel`): the level divided by the deepest level reached so far *in the
prefix*, so it needs no corpus label and stays generation-valid.

| corpus | perplexity | 95% CI | seeds | p |
|---|---|---|---|---|
| web | -0.01% | [-0.72, +0.70] | 5/10 | 0.984 |
| Python | **-1.17%** | [-1.92, -0.42] | 1/10 | 0.00633 |
| spreadsheets | -0.07% | [-0.42, +0.29] | 4/10 | 0.681 |
| chat | +0.07% | [-0.55, +0.69] | 7/10 | 0.797 |
| *every corpus* | *-0.05%* | *[-0.55, +0.44]* | *5/10* | *0.817* |

**Per-corpus constant** (`depth_scaled`): multiply each corpus's depth by a fixed factor that lines
up the four 90th percentiles. Aligning the medians instead would have clamped 9.9% of spreadsheet
tokens at level 22 against 0.64% for the p90 basis, so p90 is the basis used.

| corpus | perplexity | 95% CI | seeds | p |
|---|---|---|---|---|
| web | -0.19% | [-0.92, +0.53] | 4/10 | 0.564 |
| Python | -0.28% | [-0.78, +0.21] | 3/10 | 0.227 |
| spreadsheets | -0.11% | [-0.37, +0.14] | 3/10 | 0.342 |
| chat | -0.05% | [-0.81, +0.71] | 4/10 | 0.893 |
| *every corpus* | *-0.13%* | *[-0.65, +0.38]* | *3/10* | *0.578* |

Neither helps anywhere, and relative depth **costs** on Python, which is the one clear result in
this pair (-1.17%,
1/10, q < 0.05 in the cell's own family). Both also
lose to the per-corpus arm. One reason relative depth has little to give: it is exactly 1.0 for 80%
of chat tokens and 20% of web's (`mix4_rescale_probe.py`), so on the shallowest corpus it is
nearly constant.

This is a negative result that strengthens the main claim. The raw nesting level, shared across
four corpora with one learned step, beats every normalisation of it we tried.

## Honest accounting

**Multiplicity.** 80 comparisons, Benjamini–Hochberg: **26 survive at q < 0.05**. The
surviving set includes `depth vs seq` on web, on Python and over all four corpora; `depth_grid vs
seq` on web, Python, spreadsheets and the mixture; `both vs seq` everywhere but chat; and
`depth_rel vs depth` on Python. The contrasts that do **not** survive are the ones the prose above
reports as bounds: `depth vs seq` on spreadsheets and chat, and every sharing contrast.

**Two runs carried an allocator setting.** Two of the seventy runs fragmented GPU memory and needed
`roundup_power2_divisions:8`; the rest ran on defaults. Verified numerically inert against a
measured noise floor (`mix4_alloc_inert.py`): across replicates, pairs sharing the setting differ
from each other *more* than defaults-vs-setting pairs do (ratio 1.05). See `make_mix4_jobs.py` for
the two settings that looked right and were not.

**Runs are not bit-reproducible.** Two runs of one arm at one seed diverge from step 1 — step 0 is
identical to the last digit, so initialisation and data order are pinned by the seed, but the
backward pass accumulates nondeterministically. Measured on web pages (`reprod_spread.py`), run-to-run
spread in final validation loss is 0.0023 nats against 0.0109 seed-to-seed, 21% of the spread the
pairing removes; it is already inside the seed-to-seed spread the t-tests use, so the p-values are
unaffected.

**Val and test agree.** 80 comparisons on both splits; 4 flip sign between them; the flips are all far from significance. Not independent evidence
— same runs — but a comparison that flipped on a headline would be a warning. None did.

**Learning rate was inherited, not re-smoked** for this mixture, identically across arms.

## Reproducing

```
.venv/bin/python build_mix4_pack.py        # pack + the derived packs
.venv/bin/python build_mix4_rescale_packs.py   # depth_rel and depth_scaled
.venv/bin/python verify_mix4_pack.py       # dtype, vocabulary disjointness, split membership
.venv/bin/python verify_mix4_rescale.py    # the two rescaled coordinates, incl. prefix-validity
.venv/bin/python mix4_depth_dist.py        # the four depth distributions
.venv/bin/python mix4_rescale_probe.py     # scale factors, clamp rates, relative-depth saturation
.venv/bin/python make_mix4_configs.py      # every config from one template
./mix4_smoke.sh 1                          # 2 steps of every arm before the runs queue
.venv/bin/python make_mix4_jobs.py
.venv/bin/python runqueue.py $TOPOS_RUNS/mix4/jobs.json --cards 0,1,2,3 --max 4
.venv/bin/python mix4_verify_done.py       # every 'done' run really reached step 9060
.venv/bin/python mix_eval.py --cell mix4 --split test
.venv/bin/python mix4_fdr.py test          # q-values across the 80 comparisons
.venv/bin/python mix4_ci.py test           # intervals and detectable effects
.venv/bin/python mix4_steps.py             # the learned step sizes
.venv/bin/python mix4_val_test_agree.py    # do the splits say the same thing
```

Scores land in `$TOPOS_RUNS/analysis/mix4/mix4_{val,test}.json`.
