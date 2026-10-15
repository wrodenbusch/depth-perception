# Understanding tests, 2026-10-02

Held-out loss split by the kind of token being predicted (`understanding_evals.py`, whose
docstring defines every category and what it does not measure). Final checkpoints, test split,
paired per seed against sequence only. Full tables: `py_results.md`, `sheets_results.md`,
`webc_results.md`; per-token scores under `$TOPOS_RUNS/analysis/understanding/`.

"beyond" = category Δ − all-token Δ, paired per seed: negative means the configuration helps this
kind of token MORE than its average gain.

## Findings (survive pairing, consistent sign across seeds)

**Spreadsheets — the tree path helps structural tokens far more than average; depth helps nothing.**

| category | sequence + tree | sequence + depth | sequence + tree + depth | reference rotation | layer split |
|---|---|---|---|---|---|
| all tokens | −1.5% (10/10) | −0.2% (6/10) | −1.5% (8/10) | −2.5% (10/10) | −2.7% (5/5) |
| row-ending newline | **−13.1%** (10/10), +4.9 pt acc | −0.7% (5/10) | **−13.1%** (10/10), +5.0 pt | −12.3% (10/10) | −13.6% (5/5) |
| cell reference in a formula | −8.1% (10/10) | −0.1% | −7.5% (10/10) | **−45.6%** (10/10), +2.5 pt | −10.7% (5/5) |
| copy down a column across ≥256 tokens | −5.0% (9/10) | −1.0% (6/10) | −4.6% (9/10) | −4.5% (10/10) | **−11.1%** (5/5), +3.6 pt |

Why depth does nothing on spreadsheets, measured: in this render every plain value sits at the same
depth (sheet → row → column → content), the separators carry no coordinate, and formula nesting is
already spelled out by parentheses. Depth carries no information the text lacks; the row and column
are what matter, and only the path carries them.

**Web text — structure helps the first token of a new block 2–3× more than average.**

| category | sequence + tree | sequence + depth | sequence + tree + depth |
|---|---|---|---|
| all tokens | −0.4% (10/10) | −0.6% (10/10) | −0.6% (10/10) |
| first token of a new text node | −1.0% (10/10), beyond p<1e-6 | −1.0% (10/10), beyond p<1e-5 | **−1.3%** (10/10), beyond p<1e-6 |
| repeat of a word last seen in a different branch (64–1023 tokens back) | −1.2% (9/10) | −1.8% (9/10) | −1.5% (9/10) |

**Python — mostly spread evenly; two category effects survive.** For the configurations that keep
sequence position, the gain is spread roughly evenly over subtree closes, identifier reuse and new
bindings, and no "beyond" survives at n=5. Two effects do: tree only (n=20) helps tokens that open a
child node more than average (−2.3% vs −1.2% overall, beyond p=0.0002); and no position is hurt
most exactly where a subtree closes (+4.6% on those tokens, beyond p=0.008) — without any position
signal, counting announced children is what breaks.

## Caveat that bears on the paper

On the Python TEST split (first 8,192 records) the gains are about half the validation-split gains
and less consistent: sequence + tree −0.0042 (3/5 seeds) against −0.0095 (5/5) on validation;
sequence + tree + depth −0.0075 (4/5) against −0.0113. The scorer reproduces train.jsonl's
validation numbers to 1e-4 on the 1,024 validation records, so this is a split difference, not a
pipeline error: the test records have higher loss (0.43 vs 0.35 nats/token). Spreadsheets and web pages
reproduce their validation gains on test.

## Not run, and why

Structural minimal pairs (shuffled sections, swapped statements). Perturbing text while keeping
coordinates consistent either tells the structured model the new structure (incomparable surprise)
or feeds it contradictory input it never trained on. Either way it measures the input contract.
