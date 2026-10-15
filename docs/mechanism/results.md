# Mechanism analyses

## 1. Removing the structure at test time

**Python** (test split, first 4096 records, seeds [0, 1, 2, 3, 4])

| configuration | gain over sequence only | cost of removing structure | cost of shuffling structure | removed, vs sequence only |
|---|---|---|---|---|
| sequence + depth | 0.49% better (3/5) | +445.7% (5/5 seeds, p = 0.00014) | +653.1% (5/5 seeds, p = 3.7e-05) | +443.0% (5/5 seeds, p = 0.00013) |
| sequence + tree | 0.41% better (3/5) | +127.7% (5/5 seeds, p = 0.00018) | +436.6% (5/5 seeds, p = 1.6e-05) | +126.8% (5/5 seeds, p = 0.00018) |
| sequence + tree + depth | 0.82% better (4/5) | +500.9% (5/5 seeds, p = 2e-06) | +1390.9% (5/5 seeds, p = 1.3e-07) | +495.9% (5/5 seeds, p = 2.3e-06) |

**Spreadsheets** (test split, first 512 records, seeds [0, 1, 2, 3, 4])

| configuration | gain over sequence only | cost of removing structure | cost of shuffling structure | removed, vs sequence only |
|---|---|---|---|---|
| sequence + depth | 0.74% better (4/5) | +6.8% (5/5 seeds, p = 0.0039) | +13.4% (5/5 seeds, p = 0.00017) | +6.0% (5/5 seeds, p = 0.0096) |
| sequence + grid | 1.37% better (5/5) | +11.9% (5/5 seeds, p = 0.0003) | +29.5% (5/5 seeds, p = 2.8e-06) | +10.4% (5/5 seeds, p = 0.00072) |
| sequence + tree | 1.27% better (5/5) | +17.2% (5/5 seeds, p = 0.00025) | +42.3% (5/5 seeds, p = 2e-06) | +15.7% (5/5 seeds, p = 0.00038) |
| sequence + tree + depth | 1.24% better (4/5) | +25.1% (5/5 seeds, p = 3.1e-05) | +58.1% (5/5 seeds, p = 2.4e-05) | +23.6% (5/5 seeds, p = 3.5e-05) |

**web pages** (test split, first 512 records, seeds [0, 1, 2, 3, 4])

| configuration | gain over sequence only | cost of removing structure | cost of shuffling structure | removed, vs sequence only |
|---|---|---|---|---|
| sequence + depth | 2.05% better (5/5) | +5.4% (5/5 seeds, p = 4.9e-06) | +17.6% (5/5 seeds, p = 3.4e-05) | +3.2% (5/5 seeds, p = 0.00075) |
| sequence + tree | 1.12% better (5/5) | +9.7% (5/5 seeds, p = 1.4e-05) | +25.9% (5/5 seeds, p = 3.4e-05) | +8.5% (5/5 seeds, p = 3.9e-05) |
| sequence + tree + depth | 2.00% better (5/5) | +10.4% (5/5 seeds, p = 1.8e-06) | +30.4% (5/5 seeds, p = 5e-06) | +8.2% (5/5 seeds, p = 2.5e-06) |

## 2-3. Attention on structurally related keys, and how much the rotation moves scores

**Python, sequence + depth vs sequence only** (mean over all heads; seeds = 5)

| keys in relation | share under uniform attention | sequence only | structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |
|---|---|---|---|---|---|---|---|
| same node | 8.5% | 9.3% | 9.1% | -0.1 | 2/5 | 0.29 | -1.2 |
| sibling node | 8.5% | 11.7% | 11.9% | +0.2 | 3/5 | 0.25 | +1.9 |
| same depth, elsewhere | 14.9% | 13.1% | 13.2% | +0.2 | 4/5 | 0.076 | +2.5 |
| ancestor | 11.3% | 20.0% | 20.6% | +0.5 | 4/5 | 0.062 | +3.7 |

Switching the structure rotation on changes a head's attention scores by 0.21-0.52 (10th-90th percentile, median 0.32) of the spread of those scores, over all 60 heads x seeds (std of the change / std of the score, on 20,000 sampled query-key pairs per record).

**Python, sequence + tree + depth vs sequence only** (mean over all heads; seeds = 5)

| keys in relation | share under uniform attention | sequence only | structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |
|---|---|---|---|---|---|---|---|
| same node | 8.5% | 9.3% | 9.2% | -0.1 | 1/5 | 0.16 | -1.8 |
| sibling node | 8.5% | 11.7% | 11.7% | -0.0 | 2/5 | 0.88 | -2.3 |
| same depth, elsewhere | 14.9% | 13.1% | 13.3% | +0.2 | 4/5 | 0.29 | -1.6 |
| ancestor | 11.3% | 20.0% | 20.3% | +0.3 | 4/5 | 0.26 | +3.5 |

Switching the structure rotation on changes a head's attention scores by 0.33-0.65 (10th-90th percentile, median 0.45) of the spread of those scores, over all 60 heads x seeds (std of the change / std of the score, on 20,000 sampled query-key pairs per record).

**Spreadsheets, sequence + depth vs sequence only** (mean over all heads; seeds = 5)

| keys in relation | share under uniform attention | sequence only | structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |
|---|---|---|---|---|---|---|---|
| same row | 20.7% | 51.8% | 52.5% | +0.6 | 2/5 | 0.65 | +36.6 |
| same column, earlier row | 23.9% | 14.2% | 13.8% | -0.4 | 3/5 | 0.62 | -18.4 |
| the referenced cell | 3.2% | 4.1% | 4.3% | +0.2 | 4/5 | 0.21 | +4.7 |

Switching the structure rotation on changes a head's attention scores by 0.02-0.42 (10th-90th percentile, median 0.19) of the spread of those scores, over all 60 heads x seeds (std of the change / std of the score, on 20,000 sampled query-key pairs per record).

**Spreadsheets, sequence + grid vs sequence only** (mean over all heads; seeds = 5)

| keys in relation | share under uniform attention | sequence only | structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |
|---|---|---|---|---|---|---|---|
| same row | 20.7% | 51.8% | 50.5% | -1.3 | 1/5 | 0.25 | -25.0 |
| same column, earlier row | 23.9% | 14.2% | 18.5% | +4.3 | 5/5 | 0.018 | +32.4 |
| the referenced cell | 3.2% | 4.1% | 4.6% | +0.5 | 5/5 | 0.014 | +3.4 |

Switching the structure rotation on changes a head's attention scores by 0.05-0.81 (10th-90th percentile, median 0.27) of the spread of those scores, over all 60 heads x seeds (std of the change / std of the score, on 20,000 sampled query-key pairs per record).

**Spreadsheets, sequence + tree + depth vs sequence only** (mean over all heads; seeds = 5)

| keys in relation | share under uniform attention | sequence only | structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |
|---|---|---|---|---|---|---|---|
| same row | 20.7% | 51.8% | 50.8% | -1.1 | 3/5 | 0.57 | +25.5 |
| same column, earlier row | 23.9% | 14.2% | 20.3% | +6.1 | 5/5 | 0.0058 | +33.3 |
| the referenced cell | 3.2% | 4.1% | 4.8% | +0.7 | 5/5 | 0.013 | +4.0 |

Switching the structure rotation on changes a head's attention scores by 0.05-1.05 (10th-90th percentile, median 0.44) of the spread of those scores, over all 60 heads x seeds (std of the change / std of the score, on 20,000 sampled query-key pairs per record).

**web pages, sequence + depth vs sequence only** (mean over all heads; seeds = 5)

| keys in relation | share under uniform attention | sequence only | structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |
|---|---|---|---|---|---|---|---|
| same node | 10.8% | 37.9% | 37.1% | -0.8 | 3/5 | 0.41 | -16.5 |
| sibling node | 5.8% | 5.0% | 5.1% | +0.1 | 5/5 | 0.0016 | +1.8 |
| same depth, elsewhere | 32.8% | 22.4% | 24.2% | +1.8 | 5/5 | 0.00019 | +14.5 |
| other branch | 36.9% | 10.2% | 9.9% | -0.2 | 1/5 | 0.32 | -2.8 |

Switching the structure rotation on changes a head's attention scores by 0.06-0.42 (10th-90th percentile, median 0.18) of the spread of those scores, over all 60 heads x seeds (std of the change / std of the score, on 20,000 sampled query-key pairs per record).

**web pages, sequence + tree + depth vs sequence only** (mean over all heads; seeds = 5)

| keys in relation | share under uniform attention | sequence only | structured | change (pp) | seeds up | p | most-moved head per seed, median (pp) |
|---|---|---|---|---|---|---|---|
| same node | 10.8% | 37.9% | 37.3% | -0.6 | 1/5 | 0.061 | -5.0 |
| sibling node | 5.8% | 5.0% | 5.0% | +0.0 | 3/5 | 0.49 | -0.3 |
| same depth, elsewhere | 32.8% | 22.4% | 24.0% | +1.6 | 5/5 | 0.0048 | +16.7 |
| other branch | 36.9% | 10.2% | 9.3% | -0.9 | 1/5 | 0.039 | -10.5 |

Switching the structure rotation on changes a head's attention scores by 0.07-0.52 (10th-90th percentile, median 0.23) of the spread of those scores, over all 60 heads x seeds (std of the change / std of the score, on 20,000 sampled query-key pairs per record).

## 4. Learned rotation steps (shared by every head and layer)

| setting | configuration | learned steps, fast / slow plane (init 0.316 / 0.020 rad) |
|---|---|---|
| Python | sequence + depth | depth 0.63 / 0.36 (n=5) |
| Python | sequence + tree | level 1 0.89 / 0.164; level 2 0.66 / 0.129; level 3 0.62 / 0.176 (n=5) |
| Python | sequence + tree + depth | level 1 0.86 / 0.158; level 2 0.66 / 0.161; level 3 0.68 / 0.258; depth 0.63 / 0.48 (n=5) |
| Spreadsheets | sequence + depth | depth 0.74 / 0.25 (n=5) |
| Spreadsheets | sequence + grid | sheet 0.66 / 0.039; row 0.49 / 0.112; column 0.91 / 0.201 (n=5) |
| Spreadsheets | sequence + tree | sheet 0.61 / -0.005; row 0.46 / 0.106; column 0.90 / 0.202 (n=5) |
| Spreadsheets | sequence + tree + depth | sheet 0.60 / 0.075; row 0.40 / 0.108; column 0.91 / 0.201; depth 0.74 / 0.22 (n=5) |
| web pages | sequence + depth | depth 1.43 / 0.35 (n=5) |
| web pages | sequence + tree | level 1 0.42 / -0.000; level 2 1.17 / 0.001; level 3 1.32 / 0.000 (n=5) |
| web pages | sequence + tree + depth | level 1 0.39 / -0.000; level 2 1.12 / 0.000; level 3 1.11 / 0.000; depth 1.59 / 0.41 (n=5) |
