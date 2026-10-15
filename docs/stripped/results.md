# Stripped-text workarounds: results

## 1. Depth dropout (web pages, 10 seeds, first 1,024 validation pages)

| model, scored with | vs sequence only (ppl %, + = better) | seeds better | p |
|---|---|---|---|
| depth/true | +2.00% | 10/10 | 2.8e-05 |
| depth/nodepth | -3.41% | 0/10 | 2.9e-07 |
| depthdrop30/true | +1.49% | 9/10 | 0.0015 |
| depthdrop30/nodepth | -0.42% | 4/10 | 0.23 |
| depthdrop50/true | +1.23% | 9/10 | 0.0014 |
| depthdrop50/nodepth | -0.07% | 5/10 | 0.8 |
| depthdrop30/true_vs_depth | -0.51% | 1/10 | 0.032 |
| depthdrop50/true_vs_depth | -0.78% | 0/10 | 0.00063 |

Share of plain depth's reading gain kept, depthdrop30: 75%

Share of plain depth's reading gain kept, depthdrop50: 61%

## 2. Compact depth-change markers (1024 validation pages in both packs)

| comparison | Δ (ppl %, + = better) | seeds better | p |
|---|---|---|---|
| visible marked_seq vs seq | -0.27% | 6/10 | 0.42 |
| visible marked_depth vs seq | +0.43% | 7/10 | 0.24 |
| visible marked_depth vs marked_seq | +0.70% | 9/10 | 0.0015 |
| visible marked_depth vs depth | -1.56% | 1/10 | 0.0013 |
| visible depth vs seq | +1.96% | 10/10 | 2.8e-05 |
| page total marked_seq vs seq (nats/page) | +73.6208 | 0/10 | |
| marker loss marked_seq (nats/marker) | +1.3152 | / | |
| page total marked_depth vs seq (nats/page) | +65.1998 | 0/10 | |
| marker loss marked_depth (nats/marker) | +1.2819 | / | |

Generation check: depth from the markers in the prefix vs training depth: 0 mismatches in 1,016,270 tokens.
