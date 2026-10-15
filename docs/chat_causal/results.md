# Chat: the prefix-causal tree

Perplexity reduction against sequence only (positive = better), paired by seed, at the
final step (13,100). The `c_` arms are trained on the prefix-causal tree, so what they
read is what a generator can supply.

| arm | old tree: reading | old tree: generation | causal tree: reading | causal tree: generation |
|---|---|---|---|---|
| sequence + depth | +0.21% (5/10, p=0.59) | -0.10% (10 seeds) | +0.16% (5/10, p=0.71) | +0.16% (10 seeds) |
| sequence + grid + depth | +0.91% (9/10, p=0.0051) | +0.66% (10 seeds) | +1.08% (10/10, p=0.00058) | +1.14% (10 seeds) |
| sequence + depth + rank | +0.39% (9/10, p=0.017) | not scored | +0.22% (6/10, p=0.44) | not scored |
| sequence + full tree | +0.91% (9/10, p=0.0078) | not scored | +0.94% (9/10, p=0.00046) | not scored |
| sequence + tree + depth | +0.48% (8/10, p=0.11) | not scored | +0.32% (8/10, p=0.34) | not scored |

## Causal against original, same seeds

| arm | causal − original |
|---|---|
| sequence + depth | -0.04% (4/10, p=0.88) |
| sequence + grid + depth | +0.17% (7/10, p=0.17) |
| sequence + depth + rank | -0.17% (4/10, p=0.55) |
| sequence + full tree | +0.03% (5/10, p=0.92) |
| sequence + tree + depth | -0.16% (4/10, p=0.48) |
