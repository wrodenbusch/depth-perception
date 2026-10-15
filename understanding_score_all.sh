#!/bin/sh
# Score every finished run for the understanding tests: shard $1 of $2, on whatever card
# CUDA_VISIBLE_DEVICES names. Runs settings in order; already-scored runs are skipped.
cd "$(dirname "$0")"
for s in sheets webc py; do
    ./.venv/bin/python understanding_evals.py score --setting "$s" --shard "$1/$2"
done
