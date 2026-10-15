#!/bin/bash
# Every mechanism analysis, in order, on card 1 only. Resumable: each step skips finished runs.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python
$PY analysis/mechanism.py steps
for s in py sheets webc; do $PY analysis/mechanism.py remove --setting $s; done
for s in py sheets webc; do $PY analysis/mechanism.py attn --setting $s; done
echo ALL DONE
