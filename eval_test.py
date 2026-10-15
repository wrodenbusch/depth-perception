#!/usr/bin/env python
"""Held-out evaluation of one milestone checkpoint on the TEST split.

    .venv/bin/python eval_test.py $TOPOS_RUNS/web2/rope_leaf_s0            # best-by-val milestone
    .venv/bin/python eval_test.py <run_dir> --ckpt ckpt_12660.pt --split test --records 1024

The validation set picks each run's best milestone; that choice is a (mild)
selection on val, so the number quoted as held-out must come from records the
selection never saw. This scores the first ``--records`` records of ``--split``
in fixed batches, exactly as ``train.py``'s ``evaluate_fixed_val`` does
(token-weighted nats/token over non-pad targets, and nats/record), and appends
one JSON line to ``<run_dir>/heldout.jsonl``.

The checkpoint's ``model_args`` carry ``pos_mode`` and the rope allocation, so
the right sidecar and rotation are rebuilt from the checkpoint alone; the pack
comes from the run's ``config`` if the full checkpoint has one, else ``--pack``.
bf16 autocast on CUDA, no torch.compile.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch

from model import GPT, GPTConfig
from topos_data import PackedWindows
from rope_pos import TREE_MODES, SLOT_MODES, PLAN_MODE, plan_needs_path

def sidecar_for(pos_mode: str, layer_plan: str = "", depth_emb: bool = False) -> str | None:
    """Which sidecar an arm needs. Derived from rope_pos, not a hardcoded map.

    The old hardcoded map here was missing 'rope_all' AND both split arms, so scoring a
    rope_all run would have died with "needs a DOM path" (found 2026-09-26).
    """
    if pos_mode == 'tree':
        return 'tree'
    if depth_emb:                      # the input-embedding depth control reads the path sidecar
        return 'path'
    if pos_mode in TREE_MODES:
        return 'path'
    if pos_mode == PLAN_MODE:
        return 'path' if plan_needs_path(layer_plan) else None
    return None


def best_milestone(run_dir: str) -> str:
    evals = [json.loads(l) for l in open(os.path.join(run_dir, "train.jsonl")) if '"val_nats_tok"' in l]
    if not evals:
        raise SystemExit(f"no eval lines in {run_dir}/train.jsonl")
    b = min(evals, key=lambda e: e["val_nats_tok"])
    return f"ckpt_{b['step']}.pt"


def score(model, pw, split, records, batch_size, device, ctx):
    lo, hi = pw.splits[split]
    hi = min(hi, lo + records)
    tot, n_tok, n_rec = 0.0, 0, 0
    for start in range(lo, hi, batch_size):
        ix = list(range(start, min(start + batch_size, hi)))
        X, Y, POS = pw.get_batch_records(ix, device)
        with ctx:
            _, loss = model(X, Y, pos=POS)
        nt = int((Y != -1).sum())
        tot += float(loss) * nt
        n_tok += nt
        n_rec += len(ix)
    return tot / n_tok, tot / n_rec, n_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default=None, help="milestone file name; default = best-by-val from train.jsonl")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--records", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--pack", default=None, help="override the pack (else the checkpoint's config)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ckpt_name = a.ckpt or best_milestone(a.run_dir)
    ck = torch.load(os.path.join(a.run_dir, ckpt_name), map_location=a.device)
    margs = ck["model_args"]
    pack = a.pack or (ck.get("config") or {}).get("pack")
    if pack is None:
        full = os.path.join(a.run_dir, "ckpt.pt")
        if os.path.exists(full):
            pack = torch.load(full, map_location="cpu")["config"]["pack"]
    if pack is None:
        raise SystemExit("no pack recorded in the checkpoint; pass --pack")
    model = GPT(GPTConfig(**margs))
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(a.device).eval()
    pw = PackedWindows(pack, need_tree=False,
                       sidecar=sidecar_for(margs["pos_mode"], margs.get("rope_layer_plan", ""),
                                           margs.get("depth_emb", False)),
                       need_slot=margs["pos_mode"] in SLOT_MODES)
    device = torch.device(a.device)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        nats_tok, nats_page, n = score(model, pw, a.split, a.records, a.batch_size, device, ctx)
    rec = dict(ckpt=ckpt_name, step=int(ck.get("iter_num", -1)), split=a.split, records=n,
               nats_tok=nats_tok, nats_page=nats_page, pack=pack, pos_mode=margs["pos_mode"])
    with open(os.path.join(a.run_dir, "heldout.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
