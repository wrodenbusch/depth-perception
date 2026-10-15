#!/usr/bin/env python
"""Tokens/s per candidate width at block 4096 on ONE card, and the spec's choice.

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python throughput.py --train-tokens <meta.json total train tokens>

Every candidate has 128-wide heads (the rope allocation needs 64 planes). The
rule from the spec: twelve runs in three waves of four plus the smokes must fit
in about 48 hours, so one run gets ``--hours`` (13) of one card; take the
LARGEST candidate whose 13-hour token budget is at least ``--min-tpp`` (20)
tokens per non-embedding parameter, capped at one pass over the train pack.

Measured with the real model under pos_mode='rope_both' (the most expensive
arm: path gather plus rotation on every layer), bf16 autocast, torch.compile,
AdamW step included, synthetic tokens and paths.
"""
import argparse
import json
import time

import torch

from model import GPT, GPTConfig

CANDS = [(6, 256, 2), (6, 384, 3), (8, 512, 4), (6, 640, 5), (10, 512, 4)]


def bench(n_layer, n_embd, n_head, block, steps, micro_batches=(8, 4, 2, 1)):
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    for mb in micro_batches:
        m = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cfg = GPTConfig(block_size=block, vocab_size=50304, n_layer=n_layer, n_head=n_head,
                            n_embd=n_embd, bias=False, dropout=0.0, pos_mode="rope_both")
            m = GPT(cfg).cuda()
            nonemb = m.get_num_params() - m.transformer.wte.weight.numel()
            opt = m.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda")
            mc = torch.compile(m)
            x = torch.randint(0, 50304, (mb, block), device="cuda")
            p = torch.randint(0, 20, (mb, block, 24), device="cuda")
            p[:, :, 8:] = -1
            for i in range(steps + 3):
                if i == 3:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with ctx:
                    _, loss = mc(x, x, pos={"path": p})
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tps = mb * block * steps / (time.time() - t0)
            return dict(n_layer=n_layer, n_embd=n_embd, n_head=n_head, nonemb=int(nonemb),
                        micro_batch=mb, tokens_per_s=round(tps),
                        mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Inductor's triton launcher surfaces a driver OOM as a plain RuntimeError.
            if "out of memory" not in str(e):
                raise
            del m
            torch.cuda.empty_cache()
            continue
    return dict(n_layer=n_layer, n_embd=n_embd, n_head=n_head, oom=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=float, required=True)
    ap.add_argument("--hours", type=float, default=13.0)
    ap.add_argument("--min-tpp", type=float, default=20.0)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default=None, help="write the rows + choice as JSON here")
    a = ap.parse_args()
    rows = []
    for L, D, H in CANDS:
        r = bench(L, D, H, a.block, a.steps)
        if "oom" not in r:
            budget = min(r["tokens_per_s"] * a.hours * 3600, a.train_tokens)
            r.update(budget_tokens=int(budget), tokens_per_param=round(budget / r["nonemb"], 1),
                     hours_for_budget=round(budget / r["tokens_per_s"] / 3600, 1))
        rows.append(r)
        print(json.dumps(r), flush=True)
    ok = [r for r in rows if "oom" not in r and r["tokens_per_param"] >= a.min_tpp]
    if ok:
        pick = max(ok, key=lambda r: r["nonemb"])
    else:
        pick = min((r for r in rows if "oom" not in r), key=lambda r: r["nonemb"])
        pick = dict(pick, note="no candidate reached min tokens/param; smallest taken")
    print("CHOICE", json.dumps(pick))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(rows=rows, choice=pick, args=vars(a)), f, indent=1)


if __name__ == "__main__":
    main()
