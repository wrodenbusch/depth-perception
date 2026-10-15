#!/usr/bin/env python
"""Tokens/s against ``token_budget`` for ONE fixed geometry, on the REAL pack. One card.

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python budget_sweep.py \
        --pack $TOPOS_PACKS/web_whole_b2048 --sidecar path --block 2048

WHAT THIS ANSWERS, and why it was worth measuring rather than assuming. Both existing cells
inherited ``token_budget = 8192``, a figure sized for a 31.5M-parameter model at a 4,096 context.
A 4.7M-parameter model does about a seventh of the arithmetic per token, but that saving only
becomes wall-clock if the card is actually filled -- at a budget that small a tiny model spends
its time on kernel launches and the optimizer step, not on matmuls. The claim "a smaller model
finishes sooner" is therefore not free; it has to be bought with a bigger batch, and this measures
how much bigger before the return flattens.

USES THE REAL LOADER, not synthetic tensors, because two of the costs are in the data path:
bucket batching groups records by length, and ``PAD_TO`` rounds every group up, so a sizeable
fraction of the positions the GPU processes are padding. A synthetic benchmark reports the
throughput of a batch shape the loader would never hand you.

REPORTS TOKENS/S AND ALSO PADDING. The headline number is real tokens per second -- padding
excluded from the numerator, included in the cost -- since that is what sets a run's wall clock.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import topos_data
from model import GPT, GPTConfig


def build(block: int, vocab: int, n_layer: int, n_embd: int, n_head: int, device):
    cfg = GPTConfig(block_size=block, vocab_size=vocab, n_layer=n_layer, n_head=n_head,
                    n_embd=n_embd, dropout=0.0, bias=False, pos_mode="rope_both",
                    rope_seq_planes=16, rope_levels=22, rope_planes_per_level=2,
                    rope_token_planes=4, rope_decay=False)
    m = GPT(cfg).to(device)
    return m, torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--sidecar", default="path", choices=("path", "tree"))
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--budgets", default="8192,16384,32768,65536,131072,262144")
    ap.add_argument("--records-per-step", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6, help="timed steps per budget, after a warmup")
    ap.add_argument("--no-compile", dest="compile", action="store_false", default=True,
                    help="skip torch.compile -- faster to start, so use it to check the "
                         "harness, never to quote a throughput number")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "one card, set CUDA_VISIBLE_DEVICES"
    device = torch.device("cuda")
    pw = topos_data.PackedWindows(pack_dir=a.pack, need_tree=False, sidecar=a.sidecar)
    print(f"{pw}\n")
    model, opt = build(a.block, pw.vocab_size, a.n_layer, a.n_embd, a.n_head, device)
    nonemb = sum(p.numel() for n, p in model.named_parameters() if "wte" not in n and "wpe" not in n)
    print(f"non-embedding parameters {nonemb/1e6:.2f}M\n")
    if a.compile:
        model = torch.compile(model)

    gen = torch.Generator().manual_seed(0)
    rows = []
    print(f"{'budget':>9} {'groups':>7} {'real tok':>10} {'padded':>10} {'pad%':>6} "
          f"{'s/step':>8} {'real tok/s':>11} {'h for 300M':>11}")
    for budget in [int(x) for x in a.budgets.split(",")]:
        try:
            timings, real_tok, pad_tok, ngroups = [], 0, 0, 0
            for it in range(a.steps + 2):                       # first two are warmup
                groups, step_tokens = pw.step_batches("train", a.records_per_step, budget,
                                                      generator=gen)
                if it == 2:
                    torch.cuda.synchronize()
                t0 = time.time()
                opt.zero_grad(set_to_none=True)
                r = p = 0
                for gi in groups:
                    out = pw.get_batch_records(gi, device)
                    x, y, pos = out[0], out[1], out[2]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = model(x, y, pos=pos)
                    (loss / len(groups)).backward()
                    p += x.numel()
                    r += int((y >= 0).sum()) if y.dtype == torch.long else x.numel()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                torch.cuda.synchronize()
                if it >= 2:
                    timings.append(time.time() - t0)
                    real_tok += step_tokens
                    pad_tok += p
                    ngroups = len(groups)
            s = sum(timings) / len(timings)
            tps = (real_tok / len(timings)) / s
            padfrac = 1 - (real_tok / pad_tok) if pad_tok else 0
            hours = 300e6 / tps / 3600
            rows.append(dict(budget=budget, groups=ngroups, s_per_step=s, real_tok_s=tps,
                             pad_frac=padfrac, hours_300M=hours))
            print(f"{budget:>9} {ngroups:>7} {real_tok//len(timings):>10} "
                  f"{pad_tok//len(timings):>10} {100*padfrac:>5.1f}% {s:>8.3f} "
                  f"{tps:>11,.0f} {hours:>11.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{budget:>9} {'OOM':>7} -- stopping the sweep here")
            torch.cuda.empty_cache()
            break
    if rows:
        best = min(rows, key=lambda r: r["hours_300M"])
        print(f"\nfastest: budget {best['budget']} -> {best['real_tok_s']:,.0f} real tokens/s, "
              f"{best['hours_300M']:.2f} h per 300M tokens per card")
        base = rows[0]
        print(f"against the inherited budget of {base['budget']}: "
              f"{base['hours_300M']/best['hours_300M']:.2f}x faster")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
