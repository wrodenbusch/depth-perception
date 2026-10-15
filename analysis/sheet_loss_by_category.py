#!/usr/bin/env python
"""Per-token loss on the spreadsheet val/test splits, split by token category and depth.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python analysis/sheet_loss_by_category.py \
        --out $TOPOS_RUNS/analysis/sheets_depth

Scores the FINAL checkpoint (step 8,240) of every seed of five configurations on EVERY val and
test record, keeps the loss of each target token, and sums it by the target's category and depth
(labels from ``sheet_token_categories.py``). Batches are built by the same
``PackedWindows.get_batch_records`` the training-time evaluation uses, so positions and
coordinates are exactly what the model saw. A paired comparison per seed then asks where, if
anywhere, depth's gain on spreadsheets lives.
"""
from __future__ import annotations
from topos_paths import PACKS, RUNS  # noqa: E402

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from topos_data import PackedWindows

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_test import sidecar_for          # noqa: E402
from paired import p_two_sided             # noqa: E402

ROOT = f"{RUNS}/sheets"
ARMS = {"seq": ("seq_v2", f"{PACKS}/sheet_values_v2"),
        "depth": ("depth_v2", f"{PACKS}/sheet_values_v2"),
        "tree": ("both_v2", f"{PACKS}/sheet_values_v2"),
        "tree_depth": ("rel_v2", f"{PACKS}/sheet_values_v2"),
        "toplevel": ("toplevel_v2", f"{PACKS}/sheet_values_v2_lvl1")}
FINAL = "ckpt_8240.pt"
DEPTH_BUCKETS = [(0, 0), (1, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 99)]


def score_run(run_dir, pack, labels, device, batch_tokens=12000):
    ck = torch.load(os.path.join(run_dir, FINAL), map_location=device)
    margs = ck["model_args"]
    model = GPT(GPTConfig(**margs))
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
          for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    pw = PackedWindows(pack, need_tree=False,
                       sidecar=sidecar_for(margs["pos_mode"], margs.get("rope_layer_plan", "")))
    A = int(labels["start"])
    cat, depth = labels["cat"], labels["depth"]
    ncat = len(labels["cats"])
    dbin = np.zeros(128, np.int64)
    for i, (lo, hi) in enumerate(DEPTH_BUCKETS):
        dbin[lo:hi + 1] = i
    out = {}
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    for split in ("val", "test"):
        lo, hi = pw.splits[split]
        recs = sorted(range(lo, hi), key=lambda r: int(pw.tok_off[r + 1] - pw.tok_off[r]))
        cs, cn = np.zeros(ncat), np.zeros(ncat, np.int64)
        ds, dn = np.zeros(len(DEPTH_BUCKETS)), np.zeros(len(DEPTH_BUCKETS), np.int64)
        batch = []
        def flush(batch):
            X, Y, POS = pw.get_batch_records(batch, device)
            with torch.no_grad(), ctx:
                logits, _ = model(X, Y, pos=POS)
            ce = F.cross_entropy(logits.float().transpose(1, 2), Y, ignore_index=-1,
                                 reduction="none").cpu().numpy()
            for i, r in enumerate(batch):
                t = int(pw.tok_off[r + 1] - pw.tok_off[r]) - 1
                g = int(pw.tok_off[r]) - A + 1             # label of the TARGET token
                c = cat[g:g + t]
                d = dbin[depth[g:g + t]]
                l = ce[i, :t]
                np.add.at(cs, c, l); np.add.at(cn, c, 1)
                np.add.at(ds, d, l); np.add.at(dn, d, 1)
        for r in recs:
            L = int(pw.tok_off[r + 1] - pw.tok_off[r])
            if batch and (len(batch) + 1) * L > batch_tokens:
                flush(batch); batch = []
            batch.append(r)
        if batch:
            flush(batch)
        out[split] = dict(cat_sum=cs.tolist(), cat_n=cn.tolist(), d_sum=ds.tolist(),
                          d_n=dn.tolist(), overall=float(cs.sum() / cn.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-9")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    labels = dict(np.load(os.path.join(a.out, "token_labels.npz")))
    device = torch.device("cuda")
    raw_path = os.path.join(a.out, "loss_by_category_raw.json")
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    for arm, (d, pack) in ARMS.items():
        for s in range(lo, hi + 1):
            key = f"{arm}_s{s}"
            if key in raw:
                continue
            rd = os.path.join(ROOT, f"{d}_s{s}")
            if not os.path.exists(os.path.join(rd, FINAL)):
                print("missing", rd); continue
            raw[key] = score_run(rd, pack, labels, device)
            json.dump(raw, open(raw_path, "w"))
            print(key, {k: round(v["overall"], 4) for k, v in raw[key].items()}, flush=True)
    summarise(raw, labels, a.out)


def summarise(raw, labels, out_dir):
    cats = [str(c) for c in labels["cats"]]
    groups = {c: [i] for i, c in enumerate(cats)}
    groups["ALL formula tokens"] = [cats.index(c) for c in ("formula_eq", "ref", "func",
                                    "formula_num", "formula_op", "formula_punct")]
    groups["ALL non-formula tokens"] = [cats.index(c) for c in ("sep", "value_num", "value_text",
                                        "shallow")]
    groups["ALL"] = list(range(len(cats)))
    dnames = [f"depth {a}" if a == b else f"depth {a}-{b}" for a, b in DEPTH_BUCKETS]
    res = {}
    lines = []
    for split in ("val", "test"):
        res[split] = {}
        base = {s: raw[f"seq_s{s}"][split] for s in range(10) if f"seq_s{s}" in raw}
        share = None
        for arm in ("depth", "tree", "tree_depth", "toplevel"):
            seeds = [s for s in base if f"{arm}_s{s}" in raw]
            if not seeds:
                continue
            res[split][arm] = {}
            def mean_of(rec, idx, key="cat"):
                return sum(rec[f"{key}_sum"][i] for i in idx) / max(1, sum(rec[f"{key}_n"][i] for i in idx))
            rows = list(groups.items()) + [(n, [i]) for i, n in enumerate(dnames)]
            for gname, idx in rows:
                key = "d" if gname.startswith("depth") else "cat"
                diffs = [mean_of(raw[f"{arm}_s{s}"][split], idx, key) - mean_of(base[s], idx, key)
                         for s in seeds]
                n_tok = sum(base[seeds[0]][f"{key}_n"][i] for i in idx)
                tot = sum(sum(base[seeds[0]][f"{key}_n"]) for _ in [0])
                m = st.mean(diffs)
                sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
                t = m / (sd / len(diffs) ** 0.5) if sd else float("inf")
                p = p_two_sided(t, len(diffs) - 1) if sd else 0.0
                base_mean = st.mean(mean_of(base[s], idx, key) for s in seeds)
                res[split][arm][gname] = dict(
                    share=n_tok / tot, seq_nats=base_mean, diff=m, sd=sd, t=t, p=p,
                    won=sum(d < 0 for d in diffs), n=len(diffs),
                    contribution=m * n_tok / tot)
    json.dump(res, open(os.path.join(out_dir, "loss_by_category_summary.json"), "w"), indent=1)
    for split in ("val", "test"):
        for arm, g in res[split].items():
            lines.append(f"\n== {split}: {arm} minus sequence only (nats/token of the target; "
                         f"negative = better; contribution = diff x share)")
            lines.append(f"{'group':26s} {'share':>6s} {'seq':>7s} {'diff':>8s} {'won':>6s} "
                         f"{'p':>7s} {'contrib':>8s}")
            for gname, r in g.items():
                lines.append(f"{gname:26s} {r['share']:6.3f} {r['seq_nats']:7.3f} "
                             f"{r['diff']:+8.4f} {r['won']:>3d}/{r['n']:<2d} {r['p']:7.4f} "
                             f"{r['contribution']:+8.5f}")
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "loss_by_category.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
