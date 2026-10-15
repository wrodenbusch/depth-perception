#!/usr/bin/env python
"""Verify, end to end, the depth a ``rope_depth`` / ``rope_rel`` model actually receives.

    .venv/bin/python analysis/depth_trace.py --out $TOPOS_RUNS/analysis/sheets_depth

WHY. Sequence + depth barely helped on spreadsheets. Before reading anything into that, rule out
a plumbing bug: the depth is computed in ``rope_pos.GraphRoPE._rel_angles`` as the count of
non-ABSENT levels of ``pos['path']``, and ``pos['path']`` is built by
``topos_data.PackedWindows.get_batch_records`` from ``tok_node`` + ``node_off`` + ``node_path``.
This script runs that exact path -- the real loader, the real rotary module rebuilt from a trained
checkpoint, its real ``_rel_angles`` -- and recovers depth as ``angle / step_rel``, so nothing is
re-implemented. It does it on spreadsheets and on the two corpora where depth worked (Python,
Web-S) as controls, and reports how far the trained ``step_rel`` moved from its initialisation:
a depth that were constant or zero would leave that parameter with almost no gradient.
"""
from __future__ import annotations
from topos_paths import PACKS, RUNS  # noqa: E402

import argparse
import glob
import json
import os

import numpy as np
import tiktoken
import torch

from model import GPT, GPTConfig
from topos_data import PackedWindows

SETTINGS = {
    "sheets": dict(runs=f"{RUNS}/sheets/depth_v2_s*", final="ckpt_8240.pt", pack=f"{PACKS}/sheet_values_v2",
                   tokenizer="gpt2"),
    "python": dict(runs=f"{RUNS}/py/depth_s*", final="ckpt_11200.pt", tokenizer=None, pack=f"{PACKS}/py_ast_423k"),
    "webs": dict(runs=f"{RUNS}/webc/depth_s*", final="ckpt_9375.pt", tokenizer="gpt2", pack=f"{PACKS}/web_whole_b2048"),
}


def load(run_dir: str, ck_name: str):
    ck = torch.load(os.path.join(run_dir, ck_name), map_location="cpu")
    margs = ck["model_args"]
    model = GPT(GPTConfig(**margs))
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
          for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    pack = (ck.get("config") or {}).get("pack")
    return model, margs, pack, sd


def rotary_module(model):
    for m in model.modules():
        if hasattr(m, "_rel_angles") and hasattr(m, "step_rel"):
            return m
    raise SystemExit("no module with _rel_angles/step_rel")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=4)
    a = ap.parse_args()
    report = {}
    lines = []
    for name, s in SETTINGS.items():
        runs = sorted(glob.glob(s["runs"]))
        runs = [r for r in runs if os.path.exists(os.path.join(r, s["final"]))]
        model, margs, _, _ = load(runs[0], s["final"])
        pack = s["pack"]
        rot = rotary_module(model)
        pw = PackedWindows(pack, need_tree=False, sidecar="path")
        lo, hi = pw.splits["val"]
        # pick records with the most distinct depths, from the first 300 val records
        cand = []
        for r in range(lo, min(hi, lo + 300)):
            toks, p = pw.record(r)
            d = (p["path"][:, : rot.levels] >= 0).sum(1)
            cand.append((len(np.unique(d)), -len(toks), r))
        cand.sort(reverse=True)
        pick = sorted([r for _, _, r in cand[: a.records]])
        X, Y, POS = pw.get_batch_records(pick, torch.device("cpu"))
        with torch.no_grad():
            ang = rot._rel_angles(POS["path"], X.shape[0], X.shape[1], torch.device("cpu"))
        step = rot.step_rel.detach()
        depth_from_angle = (ang[..., 0] / step[0]).round().long()
        # independent recount straight from the sidecar arrays (no loader)
        indep = []
        for r in pick:
            a0, b0 = int(pw.tok_off[r]), int(pw.tok_off[r + 1])
            tn = np.asarray(pw.tok_node[a0:b0], dtype=np.int64)
            na, nb = int(pw.node_off[r]), int(pw.node_off[r + 1])
            rows = np.asarray(pw.node_path[na:nb], dtype=np.int64)
            assert tn.max() < nb - na, f"{name} record {r}: tok_node {tn.max()} >= {nb-na} nodes"
            d = np.where(tn >= 0, (rows[np.maximum(tn, 0)] >= 0).sum(1), 0)
            indep.append(d)
        mism = 0
        for i, d in enumerate(indep):
            t = len(d) - 1
            mism += int((depth_from_angle[i, :t].numpy() != d[:t]).sum())
        enc = tiktoken.get_encoding("gpt2") if s["tokenizer"] else None
        lines.append(f"\n===== {name}: {pack}  records {pick}  levels={rot.levels}")
        for i, r in enumerate(pick[:2]):
            toks = X[i].tolist()
            t = int((Y[i] != -1).sum())
            row = []
            for j in range(min(t, 140)):
                txt = enc.decode([toks[j]]) if enc else str(toks[j])
                path = [int(v) for v in POS["path"][i, j, : rot.levels].tolist() if v >= 0]
                row.append(f"{txt!r}:d{int(depth_from_angle[i, j])}{path}")
            lines.append(f"-- record {r}:\n" + "  ".join(row))
        # step movement over all seeds
        moved = []
        for rd in runs:
            sd = torch.load(os.path.join(rd, s["final"]), map_location="cpu")["model"]
            k = [k for k in sd if k.endswith("step_rel")][0]
            moved.append(sd[k].tolist())
        init = rot.step_rel.detach().clone()
        # init values are recomputed from a FRESH module, not the trained one
        fresh = GPT(GPTConfig(**margs))
        init = [m for m in fresh.modules() if hasattr(m, "step_rel")][0].step_rel.detach().tolist()
        alld = np.concatenate(indep)
        report[name] = dict(
            pack=pack, levels=rot.levels, records=pick,
            loader_vs_sidecar_mismatches=mism,
            depth_values=sorted(int(x) for x in np.unique(alld)),
            depth_counts={int(v): int(c) for v, c in zip(*np.unique(alld, return_counts=True))},
            step_rel_init=init, step_rel_trained_per_seed=moved,
            step_rel_rel_change_mean=[float(np.mean([abs(m[p] - init[p]) / abs(init[p])
                                                     for m in moved])) for p in range(len(init))],
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(report, open(os.path.join(a.out, "depth_trace.json"), "w"), indent=1)
    open(os.path.join(a.out, "depth_trace_tokens.txt"), "w").write("\n".join(lines))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "step_rel_trained_per_seed"}
                      for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
