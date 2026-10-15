#!/usr/bin/env python
"""Pick each arm's learning rate from the smoke runs and write it into config/web_<arm>.py.

    .venv/bin/python pick_lr.py $TOPOS_RUNS/web            # print the table only
    .venv/bin/python pick_lr.py $TOPOS_RUNS/web --write    # and edit the configs

The rule (spec): lowest FINAL val nats/token at 5% of the budget, per arm. A smoke
that did not reach the final eval is reported but cannot win.
"""
import argparse
import glob
import json
import os
import re

ARMS = ("none", "rope_seq", "rope_graph", "rope_both", "rope_leaf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--final-step", type=int, default=352)
    ap.add_argument("--config-prefix", default="web_", help="web_ for cell 1, web2_ for cell 2")
    ap.add_argument("--arms", default="", help="comma-separated arm names; default is the web cell's five")
    a = ap.parse_args()
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip()) or ARMS
    table = {}
    for d in sorted(glob.glob(os.path.join(a.root, "smoke_*_lr*"))):
        m = re.match(r"smoke_(.+)_lr([0-9.e-]+)$", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "train.jsonl")):
            continue
        arm, lr = m.group(1), m.group(2)
        evals = [json.loads(l) for l in open(os.path.join(d, "train.jsonl")) if '"val_nats_tok"' in l]
        final = [e for e in evals if e["step"] == a.final_step]
        table.setdefault(arm, {})[lr] = dict(final=final[0]["val_nats_tok"] if final else None,
                                              last=evals[-1]["step"] if evals else None,
                                              best=min(e["val_nats_tok"] for e in evals) if evals else None)
    choice = {}
    for arm in arms:
        rows = table.get(arm, {})
        print(f"{arm:12}", "  ".join(f"lr={lr}: {r['final'] if r['final'] is not None else f'incomplete@{r['last']}'}"
                                    for lr, r in sorted(rows.items(), key=lambda kv: float(kv[0]))))
        done = {lr: r["final"] for lr, r in rows.items() if r["final"] is not None}
        if done:
            choice[arm] = min(done, key=done.get)
    print("choice:", choice)
    if a.write:
        for arm, lr in choice.items():
            path = f"config/{a.config_prefix}{arm}.py"
            src = open(path).read()
            src = re.sub(r"^learning_rate = .*$", f"learning_rate = {lr}                  # LR smoke, {a.final_step} steps: lowest final val nats/tok", src, flags=re.M)
            src = re.sub(r"^min_lr = .*$", f"min_lr = {float(lr) / 10:.1e}", src, flags=re.M)
            open(path, "w").write(src)
            print("wrote", path, lr)


if __name__ == "__main__":
    main()
