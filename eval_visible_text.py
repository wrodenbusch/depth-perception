#!/usr/bin/env python
"""Loss on the page's VISIBLE TEXT only, page by page -- comparable across the text and tags packs.

    # score one finished run (val and test), writing per-page sums
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python eval_visible_text.py score $TOPOS_RUNS/webc_html/seq_s0
    # compare configurations across packs, paired by seed, on the pages both packs hold
    .venv/bin/python eval_visible_text.py compare

WHY. The "structure as tokens" control (build_web_tags_pack.py, config/webc_html_*.py) trains on
the Web-S pages with their tags written into the stream. Mean nats/token over the whole stream is
not comparable with a text-pack run: ~70% of the tags pack's tokens are markup, mostly easy to
predict, and averaging over them would flatter the tags models for having easier tokens. The
comparable quantity is the loss on the tokens BOTH packs contain: the visible text.

WHICH TOKENS. Text pack: every token inside a text node except the newline that separates nodes
(that newline has no counterpart in the tags pack, where a closing tag ends the node). Tags pack:
every token of a text node (``tok_is_text.npy``). The tags pack is built piecewise so that, page
by page, these two token sequences are IDENTICAL -- the builder checks it, and ``compare`` checks
it again here by hashing each page's scored target ids in every run and refusing on a mismatch.
The context each model conditions on differs (newlines vs tags); the targets do not.

UNIT. Nats per visible-text token, summed over the pages present in every compared run, per seed.
A page the tags pack dropped (too long for 8,192 even whole, or unrenderable) is excluded from both
sides. Runs are paired by seed; text-pack and tags-pack runs of one seed share the initial weights'
seed but not the batch shapes, so the pairing is by seed, not by identical data order.
"""
from __future__ import annotations
from topos_paths import RUNS  # noqa: E402

import argparse
import glob
import hashlib
import json
import math
import os
import re
import statistics as st
import sys
from contextlib import nullcontext

import numpy as np

OUT = f"{RUNS}/analysis/visible_text"
RUNS = {"text": f"{RUNS}/webc", "tags": f"{RUNS}/webc_html"}
FINAL = 9375


def visible_mask(pack: str, a: int, b: int, toks: np.ndarray, newline_id: int) -> np.ndarray:
    side = pack + "_dompath"
    if os.path.exists(os.path.join(side, "tok_is_text.npy")):
        m = np.load(os.path.join(side, "tok_is_text.npy"), mmap_mode="r")
        return np.asarray(m[a:b]).astype(bool)
    tn = np.load(os.path.join(side, "tok_node.npy"), mmap_mode="r")
    return (np.asarray(tn[a:b]) >= 0) & (toks != newline_id)


def score(run_dir: str, device: str = "cuda", limit: int = 0, out_dir: str = OUT) -> None:
    import tiktoken
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import GPT, GPTConfig
    from topos_data import PackedWindows
    from eval_test import sidecar_for

    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    if int(ck.get("iter_num", -1)) != FINAL:
        raise SystemExit(f"{run_dir}: ckpt.pt at step {ck.get('iter_num')}, not {FINAL} -- refusing")
    margs, pack = ck["model_args"], ck["config"]["pack"]
    model = GPT(GPTConfig(**margs))
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    pw = PackedWindows(pack, need_tree=False, sidecar=sidecar_for(margs["pos_mode"], ""))
    pids = json.load(open(os.path.join(pack, "page_ids.json")))
    nl = tiktoken.get_encoding("gpt2").encode_ordinary("\n")[0]
    dev = torch.device(device)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else nullcontext()
    kind = "tags" if os.path.exists(os.path.join(pack + "_dompath", "tok_is_text.npy")) else "text"
    m = re.fullmatch(r"(.+)_s(\d+)", os.path.basename(run_dir.rstrip("/")))
    arm, seed = m.group(1), int(m.group(2))
    for split in ("val", "test"):
        out = os.path.join(out_dir, f"{kind}_{arm}_s{seed}_{split}.npz")
        if os.path.exists(out):
            print(f"exists: {out}")
            continue
        lo, hi = pw.splits[split]
        if limit:
            hi = min(hi, lo + limit)
        ids, sums, counts, hashes = [], [], [], []
        with torch.no_grad():
            for r in range(lo, hi):
                X, Y, POS = pw.get_batch_records([r], dev)
                with ctx:
                    logits, _ = model(X, Y, pos=POS)
                nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)), Y.view(-1),
                                      reduction="none", ignore_index=-1)
                a, b = int(pw.tok_off[r]), int(pw.tok_off[r + 1])
                toks = np.asarray(pw.tokens[a:b], dtype=np.int64)
                mask = visible_mask(pack, a, b, toks, nl)[1:]          # target positions 1..t
                nll = nll.cpu().numpy()[: len(mask)]
                tgt = toks[1:][mask]
                ids.append(pids[r])
                sums.append(float(nll[mask].sum()))
                counts.append(int(mask.sum()))
                hashes.append(hashlib.blake2b(tgt.astype(np.int64).tobytes(), digest_size=8).hexdigest())
        os.makedirs(out_dir, exist_ok=True)
        np.savez(out, page_ids=np.array(ids), sums=np.array(sums), counts=np.array(counts),
                 hashes=np.array(hashes), pack=pack, pos_mode=margs["pos_mode"])
        print(f"{out}: {len(ids)} pages, {sum(counts):,} visible tokens, "
              f"{sum(sums) / max(sum(counts), 1):.4f} nats/token")


def _load(kind: str, arm: str, split: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(OUT, f"{kind}_{arm}_s*_{split}.npz")):
        seed = int(re.search(r"_s(\d+)_", os.path.basename(f)).group(1))
        z = np.load(f)
        out[seed] = {p: (s, c, h) for p, s, c, h in zip(z["page_ids"], z["sums"], z["counts"], z["hashes"])}
    return out


def compare(split: str) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paired import p_two_sided
    cfgs = {("text", "seq"): "text, sequence only", ("text", "depth"): "text, sequence + depth",
            ("tags", "seq"): "tags, sequence only", ("tags", "depth"): "tags, sequence + depth"}
    data = {k: _load(*k, split) for k in cfgs}
    seeds = sorted(set.intersection(*[set(v) for v in data.values() if v] or [set()]))
    if not seeds:
        raise SystemExit("no seed is scored in every configuration yet")
    pages = set.intersection(*[set(d[s]) for d in data.values() for s in seeds])
    # the scored targets must be the same tokens in every run, page by page
    bad = [p for p in pages if len({d[s][p][2] for d in data.values() for s in seeds}) != 1]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} pages score different target tokens across runs, e.g. {bad[:3]}")
    pages = sorted(pages)
    n_tok = sum(next(iter(data.values()))[seeds[0]][p][1] for p in pages)
    rate = {k: {s: sum(data[k][s][p][0] for p in pages) / n_tok for s in seeds} for k in cfgs}
    lines = [f"# Visible-text loss, {split} split: {len(pages)} pages, {n_tok:,} visible tokens, seeds {seeds}",
             "", "| configuration | nats / visible token (mean over seeds) |", "|---|---|"]
    for k, lab in cfgs.items():
        lines.append(f"| {lab} | {st.mean(rate[k].values()):.4f} |")
    lines += ["", "| contrast (A − B) | Δ | A better on | p |", "|---|---|---|---|"]
    for a, b in ((("tags", "seq"), ("text", "seq")), (("text", "depth"), ("tags", "seq")),
                 (("tags", "depth"), ("tags", "seq")), (("text", "depth"), ("text", "seq")),
                 (("tags", "depth"), ("text", "depth"))):
        d = [rate[a][s] - rate[b][s] for s in seeds]
        mean = st.mean(d)
        if len(d) > 1 and st.stdev(d) > 0:
            t = mean / (st.stdev(d) / math.sqrt(len(d)))
            p = f"{p_two_sided(t, len(d) - 1):.2g}"
        else:
            p = "n/a"
        lines.append(f"| {cfgs[a]} − {cfgs[b]} | {mean:+.4f} | {sum(x < 0 for x in d)}/{len(d)} | {p} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, f"compare_{split}.md"), "w") as f:
        f.write(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("run_dir")
    s.add_argument("--device", default="cuda")
    s.add_argument("--limit", type=int, default=0, help="validation only: first N records per split")
    s.add_argument("--out-dir", default=OUT)
    c = sub.add_parser("compare")
    c.add_argument("--split", default="val", choices=("val", "test"))
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.run_dir, a.device, a.limit, a.out_dir)
    else:
        compare(a.split)


if __name__ == "__main__":
    main()
