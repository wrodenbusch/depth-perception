#!/usr/bin/env python
"""How trained models USE the structural planes. Existing checkpoints only, no training.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python analysis/mechanism.py remove --setting sheets
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python analysis/mechanism.py attn   --setting sheets
    .venv/bin/python analysis/mechanism.py steps
    .venv/bin/python analysis/mechanism.py report

Four analyses, each answering one question a reader asks of the loss tables.

1. REMOVE  -- "does the model rely on the structure it is given?"  The same test records are
   scored three ways: the true structure; structure REMOVED (every token's path set to ABSENT,
   which is what the model saw for tokens outside the tree, so depth 0 / no coordinate); and
   structure SHUFFLED (each record's per-token paths permuted among its own tokens, so the
   coordinates are realistic but attached to the wrong tokens). The loss increase over the true
   structure is set against the model's gain over the same seed's sequence-only run: an increase
   about equal to the gain means the model simply uses the structure; a larger one means it
   came to depend on it, and without it does worse than a model that never had it.

2. ATTN    -- "how does the structure change where attention goes?"  Attention probabilities are
   recomputed from each layer's rotated queries and keys (the fused kernel does not return them).
   For every head, the attention mass a query puts on keys standing in a structural relation to
   it (same row, same column above, the referenced cell; same node, sibling, same depth, ancestor,
   other branch) is compared with the share those keys would get under uniform causal attention
   ("lift" = mass / chance), and the structured model is compared with the SAME SEED's
   sequence-only model on the same tokens. The diagonal (a token attending to itself) is
   excluded from every relation.

3. SPLIT   -- "how much of each attention score does the structure rotation move?"  For every
   head the score q.k is written as a sum over rotary planes; the planes are grouped into
   sequence, structure (tree path / grid / depth) and unrotated. Because the structural planes
   would be plain content channels if they were not rotated, the informative quantity is how
   much the structure ROTATION changes the score: the same q and k are scored with the
   structural planes rotated and unrotated, and the spread of that change is reported relative
   to the spread of the whole score.

4. STEPS   -- the learned rotation parameters (per-level step and offset, the depth step) against
   their initial values. NOTE: these are SHARED by every head and layer in this implementation --
   one rotary module serves the whole network -- so there is no per-head step to tabulate. Which
   heads are structure-sensitive is a property of each head's query/key weights, measured by 2
   and 3.

Batches are built exactly as eval_test.py / understanding_evals.py build them (whole test
records, EOT-padded inputs, -1 targets ignored, bf16 autocast for the loss pass).
"""
from __future__ import annotations
from topos_paths import PACKS, RUNS  # noqa: E402

import argparse
import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ROOT = f"{RUNS}"
OUT = f"{RUNS}/analysis/mechanism"
LEVELS = 22
SEEDS = [0, 1, 2, 3, 4]

SETTINGS = {
    "py": dict(root="py", final=11200, base_pack=f"{PACKS}/py_ast_423k", seq="rope_seq",
               arms=["depth", "rope_both", "rel"], remove_records=4096, batch=16,
               attn_records=48, attn_arms=["depth", "rel"]),
    "sheets": dict(root="sheets", final=8240, base_pack=f"{PACKS}/sheet_values_v2",
                   seq="seq_v2", arms=["depth_v2", "grid_v2", "both_v2", "rel_v2"],
                   remove_records=512, batch=2, attn_records=8, attn_arms=["depth_v2", "grid_v2", "rel_v2"]),
    "webc": dict(root="webc", final=9375, base_pack=f"{PACKS}/web_whole_b2048", seq="seq",
                 arms=["depth", "both", "rel"], remove_records=512, batch=4,
                 attn_records=16, attn_arms=["depth", "rel"]),
}
PRETTY = {"rope_seq": "sequence only", "seq_v2": "sequence only", "seq": "sequence only",
          "depth": "sequence + depth", "depth_v2": "sequence + depth",
          "rope_both": "sequence + tree", "both_v2": "sequence + tree", "both": "sequence + tree",
          "rel": "sequence + tree + depth", "rel_v2": "sequence + tree + depth",
          "grid_v2": "sequence + grid"}
SETTING_NAME = {"py": "Python", "sheets": "Spreadsheets", "webc": "Web-S"}


def run_dir(setting, arm, seed):
    return os.path.join(ROOT, SETTINGS[setting]["root"], f"{arm}_s{seed}")


def load_model(rd, final, device):
    import torch
    from model import GPT, GPTConfig
    ck = torch.load(os.path.join(rd, "ckpt.pt"), map_location="cpu", weights_only=False)
    if int(ck.get("iter_num", -1)) != final:
        raise SystemExit(f"{rd}: ckpt at step {ck.get('iter_num')}, not the final {final}")
    margs = ck["model_args"]
    model = GPT(GPTConfig(**margs))
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, margs, ck["config"]["pack"]


def windows(pack, margs):
    from topos_data import PackedWindows
    from eval_test import sidecar_for
    from rope_pos import SLOT_MODES, REF_COORD_MODES, REF_ANCHOR_MODES
    mode = margs["pos_mode"]
    return PackedWindows(pack, need_tree=False,
                         sidecar=sidecar_for(mode, margs.get("rope_layer_plan", ""), margs.get("depth_emb", False)),
                         need_slot=mode in SLOT_MODES, need_ref=mode in REF_COORD_MODES,
                         need_ref_sheet=mode in REF_ANCHOR_MODES)


def paired(diffs):
    from paired import p_two_sided
    n = len(diffs)
    m = st.mean(diffs)
    if n < 2:
        return dict(n=n, mean=m)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    t = m / se if se else float("inf")
    return dict(n=n, mean=m, sd=sd, t=t, p=p_two_sided(t, n - 1) if se else 0.0,
                pos=sum(d > 0 for d in diffs))


# ------------------------------------------------------------------ 1. remove

def perturb(pos, how, gen):
    """Return a copy of the batch's position dict with the structure removed or shuffled."""
    import torch
    if pos is None or how == "true":
        return pos
    out = dict(pos)
    path = pos["path"]
    if how == "absent":
        out["path"] = torch.full_like(path, -1)
    elif how == "shuffle":
        p = path.clone()
        for i in range(path.shape[0]):
            # permute among the record's own t positions; padding stays ABSENT
            t = int(out["_t"][i])
            perm = torch.randperm(t, generator=gen)
            p[i, :t] = path[i, :t][perm.to(path.device)]
        out["path"] = p
    out.pop("_t", None)
    return out


def score(model, pw, recs, batch, how, device, seed=0):
    import torch
    gen = torch.Generator().manual_seed(1000 + seed)
    tot, n = 0.0, 0
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for s in range(0, len(recs), batch):
            ix = recs[s:s + batch]
            X, Y, POS = pw.get_batch_records(ix, device)
            if POS is not None and how != "true":
                POS = dict(POS)
                POS["_t"] = [int(pw.tok_off[r + 1] - pw.tok_off[r]) - 1 for r in ix]
                POS = perturb(POS, how, gen)
            with ctx:
                logits, _ = model(X, Y, pos=POS)
            lp = torch.log_softmax(logits.float(), dim=-1)
            m = Y >= 0
            nll = -lp.gather(-1, Y.clamp(min=0)[..., None])[..., 0]
            tot += float(nll[m].sum())
            n += int(m.sum())
    return tot / n


def cmd_remove(setting):
    import torch
    cfg = SETTINGS[setting]
    dev = torch.device("cuda")
    out_path = os.path.join(OUT, f"remove_{setting}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for seed in SEEDS:
        for arm in [cfg["seq"]] + cfg["arms"]:
            key = f"{arm}_s{seed}"
            hows = ["true"] if arm == cfg["seq"] else ["true", "absent", "shuffle"]
            if key in res and all(h in res[key] for h in hows):
                continue
            rd = run_dir(setting, arm, seed)
            if not os.path.exists(os.path.join(rd, "ckpt.pt")):
                print("missing", rd)
                continue
            model, margs, pack = load_model(rd, cfg["final"], dev)
            pw = windows(pack, margs)
            lo, hi = pw.splits["test"]
            recs = list(range(lo, min(hi, lo + cfg["remove_records"])))
            res[key] = {h: score(model, pw, recs, cfg["batch"], h, dev, seed) for h in hows}
            print(setting, key, {h: round(v, 4) for h, v in res[key].items()}, flush=True)
            os.makedirs(OUT, exist_ok=True)
            json.dump(res, open(out_path, "w"), indent=1)
            del model
            torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ 2+3. attention

def structure_ids(path):
    """Per token: depth, node id, parent id, and prefix ids per level, from a (T, >=22) path."""
    import torch
    P = path[:, :LEVELS]
    d = (P >= 0).sum(-1)
    T = P.shape[0]
    ids = torch.full((LEVELS + 1, T), -1, dtype=torch.long, device=P.device)
    for k in range(1, LEVELS + 1):
        pre = P[:, :k].clone()
        ok = d >= k
        if ok.any():
            _, inv = torch.unique(pre[ok], dim=0, return_inverse=True)
            ids[k, ok] = inv
    node = ids.gather(0, d.clamp(min=0)[None])[0]
    node[d == 0] = -1
    parent = ids.gather(0, (d - 1).clamp(min=0)[None])[0]
    parent[d <= 1] = -1
    return d, node, parent, ids


def relations(setting, path, ref):
    """{name: (query_mask (T,), key_relation (T,T) bool)} with j < i, diagonal excluded."""
    import torch
    T = path.shape[0]
    dev = path.device
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev), diagonal=-1)
    d, node, parent, ids = structure_ids(path)
    rel = {}
    intree = d > 0
    if setting == "sheets":
        sh, row, col = path[:, 0], path[:, 1], path[:, 2]
        cell = d >= 3
        same_row = (sh[:, None] == sh[None]) & (row[:, None] == row[None]) & cell[:, None] & cell[None]
        same_col_above = ((sh[:, None] == sh[None]) & (col[:, None] == col[None]) &
                          (row[None] < row[:, None]) & cell[:, None] & cell[None])
        rel["same row"] = (cell, same_row & causal)
        rel["same column, earlier row"] = (cell, same_col_above & causal)
        if ref is not None:
            has = (ref >= 0).all(-1)
            tgt = ((sh[None] == ref[:, 0, None]) & (row[None] == ref[:, 1, None]) &
                   (col[None] == ref[:, 2, None]) & cell[None])
            rel["the referenced cell"] = (has, tgt & causal)
    else:
        same_node = (node[:, None] == node[None]) & intree[:, None] & intree[None]
        sib = (parent[:, None] == parent[None]) & (parent[:, None] >= 0) & ~same_node
        same_depth = (d[:, None] == d[None]) & intree[:, None] & ~same_node & ~sib
        rel["same node"] = (intree, same_node & causal)
        rel["sibling node"] = (intree, sib & causal)
        rel["same depth, elsewhere"] = (intree, same_depth & causal)
        if setting == "py":
            # key's node is a strict ancestor of the query's node
            M = ids.T.gather(1, d[None].expand(T, T).clamp(min=0))  # M[i,j] = ids[d_j, i]
            anc = (M == node[None]) & (d[None] < d[:, None]) & (d[None] > 0)
            rel["ancestor"] = (intree, anc & causal)
        else:
            top2 = ids[2]
            cross = (top2[:, None] != top2[None]) & (d[:, None] >= 2) & (d[None] >= 2)
            rel["other branch"] = (intree, cross & causal)
    return rel


def capture(model):
    """Register pre-hooks storing (x, rot) for every attention module."""
    store = []
    hooks = []
    for blk in model.transformer.h:
        def pre(mod, args, _s=store):
            x = args[0]
            rot = args[1] if len(args) > 1 else None
            _s.append((x.detach(), rot))
        hooks.append(blk.attn.register_forward_pre_hook(pre))
    return store, hooks


def qk(attn, x, rot):
    import torch
    from rope_pos import apply_rotary
    B, T, C = x.shape
    q, k, _ = attn.c_attn(x.float()).split(attn.n_embd, dim=2)
    q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    if rot is not None:
        cos, sin = rot
        q, k = apply_rotary(q, cos.float(), sin.float()), apply_rotary(k, cos.float(), sin.float())
    return q, k


def plane_groups(model, pos, T, device):
    """Which planes are sequence / structure / unrotated for this model, from its own angles."""
    import torch
    rope = model.rope
    if rope is None:
        return None
    path = pos["path"] if pos is not None else None
    a = rope.angles(T, path, device)  # (B,T,64)
    nseq = rope.seq_planes if rope.use_seq else 0
    varies = (a[..., nseq:].abs().amax(dim=(0, 1)) > 0)
    struct = torch.zeros(a.shape[-1], dtype=torch.bool, device=device)
    struct[nseq:] = varies
    seq = torch.zeros_like(struct)
    seq[:nseq] = True
    return seq, struct, a


def cmd_attn(setting):
    import torch
    cfg = SETTINGS[setting]
    dev = torch.device("cuda")
    from topos_data import PackedWindows
    base = PackedWindows(cfg["base_pack"], need_tree=False, sidecar="path",
                         need_ref=(setting == "sheets"), need_ref_sheet=(setting == "sheets"))
    lo, hi = base.splits["test"]
    recs = list(range(lo, min(hi, lo + cfg["attn_records"])))
    out_path = os.path.join(OUT, f"attn_{setting}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    example = {}
    for seed in SEEDS:
        for arm in [cfg["seq"]] + cfg["attn_arms"]:
            key = f"{arm}_s{seed}"
            if key in res:
                continue
            rd = run_dir(setting, arm, seed)
            model, margs, pack = load_model(rd, cfg["final"], dev)
            pw = windows(pack, margs)
            assert pw.splits["test"] == base.splits["test"], "test split differs from the base pack"
            L, H = margs["n_layer"], margs["n_head"]
            acc = {}  # rel -> [L][H] sums of mass, chance, count
            split = [[dict(tot=[], dstruct=[], seqc=[], structc=[]) for _ in range(H)] for _ in range(L)]
            store, hooks = capture(model)
            for r in recs:
                X, Y, POS = pw.get_batch_records([r], dev)
                _, _, BPOS = base.get_batch_records([r], dev)
                assert torch.equal(X, base.get_batch_records([r], dev)[0]), "token streams differ"
                store.clear()
                with torch.no_grad():
                    model(X, Y, pos=POS)
                T = X.shape[1]
                bpath = BPOS["path"][0]
                bref = BPOS.get("ref")[0] if BPOS.get("ref") is not None else None
                rels = relations(setting, bpath, bref)
                causal_cnt = torch.arange(T, device=dev).float()  # keys strictly before i
                groups = plane_groups(model, POS, T, dev)
                with torch.no_grad():
                    for li, (x, rot) in enumerate(store):
                        q, k = qk(model.transformer.h[li].attn, x, rot)
                        hd = q.shape[-1]
                        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
                        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
                        att = att.masked_fill(mask, float("-inf")).softmax(-1)[0]  # (H,T,T)
                        for name, (qm, km) in rels.items():
                            cnt = km.sum(-1).float()
                            use = qm & (cnt > 0) & (causal_cnt > 0)
                            if not use.any():
                                continue
                            mass = (att[:, use] * km[use][None]).sum(-1)  # (H, nq)
                            chance = (cnt[use] / (causal_cnt[use] + 1))  # keys incl. self
                            a = acc.setdefault(name, [[[0.0, 0.0, 0] for _ in range(H)] for _ in range(L)])
                            for h in range(H):
                                a[li][h][0] += float(mass[h].sum())
                                a[li][h][1] += float(chance.sum())
                                a[li][h][2] += int(use.sum())
                        if groups is not None and groups[1].any():
                            seqm, structm, ang = groups
                            # q,k already rotated; recompute with structure planes UNrotated
                            from rope_pos import apply_rotary
                            Bq = x.shape[0]
                            qq, kk, _ = model.transformer.h[li].attn.c_attn(x.float()).split(model.transformer.h[li].attn.n_embd, dim=2)
                            qq = qq.view(Bq, T, H, hd).transpose(1, 2)
                            kk = kk.view(Bq, T, H, hd).transpose(1, 2)
                            a0 = ang.clone()
                            a0[..., structm] = 0
                            q0 = apply_rotary(qq, a0.cos(), a0.sin())
                            k0 = apply_rotary(kk, a0.cos(), a0.sin())
                            idx = torch.tril_indices(T, T, -1, device=dev)
                            sel = torch.randperm(idx.shape[1], device=dev)[:20000]
                            ii, jj = idx[0, sel], idx[1, sel]
                            half = hd // 2
                            for h in range(H):
                                def contrib(qa, ka, planes):
                                    pm = torch.cat([planes, planes])
                                    return (qa[0, h, ii][:, pm] * ka[0, h, jj][:, pm]).sum(-1) / math.sqrt(hd)
                                tot = (q[0, h, ii] * k[0, h, jj]).sum(-1) / math.sqrt(hd)
                                rot_s = contrib(q, k, structm)
                                unrot_s = contrib(q0, k0, structm)
                                seqc = contrib(q, k, seqm)
                                sp = split[li][h]
                                sp["tot"].append(float(tot.std()))
                                sp["dstruct"].append(float((rot_s - unrot_s).std()))
                                sp["seqc"].append(float(seqc.std()))
                                sp["structc"].append(float(rot_s.std()))
                            del q0, k0, half
                        del att
            for hk in hooks:
                hk.remove()
            res[key] = dict(
                rel={n: [[dict(mass=v[0] / max(v[2], 1), chance=v[1] / max(v[2], 1), n=v[2]) for v in row] for row in a]
                     for n, a in acc.items()},
                split=[[{k2: (float(np.mean(v)) if v else None) for k2, v in sp.items()} for sp in row] for row in split])
            print(setting, key, "done", flush=True)
            os.makedirs(OUT, exist_ok=True)
            json.dump(res, open(out_path, "w"), indent=1)
            del model
            torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ worked example

def cmd_example(setting, seed=0, rec_offset=0, which=None):
    """Attention of one query (a row-ending/cell-start or a block start) in seq vs structured."""
    import torch
    cfg = SETTINGS[setting]
    dev = torch.device("cuda")
    from topos_data import PackedWindows
    base = PackedWindows(cfg["base_pack"], need_tree=False, sidecar="path",
                         need_ref=(setting == "sheets"), need_ref_sheet=(setting == "sheets"))
    lo, _ = base.splits["test"]
    r = lo + rec_offset
    out = {}
    for arm in [cfg["seq"], which or cfg["attn_arms"][-1]]:
        model, margs, pack = load_model(run_dir(setting, arm, seed), cfg["final"], dev)
        pw = windows(pack, margs)
        X, Y, POS = pw.get_batch_records([r], dev)
        store, hooks = capture(model)
        with torch.no_grad():
            model(X, Y, pos=POS)
        for hk in hooks:
            hk.remove()
        T = X.shape[1]
        maps = []
        for li, (x, rot) in enumerate(store):
            q, k = qk(model.transformer.h[li].attn, x, rot)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
            maps.append(att.masked_fill(mask, float("-inf")).softmax(-1)[0].cpu().numpy())
        out[arm] = np.stack(maps)  # (L,H,T,T)
    _, _, BPOS = base.get_batch_records([r], dev)
    np.savez_compressed(os.path.join(OUT, f"example_{setting}.npz"), tokens=X[0].cpu().numpy(),
                        path=BPOS["path"][0].cpu().numpy(), **{k: v.astype(np.float16) for k, v in out.items()})
    print("wrote example", setting)


# ------------------------------------------------------------------ 4. steps

def cmd_steps():
    import torch
    rows = []
    for setting, cfg in SETTINGS.items():
        for arm in cfg["arms"]:
            for seed in SEEDS:
                rd = run_dir(setting, arm, seed)
                if not os.path.exists(os.path.join(rd, "ckpt.pt")):
                    continue
                ck = torch.load(os.path.join(rd, "ckpt.pt"), map_location="cpu", weights_only=False)
                sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
                rec = dict(setting=setting, arm=arm, seed=seed)
                for name in ("rope.step", "rope.theta", "rope.step_rel"):
                    if name in sd:
                        rec[name] = sd[name].float().numpy().tolist()
                rows.append(rec)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "steps.json"), "w"))
    print("steps:", len(rows), "runs")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remove", "attn", "example", "steps"])
    ap.add_argument("--setting", choices=list(SETTINGS))
    ap.add_argument("--rec", type=int, default=0)
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    if a.cmd == "remove":
        cmd_remove(a.setting)
    elif a.cmd == "attn":
        cmd_attn(a.setting)
    elif a.cmd == "example":
        cmd_example(a.setting, rec_offset=a.rec, which=a.arm)
    else:
        cmd_steps()


if __name__ == "__main__":
    main()
