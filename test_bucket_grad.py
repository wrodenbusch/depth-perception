"""Bucketed accumulation must give the same gradient as one-record-at-a-time.

The whole point of the change is that GROUPING records differently must not change
the result. The reference here is the unambiguous one: every record forward-and-
backward on its own, each weighted by its share of the step's target tokens. If the
bucketed path matches that, padding and grouping are provably not affecting the
gradient (up to floating-point reassociation).
"""
import numpy as np
import torch

from model import GPT, GPTConfig
from topos_data import PackedWindows


def _pack(tmp_path, n=12):
    import json
    d = tmp_path / "pk"; s = tmp_path / "pk_dompath"; d.mkdir(); s.mkdir()
    rng = np.random.default_rng(0)
    toks, off, tn, npth, noff, tw = [], [0], [], [], [0], []
    for i in range(n):
        L = int(rng.integers(20, 140))                 # very uneven lengths
        body = rng.integers(1, 200, size=L).astype(np.uint16)
        ids = np.concatenate([[255], body, [255]]).astype(np.uint16)
        nodes = max(2, L // 5)
        toks.append(ids); off.append(off[-1] + len(ids))
        t = np.concatenate([[-1], rng.integers(0, nodes, size=L), [-1]]).astype(np.int32)
        tn.append(t)
        p = np.full((nodes, 22), -1, dtype=np.int16)
        p[:, 0] = rng.integers(0, 5, size=nodes)
        npth.append(p); noff.append(noff[-1] + nodes)
        tw.append(np.concatenate([[-1], rng.integers(0, 9, size=L), [-1]]).astype(np.int16))
    np.save(d / "tokens.npy", np.concatenate(toks))
    np.save(d / "tok_off.npy", np.array(off, dtype=np.int64))
    np.save(s / "tok_node.npy", np.concatenate(tn))
    np.save(s / "node_path.npy", np.concatenate(npth))
    np.save(s / "node_off.npy", np.array(noff, dtype=np.int64))
    np.save(s / "tok_within.npy", np.concatenate(tw))
    json.dump(dict(block_size=256, vocab_size=256, eot=255, n_train=n - 4, n_val=2, n_test=2),
              open(d / "meta.json", "w"))
    return str(d)


def _grads(model, batches, total):
    model.zero_grad(set_to_none=True)
    for x, y, pos, n in batches:
        _, loss = model(x, y, pos=pos)
        (loss * (n / total)).backward()
    return {k: v.grad.detach().clone() for k, v in model.named_parameters() if v.grad is not None}


def test_bucketed_gradient_matches_per_record(tmp_path):
    pw = PackedWindows(_pack(tmp_path), sidecar="path")
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=256, vocab_size=256, n_layer=1, n_head=1, n_embd=128, bias=False,
                    dropout=0.0, pos_mode="rope_leaf", rope_seq_planes=16, rope_levels=22,
                    rope_planes_per_level=2, rope_token_planes=4, rope_decay=False)
    model = GPT(cfg).double()
    dev = torch.device("cpu")

    groups, total = pw.step_batches("train", 8, token_budget=512,
                                    generator=torch.Generator().manual_seed(3))
    flat = [r for g in groups for r in g]

    def mk(gs):
        out = []
        for g in gs:
            x, y, p = pw.get_batch_records(g, dev)
            out.append((x, y, p, int((y != -1).sum())))
        return out

    bucketed = _grads(model, mk(groups), total)
    per_record = _grads(model, mk([[r] for r in flat]), total)

    assert bucketed.keys() == per_record.keys()
    worst = max((bucketed[k] - per_record[k]).abs().max().item() for k in bucketed)
    scale = max(per_record[k].abs().max().item() for k in per_record)
    assert worst / scale < 1e-9, f"relative gradient mismatch {worst/scale:.2e}"


def test_fixed_accumulation_does_not_match(tmp_path):
    """The old 1/n_micro weighting really is different -- so this is a real change,
    not a no-op dressed up as one."""
    pw = PackedWindows(_pack(tmp_path), sidecar="path")
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=256, vocab_size=256, n_layer=1, n_head=1, n_embd=128, bias=False,
                    dropout=0.0, pos_mode="rope_seq", rope_seq_planes=16, rope_levels=22,
                    rope_planes_per_level=2, rope_token_planes=4)
    model = GPT(cfg).double()
    dev = torch.device("cpu")
    groups, total = pw.step_batches("train", 8, token_budget=512,
                                    generator=torch.Generator().manual_seed(3))
    mk = lambda gs: [(*pw.get_batch_records(g, dev), 0) for g in gs]
    tok = lambda g: int((pw.get_batch_records(g, dev)[1] != -1).sum())
    weighted = _grads(model, [(x, y, p, tok(g)) for (x, y, p, _), g in zip(mk(groups), groups)], total)
    model.zero_grad(set_to_none=True)
    for g in groups:
        x, y, p = pw.get_batch_records(g, dev)
        _, loss = model(x, y, pos=p)
        (loss / len(groups)).backward()
    fixed = {k: v.grad.detach().clone() for k, v in model.named_parameters() if v.grad is not None}
    worst = max((weighted[k] - fixed[k]).abs().max().item() for k in weighted)
    assert worst > 1e-6, "expected the two weightings to differ"
