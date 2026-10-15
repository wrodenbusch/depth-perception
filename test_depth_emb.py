"""Tests for the depth-as-input-embedding control (GPTConfig.depth_emb).

The control is only meaningful if it carries EXACTLY the information the rotary depth marker
carries and starts from exactly the sequence-only baseline. These pin both.
"""
import pytest
import torch

from model import GPT, GPTConfig
from rope_pos import ABSENT, GraphRoPE

LEVELS = 22


def cfg(**kw):
    base = dict(block_size=64, vocab_size=128, n_layer=2, n_head=1, n_embd=128, bias=False,
                dropout=0.0, pos_mode="rope_seq", rope_seq_planes=16, rope_levels=LEVELS,
                rope_planes_per_level=2, rope_token_planes=4, rope_decay=False)
    base.update(kw)
    return GPTConfig(**base)


def path_batch(b=2, t=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = torch.randint(0, LEVELS + 1, (b, t), generator=g)
    p = torch.full((b, t, LEVELS), ABSENT, dtype=torch.long)
    for i in range(b):
        for j in range(t):
            p[i, j, : depth[i, j]] = torch.randint(0, 9, (int(depth[i, j]),), generator=g)
    return p, depth


def test_starts_as_rope_seq():
    """Zero-initialised table: identical logits to rope_seq at the same seed."""
    idx = torch.randint(0, 128, (2, 32), generator=torch.Generator().manual_seed(1))
    path, _ = path_batch()
    torch.manual_seed(0)
    base = GPT(cfg())
    torch.manual_seed(0)
    ctrl = GPT(cfg(depth_emb=True))
    assert torch.count_nonzero(ctrl.depth_emb.weight) == 0
    sd = {k: v for k, v in ctrl.state_dict().items() if not k.startswith("depth_emb")}
    assert all(torch.equal(sd[k], v) for k, v in base.state_dict().items())
    a, _ = base(idx, idx, pos={"path": path})
    b, _ = ctrl(idx, idx, pos={"path": path})
    assert torch.equal(a, b)


def test_depth_matches_rope_depth():
    """The table is indexed by the same depth rope_depth multiplies by its step."""
    path, depth = path_batch()
    rope = GraphRoPE(128, "rope_depth", seq_planes=16, levels=LEVELS, planes_per_level=2,
                     rel_planes=2)
    rel = rope._rel_angles(path, path.shape[0], path.shape[1], path.device)
    assert torch.allclose(rel[..., 0] / rope.step_rel[0], depth.float())
    m = GPT(cfg(depth_emb=True))
    with torch.no_grad():
        m.depth_emb.weight.copy_(torch.arange(LEVELS + 1, dtype=torch.float32)[:, None]
                                 .expand(-1, 128))
    seen = {}
    m.depth_emb.register_forward_hook(lambda mod, inp, out: seen.update(d=inp[0]) and None)
    idx = torch.zeros(path.shape[:2], dtype=torch.long)
    m(idx, idx, pos={"path": path})
    assert torch.equal(seen["d"], depth)


def test_learns():
    path, _ = path_batch()
    m = GPT(cfg(depth_emb=True))
    idx = torch.randint(0, 128, path.shape[:2])
    _, loss = m(idx, idx, pos={"path": path})
    loss.backward()
    assert m.depth_emb.weight.grad is not None and m.depth_emb.weight.grad.abs().sum() > 0


def test_refuses_without_path_and_on_other_modes():
    m = GPT(cfg(depth_emb=True))
    idx = torch.zeros((1, 8), dtype=torch.long)
    with pytest.raises(ValueError):
        m(idx, idx, pos=None)
    with pytest.raises(ValueError):
        GPT(cfg(depth_emb=True, pos_mode="rope_both"))
