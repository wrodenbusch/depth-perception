"""Tests for rope_pos.GraphRoPE and the rotary arms in model.py."""
from topos_paths import PACKS, SOURCES  # noqa: E402
import math
import os

import numpy as np

import pytest
import torch

from rope_pos import (ABSENT, BASE_STEP, LADDER_RATIO, PLAN_CHARS, PLAN_MODE, ROPE_MODES,
                      GraphRoPE, apply_rotary, DEPTH_MODE)

HD = 128


def rand_path(b, t, levels=24, depth=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.full((b, t, levels), ABSENT, dtype=torch.long)
    p[:, :, :depth] = torch.randint(0, 20, (b, t, depth), generator=g)
    p[:, 0, :] = ABSENT  # framing token
    return p


def test_modes():
    assert ROPE_MODES == ("rope_seq", "rope_graph", "rope_both", "rope_leaf", "rope_all",
                          "rope_head_split", "rope_layer_split", "rope_layer_plan",
                          "rope_ref_slot", "rope_rel", "rope_ref_path", "rope_ref_anchor",
                          "rope_ref_delta", "rope_depth")


def test_refuses_bad_allocation():
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_both", seq_planes=16, levels=23, planes_per_level=2)


def test_modes_have_expected_parameters():
    assert GraphRoPE(HD, "rope_seq").n_params() == 0
    assert GraphRoPE(HD, "rope_graph").n_params() == 24 * 2 * 2
    assert GraphRoPE(HD, "rope_both").n_params() == 24 * 2 * 2


def test_init_values():
    r = GraphRoPE(HD, "rope_both")
    assert torch.equal(r.theta, torch.zeros(24, 2))
    assert torch.allclose(r.step[:, 0], torch.full((24,), BASE_STEP))
    assert torch.allclose(r.step[:, 1], torch.full((24,), BASE_STEP / LADDER_RATIO))
    assert BASE_STEP == math.sqrt(2 * (1 - 0.95))


def test_seq_matches_reference_rope_over_16_planes():
    r = GraphRoPE(HD, "rope_seq")
    a = r.angles(10, None, "cpu")  # (1, 10, 64)
    assert a.shape == (1, 10, 64)
    pos = torch.arange(10, dtype=torch.float32)
    inv = 10000.0 ** (-torch.arange(16, dtype=torch.float32) / 16)
    assert torch.allclose(a[0, :, :16], pos[:, None] * inv)
    assert torch.equal(a[0, :, 16:], torch.zeros(10, 48))


def test_graph_leaves_seq_planes_at_zero_and_absent_levels_at_zero():
    r = GraphRoPE(HD, "rope_graph")
    p = rand_path(2, 10)
    a = r.angles(10, p, "cpu")
    assert a.shape == (2, 10, 64)
    assert torch.equal(a[:, :, :16], torch.zeros(2, 10, 16))
    assert torch.equal(a[:, 0, :], torch.zeros(2, 64))  # framing row: all planes zero
    assert torch.equal(a[:, 1:, 16 + 2 * 6:], torch.zeros(2, 9, 48 - 12))  # levels 6.. absent


def test_graph_angle_formula():
    r = GraphRoPE(HD, "rope_graph")
    with torch.no_grad():
        r.theta.fill_(0.5)
    p = torch.full((1, 1, 24), ABSENT, dtype=torch.long)
    p[0, 0, 0] = 3
    a = r.angles(1, p, "cpu")[0, 0]
    assert torch.allclose(a[16], torch.tensor(3 * BASE_STEP + 0.5))
    assert torch.allclose(a[17], torch.tensor(3 * BASE_STEP / LADDER_RATIO + 0.5))
    assert torch.equal(a[18:], torch.zeros(46))


def test_both_is_sum_of_the_two():
    p = rand_path(2, 10)
    both = GraphRoPE(HD, "rope_both").angles(10, p, "cpu")
    seq = GraphRoPE(HD, "rope_seq").angles(10, None, "cpu")
    graph = GraphRoPE(HD, "rope_graph").angles(10, p, "cpu")
    assert torch.allclose(both, seq + graph)


def test_graph_modes_require_path():
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_graph").angles(4, None, "cpu")
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_both").angles(4, torch.zeros(1, 4, 23, dtype=torch.long), "cpu")


def test_apply_rotary_preserves_norm_and_dtype():
    r = GraphRoPE(HD, "rope_both")
    cos, sin = r(10, rand_path(2, 10), "cpu")
    x = torch.randn(2, 3, 10, HD, dtype=torch.bfloat16)
    y = apply_rotary(x, cos, sin)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.allclose(y.float().norm(dim=-1), x.float().norm(dim=-1), atol=0.05, rtol=0.02)


def test_apply_rotary_is_identity_at_zero_angle():
    cos, sin = torch.ones(1, 5, 64), torch.zeros(1, 5, 64)
    x = torch.randn(1, 2, 5, HD)
    assert torch.equal(apply_rotary(x, cos, sin), x)


def test_relative_property_on_seq_planes():
    # q.k after rotation depends only on the offset for the sequence planes
    r = GraphRoPE(HD, "rope_seq")
    cos, sin = r(6, None, "cpu")
    q = torch.randn(1, 1, 1, HD).expand(1, 1, 6, HD).contiguous()
    k = torch.randn(1, 1, 1, HD).expand(1, 1, 6, HD).contiguous()
    Q, K = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    s = Q[0, 0] @ K[0, 0].T
    assert torch.allclose(s[2, 0], s[4, 2], atol=1e-4) and torch.allclose(s[3, 1], s[5, 3], atol=1e-4)


def test_building_the_module_does_not_consume_rng():
    torch.manual_seed(3)
    a = torch.randn(4)
    torch.manual_seed(3)
    GraphRoPE(HD, "rope_both")
    b = torch.randn(4)
    assert torch.equal(a, b)


# ---- model.py: the four arms ------------------------------------------------
from model import GPT, GPTConfig  # noqa: E402


def small(mode):
    return GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=256,
                     bias=False, dropout=0.0, pos_mode=mode)


def test_model_refuses_rope_with_wrong_head_width():
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=4, n_embd=256, pos_mode="rope_both")
    with pytest.raises(ValueError):
        GPT(cfg)


def test_rope_arms_build_no_position_table_and_share_the_block_stack():
    sds = {}
    for mode in ("none", "rope_seq", "rope_graph", "rope_both"):
        torch.manual_seed(7)
        m = GPT(small(mode))
        assert not hasattr(m.transformer, "wpe")
        assert m.tree_pos is None
        assert (m.rope is None) == (mode == "none")
        sds[mode] = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith("rope.")}
    for mode in ("rope_seq", "rope_graph", "rope_both"):
        assert sds[mode].keys() == sds["none"].keys()
        for k in sds["none"]:
            assert torch.equal(sds[mode][k], sds["none"][k]), k


def test_rope_parameter_counts():
    assert GPT(small("rope_seq")).rope.n_params() == 0
    assert GPT(small("rope_both")).rope.n_params() == 96
    # rope parameters are the mechanism under test and are NOT subtracted
    assert GPT(small("rope_both")).get_num_params() == GPT(small("none")).get_num_params() + 96


def test_rope_forward_runs_and_graph_needs_path():
    idx = torch.randint(0, 256, (2, 16))
    path = rand_path(2, 16)
    for mode in ("rope_seq", "rope_graph", "rope_both"):
        torch.manual_seed(0)
        m = GPT(small(mode))
        logits, loss = m(idx, idx, pos={"path": path})
        assert logits.shape == (2, 16, 256) and torch.isfinite(loss)
    with pytest.raises(ValueError):
        GPT(small("rope_graph"))(idx, idx, pos=None)
    with pytest.raises(ValueError):
        GPT(small("rope_graph"))(idx, idx, pos={"path": None})


def test_rope_seq_is_order_dependent_where_none_is_not():
    # One layer: under 'none' the output at position 2 is a symmetric function of
    # the SET of tokens 0..2 (attention over a set), so swapping tokens 0 and 1
    # changes nothing there. Under rope_seq the angles differ by offset, so it does.
    def one_layer(mode):
        return GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2, n_embd=256,
                         bias=False, dropout=0.0, pos_mode=mode)
    a = torch.tensor([[3, 9, 5, 7]])
    b = torch.tensor([[9, 3, 5, 7]])
    torch.manual_seed(0)
    n = GPT(one_layer("none")).eval()
    ln_a, _ = n(a, a); ln_b, _ = n(b, b)
    assert torch.allclose(ln_a[0, 2], ln_b[0, 2], atol=1e-5)
    torch.manual_seed(0)
    s = GPT(one_layer("rope_seq")).eval()
    ls_a, _ = s(a, a); ls_b, _ = s(b, b)
    assert not torch.allclose(ls_a[0, 2], ls_b[0, 2], atol=1e-3)


def test_rope_seq_ignores_a_path_and_none_ignores_pos():
    idx = torch.randint(0, 256, (1, 8))
    torch.manual_seed(0)
    m = GPT(small("rope_seq")).eval()
    a, _ = m(idx, idx, pos=None)
    b, _ = m(idx, idx, pos={"path": rand_path(1, 8)})
    assert torch.equal(a, b)
    torch.manual_seed(0)
    n = GPT(small("none")).eval()
    a, _ = n(idx, idx, pos=None)
    b, _ = n(idx, idx, pos={"path": rand_path(1, 8)})
    assert torch.equal(a, b)


# ---- the token-leaf level (cell 2) --------------------------------------------

LEAF = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)


def leaf_path(b, t, depth=6, seed=0):
    """(B, t, 23): 22 tree levels + the within-node index in the last column."""
    p = rand_path(b, t, levels=23, depth=depth, seed=seed)
    p[:, 1:, 22] = torch.arange(t - 1) % 7        # a within-node index; framing row stays ABSENT
    return p


def test_leaf_allocation_refusal_counts_token_planes():
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_leaf", seq_planes=16, levels=24, planes_per_level=2, token_planes=4)  # 68
    GraphRoPE(HD, "rope_leaf", **LEAF)                                                            # 64


def test_leaf_parameter_count_and_init():
    r = GraphRoPE(HD, "rope_leaf", **LEAF)
    assert r.n_params() == 22 * 2 * 2 + 4 * 2
    assert torch.equal(r.theta_tok, torch.zeros(4))
    assert torch.allclose(r.step_tok, torch.tensor([BASE_STEP / 4**p for p in range(4)]))
    # existing modes: unchanged counts, no token parameters
    assert GraphRoPE(HD, "rope_seq", **LEAF).n_params() == 0
    assert GraphRoPE(HD, "rope_graph", **LEAF).n_params() == 22 * 2 * 2
    assert GraphRoPE(HD, "rope_both", **LEAF).n_params() == 22 * 2 * 2


def test_leaf_angle_formula_on_one_token():
    r = GraphRoPE(HD, "rope_leaf", **LEAF)
    with torch.no_grad():
        r.theta_tok.fill_(0.25)
    p = torch.full((1, 1, 23), ABSENT, dtype=torch.long)
    p[0, 0, 0] = 2          # rank 2 at level 0
    p[0, 0, 22] = 5         # 6th token of its node
    a = r.angles(1, p, "cpu")[0, 0]
    assert torch.equal(a[:16], torch.zeros(16))                              # sequence planes at 0
    assert torch.allclose(a[16], torch.tensor(2 * BASE_STEP))                # level 0 fast plane
    assert torch.allclose(a[60:64], 5 * torch.tensor([BASE_STEP / 4**k for k in range(4)]) + 0.25)


def test_leaf_absent_token_column_gives_zero_token_planes():
    r = GraphRoPE(HD, "rope_leaf", **LEAF)
    p = leaf_path(2, 8)
    a = r.angles(8, p, "cpu")
    assert torch.equal(a[:, 0, :], torch.zeros(2, 64))                       # framing: all zero
    assert torch.equal(a[:, :, :16], torch.zeros(2, 8, 16))


def test_graph_and_both_ignore_the_token_column():
    p23 = leaf_path(2, 8)
    p22 = p23[..., :22]
    for mode in ("rope_graph", "rope_both"):
        r = GraphRoPE(HD, mode, **LEAF)
        a23, a22 = r.angles(8, p23, "cpu"), r.angles(8, p22, "cpu")
        assert torch.equal(a23, a22)
        assert torch.equal(a23[..., 60:64], torch.zeros(2, 8, 4))            # token planes at 0


def test_leaf_requires_the_token_column():
    r = GraphRoPE(HD, "rope_leaf", **LEAF)
    with pytest.raises(ValueError):
        r.angles(8, leaf_path(2, 8)[..., :22], "cpu")


def test_leaf_model_builds_and_shares_the_block_stack():
    from model import GPT, GPTConfig
    def cfg(mode):
        return GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=256, bias=False, dropout=0.0,
                         pos_mode=mode, rope_seq_planes=16, rope_levels=22, rope_planes_per_level=2, rope_token_planes=4)
    torch.manual_seed(7); m_none = GPT(cfg("none"))
    torch.manual_seed(7); m_leaf = GPT(cfg("rope_leaf"))
    sd_n = m_none.state_dict()
    for k, v in m_leaf.state_dict().items():
        if not k.startswith("rope."):
            assert torch.equal(v, sd_n[k]), k
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = m_leaf(idx, idx, pos={"path": leaf_path(2, 16)})
    assert logits.shape == (2, 16, 256) and torch.isfinite(loss)
    with pytest.raises(ValueError):
        GPT(GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2, n_embd=256, pos_mode="rope_leaf",
                      rope_levels=24, rope_token_planes=4))


def _opt_groups(mode, rope_decay):
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=2, n_embd=256, bias=False,
                    pos_mode=mode, rope_seq_planes=16, rope_levels=22, rope_planes_per_level=2,
                    rope_token_planes=4, rope_decay=rope_decay)
    m = GPT(cfg)
    opt = m.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu")
    rope_ids = {id(p) for p in m.rope.parameters()} if m.rope is not None else set()
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    return rope_ids, decayed


def test_rope_decay_flag_moves_only_the_rotation_tensors():
    # Default True reproduces the banked cell-2 runs: the 2-D rotation tables are decayed.
    rope_ids, decayed = _opt_groups("rope_leaf", True)
    assert len(rope_ids & decayed) == 2, "step and theta should be in the decay group by default"
    # False takes every rotation tensor out of the decay group and touches nothing else.
    rope_ids2, decayed2 = _opt_groups("rope_leaf", False)
    assert not (rope_ids2 & decayed2)
    assert len(decayed) - len(decayed2) == 2


def test_rope_decay_flag_is_a_no_op_without_rotation_parameters():
    for mode in ("none", "rope_seq"):
        # Two model instances, so compare group SIZES -- parameter ids differ by construction.
        a, b = _opt_groups(mode, True)[1], _opt_groups(mode, False)[1]
        assert len(a) == len(b), f"{mode} has no rotation parameters; the flag must change nothing"


def test_apply_rotary_matches_the_float32_reference_within_bf16_tolerance():
    """The 16-bit rotation must agree with the old float32-promoted one.

    Reference = the pre-2026-09-24 implementation, written out here so the change is
    pinned rather than trusted.
    """
    def reference(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half].float(), x[..., half:].float()
        c, s = cos[:, None], sin[:, None]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)

    torch.manual_seed(0)
    r = GraphRoPE(HD, "rope_both")
    cos, sin = r(64, rand_path(3, 64), "cpu")
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 3e-2), (torch.float16, 3e-3)):
        x = torch.randn(3, 4, 64, HD, dtype=dtype)
        got, want = apply_rotary(x, cos, sin), reference(x, cos, sin)
        assert got.dtype == x.dtype
        err = (got.float() - want.float()).abs().max().item()
        scale = want.float().abs().max().item()
        assert err / scale < tol, f"{dtype}: relative error {err/scale:.2e} exceeds {tol}"


def test_apply_rotary_preserves_norm_in_float32():
    """A rotation is norm-preserving; in float32 that must hold tightly."""
    torch.manual_seed(0)
    r = GraphRoPE(HD, "rope_both")
    cos, sin = r(32, rand_path(2, 32), "cpu")
    x = torch.randn(2, 3, 32, HD)
    y = apply_rotary(x, cos, sin)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4)


def test_rope_all_is_the_superset_of_every_other_arm():
    """rope_all must rotate sequence, tree AND token planes -- and its angles must equal
    the sum of what rope_both and rope_leaf each contribute, since the plane groups are
    disjoint. This is the arm that makes 'more information' a nested comparison."""
    torch.manual_seed(0)
    kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
    p = rand_path(2, 12, levels=22)
    p = torch.cat([p, torch.randint(0, 5, (2, 12, 1))], dim=-1)   # append the within-node column
    a_all = GraphRoPE(HD, "rope_all", **kw)
    a_both = GraphRoPE(HD, "rope_both", **kw)
    a_leaf = GraphRoPE(HD, "rope_leaf", **kw)
    for m in (a_both, a_leaf):                      # share the learned tables so the sum is exact
        m.step.data.copy_(a_all.step.data); m.theta.data.copy_(a_all.theta.data)
        if hasattr(m, "step_tok"):
            m.step_tok.data.copy_(a_all.step_tok.data); m.theta_tok.data.copy_(a_all.theta_tok.data)
    A = a_all.angles(12, p, "cpu")
    assert not torch.equal(A[..., :16], torch.zeros_like(A[..., :16])), "sequence planes must rotate"
    assert not torch.equal(A[..., 16:60], torch.zeros_like(A[..., 16:60])), "tree planes must rotate"
    assert not torch.equal(A[..., 60:], torch.zeros_like(A[..., 60:])), "token planes must rotate"
    B, L = a_both.angles(12, p, "cpu"), a_leaf.angles(12, p, "cpu")
    assert torch.equal(B[..., 60:], torch.zeros_like(B[..., 60:])), "rope_both must NOT rotate token planes"
    assert torch.equal(L[..., :16], torch.zeros_like(L[..., :16])), "rope_leaf must NOT rotate sequence planes"
    # disjoint groups: all = seq(from both) + tree(shared) + token(from leaf)
    expect = torch.cat([B[..., :16], B[..., 16:60], L[..., 60:]], dim=-1)
    assert torch.allclose(A, expect, atol=1e-6)


# ---------------------------------------------------------------------------
# The SPLIT arms (2026-09-26): a whole head or a whole layer per factor, at full
# width, instead of two narrow bands sharing one head.
# ---------------------------------------------------------------------------

SPLIT = dict(seq_planes=64, levels=22, planes_per_level=2, token_planes=4)


def hand_angles(m, t, path):
    """Recompute a PRE-SPLIT arm's angles from first principles.

    Deliberately does not call ``angles()``: this is the regression guard that the
    split work did not perturb the arms already in flight. Written as explicit loops
    so it shares no code, and no broadcasting bugs, with the implementation.
    """
    b = path.shape[0] if path is not None else 1
    out = torch.zeros(b, t, m.n_planes, dtype=torch.float32)
    if m.mode in ("rope_seq", "rope_both", "rope_all"):
        for i in range(m.seq_planes):
            freq = 10000.0 ** (-i / m.seq_planes)
            for tt in range(t):
                out[:, tt, i] = tt * freq
    if m.mode in ("rope_graph", "rope_both", "rope_leaf", "rope_all"):
        for k in range(m.levels):
            r = path[..., k]
            pres = (r >= 0).to(torch.float32)
            rank = r.clamp(min=0).to(torch.float32)
            for pp in range(m.ppl):
                out[:, :, m.seq_planes + k * m.ppl + pp] = rank * m.step[k, pp] + pres * m.theta[k, pp]
    if m.mode in ("rope_leaf", "rope_all"):
        w = path[..., m.levels]
        pres = (w >= 0).to(torch.float32)
        wi = w.clamp(min=0).to(torch.float32)
        for pp in range(m.token_planes):
            out[:, :, m.seq_planes + m.levels * m.ppl + pp] = wi * m.step_tok[pp] + pres * m.theta_tok[pp]
    return out


@pytest.mark.parametrize("mode", ["rope_seq", "rope_graph", "rope_both", "rope_leaf", "rope_all"])
def test_split_work_left_the_existing_arms_unchanged(mode):
    """REGRESSION: the five arms that predate the split arms must be numerically
    untouched, because runs using them are live and a stalled job re-imports this file."""
    torch.manual_seed(3)
    m = GraphRoPE(HD, mode, **LEAF)
    with torch.no_grad():                      # move the tables off their init so a
        for p in m.parameters():               # zero-valued bug cannot pass by luck
            p.add_(torch.randn_like(p) * 0.1)
    p = leaf_path(2, 9)
    got = m.angles(9, p, "cpu")
    assert got.shape == (2, 9, 64)
    torch.testing.assert_close(got, hand_angles(m, 9, p), rtol=1e-6, atol=1e-7)


def test_apply_rotary_3d_path_is_unchanged_by_the_4d_branch():
    """The 3-D cos/sin path is the hot path for every pre-existing arm. Compare it to a
    hand-written rotate-half, and confirm the 4-D branch agrees when all heads share angles."""
    torch.manual_seed(5)
    x = torch.randn(2, 4, 7, HD)
    cos, sin = torch.randn(2, 7, HD // 2).cos(), torch.randn(2, 7, HD // 2).sin()
    half = HD // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[:, None], sin[:, None]
    want = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    assert torch.equal(apply_rotary(x, cos, sin), want)
    # broadcasting the same angles to 4-D must give the identical result
    cos4 = cos[:, None].expand(2, 4, 7, half).contiguous()
    sin4 = sin[:, None].expand(2, 4, 7, half).contiguous()
    assert torch.equal(apply_rotary(x, cos4, sin4), want)


def test_split_arms_refuse_a_wrong_allocation():
    # the sequence factor spans the WHOLE head, so seq_planes must be head_dim/2
    with pytest.raises(ValueError, match="FULL head"):
        GraphRoPE(HD, "rope_layer_split", seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
    # the tree must fit inside one head
    with pytest.raises(ValueError, match="exceeds"):
        GraphRoPE(HD, "rope_layer_split", seq_planes=64, levels=31, planes_per_level=2, token_planes=4)
    # every token its own leaf is the point of these arms
    with pytest.raises(ValueError, match="token_planes"):
        GraphRoPE(HD, "rope_layer_split", seq_planes=64, levels=22, planes_per_level=2, token_planes=0)
    # head_split needs to know, and be able to halve, the head count
    with pytest.raises(ValueError, match="n_head > 0"):
        GraphRoPE(HD, "rope_head_split", **SPLIT)
    with pytest.raises(ValueError, match="even n_head"):
        GraphRoPE(HD, "rope_head_split", n_head=3, **SPLIT)
    GraphRoPE(HD, "rope_head_split", n_head=4, **SPLIT)      # 48 of 64 rotated, 16 idle
    GraphRoPE(HD, "rope_layer_split", **SPLIT)


def test_head_split_shape_and_per_head_factors():
    """Heads 0,1 must carry sequence and NO tree; heads 2,3 the tree and NO sequence."""
    torch.manual_seed(1)
    m = GraphRoPE(HD, "rope_head_split", n_head=4, **SPLIT)
    with torch.no_grad():
        m.theta.add_(0.3); m.theta_tok.add_(0.3)     # off zero so 'present' is visible
    p = leaf_path(2, 10)
    a = m.angles(10, p, "cpu")
    assert a.shape == (2, 4, 10, 64), a.shape
    seq, tree = a[:, :2], a[:, 2:]
    assert torch.equal(seq[:, 0], seq[:, 1]), "the two sequence heads must agree"
    assert torch.equal(tree[:, 0], tree[:, 1]), "the two tree heads must agree"
    assert not torch.allclose(seq[:, 0], tree[:, 0]), "sequence and tree heads must differ"
    # a sequence head is position-only: token 0 is the origin, so all its angles are 0
    assert torch.equal(seq[:, 0, 0], torch.zeros(2, 64))
    # and it is strictly increasing in t on the fastest plane, with no path dependence
    assert torch.all(seq[0, 0, 1:, 0] > seq[0, 0, :-1, 0])
    assert torch.equal(seq[0], seq[1]), "sequence heads must ignore the batch's paths"
    # a tree head leaves the top 16 planes unrotated and rotates the low 48
    assert torch.equal(tree[..., 48:], torch.zeros_like(tree[..., 48:]))
    assert not torch.allclose(tree[..., :48], torch.zeros_like(tree[..., :48]))


def test_head_split_tree_heads_match_a_full_width_leaf_computation():
    """The tree half must be exactly rope_leaf's arithmetic, only laid out from plane 0."""
    torch.manual_seed(2)
    m = GraphRoPE(HD, "rope_head_split", n_head=4, **SPLIT)
    with torch.no_grad():
        m.theta.add_(0.2); m.theta_tok.add_(0.1)
    p = leaf_path(1, 8)
    tree = m.angles(8, p, "cpu")[0, 2]                      # (t, 64) from a tree head
    for k in range(22):
        r = p[0, :, k]
        pres = (r >= 0).to(torch.float32)
        rank = r.clamp(min=0).to(torch.float32)
        for pp in range(2):
            want = rank * m.step[k, pp] + pres * m.theta[k, pp]
            torch.testing.assert_close(tree[:, k * 2 + pp], want, rtol=1e-6, atol=1e-7)
    w = p[0, :, 22]
    pres = (w >= 0).to(torch.float32)
    for pp in range(4):
        want = w.clamp(min=0).to(torch.float32) * m.step_tok[pp] + pres * m.theta_tok[pp]
        torch.testing.assert_close(tree[:, 44 + pp], want, rtol=1e-6, atol=1e-7)


def test_layer_split_returns_two_angle_sets_that_differ():
    torch.manual_seed(1)
    m = GraphRoPE(HD, "rope_layer_split", **SPLIT)
    with torch.no_grad():
        m.theta.add_(0.3); m.theta_tok.add_(0.3)
    p = leaf_path(2, 10)
    aseq, atree = m.layer_angles(10, p, "cpu")
    assert aseq.shape == (2, 10, 64) and atree.shape == (2, 10, 64)
    assert not torch.allclose(aseq, atree)
    assert torch.equal(aseq[:, 0], torch.zeros(2, 64)), "sequence origin is angle 0"
    assert torch.equal(atree[..., 48:], torch.zeros_like(atree[..., 48:])), "16 planes stay idle"
    assert torch.equal(aseq[0], aseq[1]), "the sequence set must ignore the paths"
    # forward() hands back two (cos,sin) pairs, in (even, odd) order
    (cs, ss), (ct, st) = m(10, p, "cpu")
    torch.testing.assert_close(cs, aseq.cos()); torch.testing.assert_close(st, atree.sin())
    # and angles() refuses, rather than silently returning one of the two
    with pytest.raises(ValueError, match="two angle sets"):
        m.angles(10, p, "cpu")


def test_layer_split_agrees_with_head_split_on_each_factor():
    """Both arms compute the same two factors; only the distribution differs."""
    torch.manual_seed(4)
    h = GraphRoPE(HD, "rope_head_split", n_head=4, **SPLIT)
    l = GraphRoPE(HD, "rope_layer_split", **SPLIT)
    with torch.no_grad():
        for src, dst in ((h, l),):
            dst.step.copy_(src.step); dst.theta.copy_(src.theta)
            dst.step_tok.copy_(src.step_tok); dst.theta_tok.copy_(src.theta_tok)
        h.theta.add_(0.25); l.theta.add_(0.25)
    p = leaf_path(2, 9)
    a = h.angles(9, p, "cpu")
    aseq, atree = l.layer_angles(9, p, "cpu")
    torch.testing.assert_close(a[:, 0], aseq, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(a[:, 3], atree, rtol=1e-6, atol=1e-7)


def test_split_arms_need_the_token_column():
    for mode, kw in (("rope_head_split", dict(n_head=4)), ("rope_layer_split", {})):
        m = GraphRoPE(HD, mode, **kw, **SPLIT)
        with pytest.raises(ValueError, match="DOM path"):
            m(9, None, "cpu")
        with pytest.raises(ValueError, match="token column"):
            m(9, rand_path(2, 9, levels=22), "cpu")          # 22 columns, needs 23


def test_split_arms_init_draws_no_randomness():
    """Same guarantee the other arms give: building the rope module must not move the
    block stack's init, so arms are comparable at a seed."""
    torch.manual_seed(11); before = torch.randn(4)
    torch.manual_seed(11)
    GraphRoPE(HD, "rope_head_split", n_head=4, **SPLIT)
    GraphRoPE(HD, "rope_layer_split", **SPLIT)
    assert torch.equal(torch.randn(4), before)


@pytest.mark.parametrize("mode", ["rope_head_split", "rope_layer_split"])
def test_split_model_builds_trains_one_step_and_shares_the_block_stack(mode):
    """CPU smoke: forward, backward, finite loss, and the rope tables get gradients."""
    from model import GPT, GPTConfig

    def cfg(m):
        return GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=256,
                         bias=False, dropout=0.0, pos_mode=m, rope_seq_planes=64,
                         rope_levels=22, rope_planes_per_level=2, rope_token_planes=4)

    torch.manual_seed(7); m_none = GPT(cfg("none"))
    torch.manual_seed(7); m = GPT(cfg(mode))
    sd_n = m_none.state_dict()
    for k, v in m.state_dict().items():
        if not k.startswith("rope."):
            assert torch.equal(v, sd_n[k]), k
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = m(idx, idx, pos={"path": leaf_path(2, 16)})
    assert logits.shape == (2, 16, 256) and torch.isfinite(loss)
    loss.backward()
    for name, p in m.rope.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert not torch.equal(p.grad, torch.zeros_like(p.grad)), f"{name} got a zero gradient"


def test_layer_split_model_really_alternates_layers():
    """Layer 0 must see the sequence angles and layer 1 the tree angles. Detect it by
    zeroing the tree tables: only the ODD layers' rotation should collapse."""
    from model import GPT, GPTConfig
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=256, bias=False,
                    dropout=0.0, pos_mode="rope_layer_split", rope_seq_planes=64, rope_levels=22,
                    rope_planes_per_level=2, rope_token_planes=4)
    torch.manual_seed(7); m = GPT(cfg)
    seen = []
    orig = m.transformer.h[0].attn.forward

    def spy(i):
        f = m.transformer.h[i].attn.forward
        def wrapped(x, rot=None):
            seen.append((i, rot[0].clone()))
            return f(x, rot)
        return wrapped
    m.transformer.h[0].attn.forward = spy(0)
    m.transformer.h[1].attn.forward = spy(1)
    idx = torch.randint(0, 256, (2, 16))
    m(idx, idx, pos={"path": leaf_path(2, 16)})
    assert len(seen) == 2 and seen[0][0] == 0 and seen[1][0] == 1
    cos0, cos1 = seen[0][1], seen[1][1]
    assert not torch.allclose(cos0, cos1), "the two layers must receive different angles"
    # layer 0 is the sequence set: its cos at t=0 is all ones (angle 0) on every plane
    assert torch.allclose(cos0[:, 0], torch.ones_like(cos0[:, 0]))
    # layer 1 is the tree set: its top 16 planes are unrotated, so cos == 1 there for all t
    assert torch.allclose(cos1[..., 48:], torch.ones_like(cos1[..., 48:]))
    assert orig is not None


# ---------------------------------------------------------------------------
# rope_layer_plan (2026-09-26): the per-layer generalisation of rope_layer_split.
# ---------------------------------------------------------------------------

PLAN = dict(seq_planes=64, levels=22, planes_per_level=2, token_planes=4)


def plan_rope(plan, **kw):
    k = dict(PLAN, **kw)
    return GraphRoPE(HD, "rope_layer_plan", layer_plan=plan, n_layer=len(plan), **k)


def test_plan_mode_is_registered():
    assert "rope_layer_plan" in ROPE_MODES
    assert PLAN_MODE == "rope_layer_plan"
    assert PLAN_CHARS == ("s", "t", "n")


@pytest.mark.parametrize("plan,n_layer,why", [
    ("ststst", 4, "length must equal n_layer"),
    ("stst",   6, "length must equal n_layer"),
    ("stsx st".replace(" ", ""), 6, "x is not a plan character"),
    ("nnnnnn", 6, "all-n rotates nothing"),
    ("",       6, "empty plan"),
])
def test_plan_refuses_bad_plans(plan, n_layer, why):
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_layer_plan", layer_plan=plan, n_layer=n_layer, **PLAN)


def test_plan_needs_n_layer():
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_layer_plan", layer_plan="ststst", n_layer=0, **PLAN)


def test_plan_refuses_narrow_sequence_band_when_it_uses_sequence():
    """An 's' layer spans the whole head, so rope_seq_planes must say so."""
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_layer_plan", layer_plan="ststst", n_layer=6,
                  seq_planes=16, levels=22, planes_per_level=2, token_planes=4)


def test_plan_builds_only_the_factors_it_names():
    names = lambda m: sorted(n for n, _ in m.named_parameters())
    assert names(plan_rope("ssssss")) == []                      # no tree tables at all
    assert names(plan_rope("tttttt")) == ["step", "step_tok", "theta", "theta_tok"]
    assert names(plan_rope("ststst")) == ["step", "step_tok", "theta", "theta_tok"]
    assert plan_rope("ssssss").use_tree is False
    assert plan_rope("tttttt").use_seq is False
    assert plan_rope("ssttnn").use_seq and plan_rope("ssttnn").use_tree


def test_plan_angles_shapes_and_none_slots():
    m, p = plan_rope("ststst"), leaf_path(2, 9)
    aseq, atree = m.plan_angles(9, p, "cpu")
    assert aseq.shape == (2, 9, 64) and atree.shape == (2, 9, 64)
    only_s, only_t = plan_rope("ssssss"), plan_rope("tttttt")
    assert only_s.plan_angles(9, p, "cpu")[1] is None
    assert only_t.plan_angles(9, p, "cpu")[0] is None


def test_plan_angles_equal_the_layer_split_factors():
    """'ststst' must produce the SAME two factors rope_layer_split does -- the plan is a
    re-parameterisation, not a new computation."""
    torch.manual_seed(11)
    ref = GraphRoPE(HD, "rope_layer_split", **PLAN)
    got = plan_rope("ststst")
    with torch.no_grad():                        # same non-trivial tables in both
        for a, b in zip(ref.parameters(), got.parameters()):
            v = torch.randn_like(a) * 0.1
            a.copy_(v); b.copy_(v)
    p = leaf_path(2, 9)
    rs, rt = ref.layer_angles(9, p, "cpu")
    gs, gt = got.plan_angles(9, p, "cpu")
    torch.testing.assert_close(gs, rs, rtol=0, atol=0)
    torch.testing.assert_close(gt, rt, rtol=0, atol=0)


def test_plan_angles_is_refused_by_the_scalar_accessors():
    m = plan_rope("ststst")
    with pytest.raises(ValueError):
        m.angles(9, leaf_path(2, 9), "cpu")
    with pytest.raises(ValueError):
        m.layer_angles(9, leaf_path(2, 9), "cpu")
    with pytest.raises(ValueError):
        GraphRoPE(HD, "rope_layer_split", **PLAN).plan_angles(9, leaf_path(2, 9), "cpu")


def test_plan_init_draws_no_randomness():
    torch.manual_seed(5); before = torch.randn(4)
    torch.manual_seed(5); plan_rope("ssttnn"); after = torch.randn(4)
    assert torch.equal(before, after)


def plan_model(plan, n_layer=6):
    from model import GPT, GPTConfig
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=n_layer, n_head=2, n_embd=256,
                    bias=False, dropout=0.0, pos_mode="rope_layer_plan", rope_layer_plan=plan,
                    rope_seq_planes=64, rope_levels=22, rope_planes_per_level=2,
                    rope_token_planes=4, rope_decay=False)
    torch.manual_seed(7)
    return GPT(cfg)


@pytest.mark.parametrize("plan", ["ststst", "sssttt", "tttsss", "ssttnn"])
def test_plan_model_trains_one_step_with_live_gradients(plan):
    m = plan_model(plan)
    idx = torch.randint(0, 256, (2, 16))
    _, loss = m(idx, idx, pos={"path": leaf_path(2, 16)})
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.rope.named_parameters():
        assert p.grad is not None, f"{name} got no gradient under plan {plan}"
        assert not torch.equal(p.grad, torch.zeros_like(p.grad)), f"{name} zero grad, plan {plan}"


@pytest.mark.parametrize("plan,expect", [
    ("ststst", [0, 1, 0, 1, 0, 1]),
    ("sssttt", [0, 0, 0, 1, 1, 1]),
    ("tttsss", [1, 1, 1, 0, 0, 0]),
    ("ssttnn", [0, 0, 1, 1, 2, 2]),
])
def test_plan_routes_each_layer_to_the_right_factor(plan, expect):
    """Spy on every block: an 's' layer must get the sequence cos, a 't' layer the tree cos,
    and an 'n' layer must get rot=None so apply_rotary is skipped entirely."""
    m = plan_model(plan)
    seen = []

    def spy(i):
        f = m.transformer.h[i].attn.forward
        def wrapped(x, rot=None):
            seen.append((i, None if rot is None else rot[0].clone()))
            return f(x, rot)
        return wrapped
    for i in range(6):
        m.transformer.h[i].attn.forward = spy(i)
    idx = torch.randint(0, 256, (2, 16))
    m(idx, idx, pos={"path": leaf_path(2, 16)})
    assert [i for i, _ in seen] == list(range(6))
    got_seq, got_tree = None, None
    for (i, cos), want in zip(seen, expect):
        if want == 2:
            assert cos is None, f"layer {i} is 'n' and must receive no rotation"
            continue
        assert cos is not None, f"layer {i} must receive a rotation"
        if want == 0:
            # the sequence set: angle 0 at t=0 on every plane
            assert torch.allclose(cos[:, 0], torch.ones_like(cos[:, 0]))
            got_seq = cos if got_seq is None else got_seq
            assert torch.equal(cos, got_seq)
        else:
            # the tree set: the top 16 planes are unrotated for all t
            assert torch.allclose(cos[..., 48:], torch.ones_like(cos[..., 48:]))
            got_tree = cos if got_tree is None else got_tree
            assert torch.equal(cos, got_tree)
    if 0 in expect and 1 in expect:
        assert not torch.allclose(got_seq, got_tree)


def test_plan_ststst_matches_rope_layer_split_end_to_end():
    """The whole model under plan 'ststst' must equal rope_layer_split bit for bit."""
    from model import GPT, GPTConfig
    base = dict(block_size=64, vocab_size=256, n_layer=6, n_head=2, n_embd=256, bias=False,
                dropout=0.0, rope_seq_planes=64, rope_levels=22, rope_planes_per_level=2,
                rope_token_planes=4, rope_decay=False)
    torch.manual_seed(7); a = GPT(GPTConfig(pos_mode="rope_layer_split", **base))
    torch.manual_seed(7); b = GPT(GPTConfig(pos_mode="rope_layer_plan",
                                            rope_layer_plan="ststst", **base))
    b.load_state_dict(a.state_dict())
    idx = torch.randint(0, 256, (2, 16))
    p = {"path": leaf_path(2, 16)}
    la, lossa = a(idx, idx, pos=p)
    lb, lossb = b(idx, idx, pos=p)
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    torch.testing.assert_close(lossa, lossb, rtol=0, atol=0)


def test_plan_rope_decay_puts_every_rotation_tensor_in_the_nodecay_group():
    m = plan_model("ssttnn")
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    nodecay = {id(p) for p in opt.param_groups[1]["params"]}
    assert opt.param_groups[1]["weight_decay"] == 0.0
    for name, p in m.rope.named_parameters():
        assert id(p) in nodecay, f"{name} was weight-decayed under rope_decay=False"


def test_plan_n_layers_skip_rotation_so_the_tree_cannot_reach_them():
    """An all-'n' tail means the last layers are position-free: perturbing the tree tables
    must not change what layer 4 receives."""
    m = plan_model("ssttnn")
    got = []
    f = m.transformer.h[4].attn.forward
    m.transformer.h[4].attn.forward = lambda x, rot=None: (got.append(rot), f(x, rot))[1]
    idx = torch.randint(0, 256, (2, 16))
    m(idx, idx, pos={"path": leaf_path(2, 16)})
    assert got == [None]


def test_tree_modes_covers_every_arm_that_reads_the_path():
    """REGRESSION for a real outage: train.py and eval_test.py each had a hardcoded
    pos_mode -> sidecar map. Both drifted -- the split arms were missing from train.py's
    and rope_all from eval_test.py's -- so those arms crashed with "needs a DOM path" on
    launch. The set now lives beside the modes, and this test fails if a new arm is added
    without it."""
    from rope_pos import TREE_MODES, plan_needs_path
    path = leaf_path(2, 6)
    for mode in ROPE_MODES:
        if mode == PLAN_MODE:
            continue
        kw = SPLIT if mode in ("rope_head_split", "rope_layer_split") else LEAF
        if mode == "rope_ref_slot":   # 16 seq + 22*2 tree + 2 slot, no token planes
            kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0,
                      slot_planes=2)
        if mode == "rope_rel":        # 16 seq + 22*2 tree + 2 relational, no token planes
            kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0,
                      rel_planes=2)
        if mode == "rope_ref_path":   # 16 seq + 22*2 tree + 4 grid-pointer, no token planes
            kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0,
                      ref_planes=4)
        if mode == DEPTH_MODE:        # 16 seq + 2 depth, NO tree planes, no token planes
            kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0,
                      rel_planes=2)
        m = GraphRoPE(HD, mode, n_head=4, **kw)
        # TREE_MODES means "needs the path sidecar", which is NOT the same as "rotates the
        # per-level tree planes". rope_depth is the case that separates them: it reads the path
        # to count depth and rotates no per-level plane at all. So the contract asserted here is
        # the one the loader actually depends on -- needing the path -- and use_tree is checked
        # only for the arms where the two coincide.
        if mode == DEPTH_MODE:
            assert mode in TREE_MODES, (
                "rope_depth counts depth out of the path, so it must be in TREE_MODES or the "
                "loader will hand it no sidecar and it will crash on launch"
            )
            assert m.use_tree is False, (
                "rope_depth must NOT rotate the per-level planes -- dropping them is the arm"
            )
            with pytest.raises(ValueError):
                m(6, None, "cpu")   # needs the path, even though it rotates no tree plane
            continue
        assert m.use_tree == (mode in TREE_MODES), (
            f"{mode}: use_tree={m.use_tree} but TREE_MODES membership is {mode in TREE_MODES}"
        )
        if not m.use_tree:          # must not need the path at all
            m(6, None, "cpu")
    assert plan_needs_path("ststst") and plan_needs_path("ssttnn")
    assert plan_needs_path("tttttt") and not plan_needs_path("ssssss")
    assert not plan_needs_path("") and not plan_needs_path(None)


def test_eval_test_sidecar_matches_tree_modes():
    from eval_test import sidecar_for
    from rope_pos import TREE_MODES
    assert sidecar_for("tree") == "tree"
    assert sidecar_for("none") is None and sidecar_for("rope_seq") is None
    for mode in TREE_MODES:
        assert sidecar_for(mode) == "path", f"{mode} must load the path sidecar"
    assert sidecar_for(PLAN_MODE, "ststst") == "path"
    assert sidecar_for(PLAN_MODE, "ssssss") is None


# --------------------------------------------------------------------------------------
# THE SYMBOL SLOT (2026-09-28): rope_ref_slot
#
# The slot is the first coordinate here that is not a property of the token alone -- it
# says which SYMBOL a token refers to, so all mentions of one symbol share an angle and
# become angularly coincident. These tests pin that coincidence property, the ABSENT
# behaviour, and the refusals.
# --------------------------------------------------------------------------------------
from rope_pos import REF_MODE, SLOT_MODES, SLOT_LADDER_RATIO  # noqa: E402

REF_ALLOC = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0, slot_planes=2)


def ref_module(**over):
    kw = dict(REF_ALLOC); kw.update(over)
    return GraphRoPE(HD, REF_MODE, **kw)


def test_ref_mode_is_registered_and_needs_the_path_and_slot():
    assert REF_MODE in ROPE_MODES
    from rope_pos import TREE_MODES
    assert REF_MODE in TREE_MODES, "the ref arm reads the tree, so it needs the path sidecar"
    assert REF_MODE in SLOT_MODES, "the ref arm needs tok_slot, so the loader must be told"


def test_ref_allocation_and_refusals():
    m = ref_module()
    assert m.n_planes == 64 and m.ref_used == 16 + 44 + 0 + 2   # 2 planes left unrotated
    with pytest.raises(ValueError, match="needs rope_slot_planes"):
        ref_module(slot_planes=0)
    with pytest.raises(ValueError, match="exceeds head_dim"):
        ref_module(slot_planes=40)
    # slot planes on a mode that cannot read them is a silent no-op, so it is refused
    with pytest.raises(ValueError, match="only rope_ref_slot"):
        GraphRoPE(HD, "rope_both", seq_planes=16, levels=22, planes_per_level=2,
                  token_planes=4, slot_planes=2)


def test_ref_init_is_the_ladder_and_theta_is_exactly_zero():
    m = ref_module()
    assert torch.equal(m.theta_slot, torch.zeros(2))
    # stored as float32, so compare at float32 precision
    assert math.isclose(float(m.step_slot[0].detach()), BASE_STEP, rel_tol=1e-6)
    assert math.isclose(float(m.step_slot[1].detach()), BASE_STEP / SLOT_LADDER_RATIO, rel_tol=1e-6)


def test_equal_slots_coincide_and_different_slots_do_not():
    """THE MECHANISM. Two tokens of the same symbol must have an identical angle on the
    slot planes -- that is what makes their relative rotation zero and lets a head match
    them. Two different symbols must not."""
    m = ref_module()
    path = rand_path(1, 4, levels=22)
    slot = torch.tensor([[3, 3, 7, 3]])
    a = m.angles(4, path, torch.device("cpu"), slot)
    sl = a[..., 16 + 44 : 16 + 44 + 2]
    assert torch.equal(sl[0, 0], sl[0, 1]), "same symbol must give the same angle"
    assert torch.equal(sl[0, 0], sl[0, 3])
    assert not torch.allclose(sl[0, 0], sl[0, 2]), "different symbols must differ"


def test_absent_slot_contributes_exactly_zero():
    m = ref_module()
    with torch.no_grad():                      # move theta off zero so 'present' is visible
        m.theta_slot.copy_(torch.tensor([0.5, -0.25]))
    path = rand_path(1, 3, levels=22)
    a = m.angles(3, path, torch.device("cpu"), torch.tensor([[ABSENT, 0, 5]]))
    sl = a[..., 16 + 44 : 16 + 44 + 2]
    assert torch.equal(sl[0, 0], torch.zeros(2)), "ABSENT must contribute exactly zero"
    # slot 0 is PRESENT, so it carries theta even though the index is 0 -- that is the
    # distinction between "symbol 0" and "no symbol", and it is why theta exists.
    assert torch.allclose(sl[0, 1], m.theta_slot)


def test_ref_slot_angle_formula_and_trailing_planes_unrotated():
    m = ref_module()
    with torch.no_grad():
        m.step_slot.copy_(torch.tensor([0.1, 0.02])); m.theta_slot.copy_(torch.tensor([0.3, -0.4]))
    path = rand_path(1, 2, levels=22)
    a = m.angles(2, path, torch.device("cpu"), torch.tensor([[4, ABSENT]]))
    sl = a[0, 0, 16 + 44 : 16 + 44 + 2]
    assert torch.allclose(sl, torch.tensor([4 * 0.1 + 0.3, 4 * 0.02 - 0.4]))
    assert torch.equal(a[..., 62:], torch.zeros(1, 2, 2)), "the spare planes must stay at 0"


def test_ref_mode_refuses_without_a_slot():
    m = ref_module()
    path = rand_path(1, 3, levels=22)
    with pytest.raises(ValueError, match="needs the symbol slot"):
        m.angles(3, path, torch.device("cpu"), None)
    with pytest.raises(ValueError, match=r"slot must be \(B, t\)"):
        m.angles(3, path, torch.device("cpu"), torch.zeros(1, 3, 1, dtype=torch.long))


def test_other_arms_ignore_the_slot_argument_entirely():
    """Strictly additive: passing a slot to a non-ref arm must not change its angles."""
    for mode, tok in (("rope_both", 4), ("rope_leaf", 4), ("rope_all", 4), ("rope_seq", 4)):
        m = GraphRoPE(HD, mode, seq_planes=16, levels=22, planes_per_level=2, token_planes=tok)
        path = rand_path(2, 5, levels=23)          # 22 levels + the within-node column
        base = m.angles(5, path, torch.device("cpu"))
        withslot = m.angles(5, path, torch.device("cpu"), torch.randint(0, 9, (2, 5)))
        assert torch.equal(base, withslot), f"{mode} must ignore the slot"


def test_ref_arm_trains_and_gradients_reach_the_slot_parameters():
    cfg = GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                    dropout=0.0, bias=False, pos_mode=REF_MODE, rope_seq_planes=16,
                    rope_levels=22, rope_planes_per_level=2, rope_token_planes=0,
                    rope_slot_planes=2, rope_decay=False)
    torch.manual_seed(0)
    m = GPT(cfg)
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 101, (2, 6), generator=g)
    tgt = torch.randint(0, 101, (2, 6), generator=g)
    pos = dict(path=rand_path(2, 6, levels=22), slot=torch.randint(-1, 9, (2, 6), generator=g))
    _, loss = m(idx, tgt, pos)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("step_slot", "theta_slot", "step", "theta"):
        gr = getattr(m.rope, name).grad
        assert gr is not None and torch.isfinite(gr).all(), f"no gradient for rope.{name}"
    assert float(m.rope.step_slot.grad.abs().sum()) > 0, "step_slot got a zero gradient"


def test_ref_arm_rotation_tensors_all_land_in_the_no_decay_group():
    cfg = GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                    dropout=0.0, bias=False, pos_mode=REF_MODE, rope_seq_planes=16,
                    rope_levels=22, rope_planes_per_level=2, rope_token_planes=0,
                    rope_slot_planes=2, rope_decay=False)
    torch.manual_seed(0)
    m = GPT(cfg)
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    decay_ids = {id(p) for gp in opt.param_groups if gp["weight_decay"] > 0 for p in gp["params"]}
    for p in m.rope.parameters():
        assert id(p) not in decay_ids, "a rotation tensor was left in the decay group"


def test_slot_modes_agree_with_what_actually_reads_a_slot():
    """The loader derives its requirements from SLOT_MODES. A mode that reads the slot but
    is missing from the tuple would train with no reference coordinate and look normal --
    the same class of drift that broke two launches in commit 3ca636a."""
    for mode in ROPE_MODES:
        kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
        if mode in ("rope_head_split", "rope_layer_split"):
            kw["seq_planes"] = 64; kw["n_head"] = 2
        if mode == PLAN_MODE:
            kw["seq_planes"] = 64; kw["layer_plan"] = "st"; kw["n_layer"] = 2
        if mode == REF_MODE:
            kw["token_planes"] = 0; kw["slot_planes"] = 2
        if mode == "rope_rel":
            kw["token_planes"] = 0; kw["rel_planes"] = 2
        if mode == REF_PATH_MODE:
            kw["token_planes"] = 0; kw["ref_planes"] = 4
        if mode == DEPTH_MODE:      # 16 seq + 2 depth, no tree planes, no token planes
            kw["token_planes"] = 0; kw["rel_planes"] = 2
        m = GraphRoPE(HD, mode, **kw)
        assert m.use_slot == (mode in SLOT_MODES), f"{mode}: use_slot disagrees with SLOT_MODES"


# --------------------------------------------------------------------------------------
# R11a: rope_rel -- DIRECTED RELATIONS BY CONSTANT OFFSET (2026-09-28)
# The slot group above encodes "are we the same", which is symmetric and carries one bit.
# This group encodes "what relation holds between us": its planes carry depth * step_rel,
# so a parent/child hop is EXACTLY one step, signed, and n hops are n steps. Direction,
# relation type and composition all come from the group structure. These tests pin the
# defining offset property, the absence of any presence term, and that the rank planes
# still carry sibling identity (depth alone would flatten every node at one depth).
# --------------------------------------------------------------------------------------
from rope_pos import REL_MODE, REL_LADDER_RATIO  # noqa: E402

REL_ALLOC = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0, rel_planes=2)
REL_SLICE = slice(16 + 44, 16 + 44 + 2)


def rel_module(**over):
    kw = dict(REL_ALLOC); kw.update(over)
    return GraphRoPE(HD, REL_MODE, **kw)


def rel_path(depths, ranks=None, levels=22):
    """(1, len(depths), levels+1) path where row i sits at depth depths[i]."""
    p = torch.full((1, len(depths), levels + 1), ABSENT, dtype=torch.long)
    for i, d in enumerate(depths):
        for k in range(d):
            p[0, i, k] = 0 if ranks is None else ranks[i][k]
    return p


def test_rel_mode_is_registered_and_needs_the_path():
    from rope_pos import TREE_MODES, SLOT_MODES
    assert REL_MODE in ROPE_MODES
    assert REL_MODE in TREE_MODES, "depth comes from the path, so the sidecar is required"
    assert REL_MODE not in SLOT_MODES, "rope_rel reads no slot; it must not demand tok_slot"
    with pytest.raises(ValueError, match="needs a DOM path"):
        rel_module().angles(3, None, "cpu")


def test_rel_allocation_shape_and_refusals():
    m = rel_module()
    assert m.n_planes == 64 and m.rel_used == 16 + 44 + 2      # 2 planes left unrotated
    assert m.angles(5, rel_path([0, 1, 2, 3, 4]), "cpu").shape == (1, 5, 64)
    with pytest.raises(ValueError, match="needs rope_rel_planes"):
        rel_module(rel_planes=0)
    with pytest.raises(ValueError, match="takes no token-leaf planes"):
        rel_module(token_planes=4)
    with pytest.raises(ValueError, match="exceeds head_dim/2"):
        rel_module(rel_planes=6)                                # 16 + 44 + 6 = 66 > 64
    with pytest.raises(ValueError, match="only rope_rel"):
        GraphRoPE(HD, "rope_both", seq_planes=16, levels=22, planes_per_level=2,
                  token_planes=4, rel_planes=2)


def test_rel_init_is_a_geometric_ladder_and_has_no_presence_term():
    m = rel_module()
    assert not hasattr(m, "theta_rel"), "rope_rel must have no presence term; that is the point"
    for p in range(2):
        assert m.step_rel[p].item() == pytest.approx(BASE_STEP / REL_LADDER_RATIO**p)


def test_parent_and_child_differ_by_exactly_one_step():
    """THE defining property. One hop is bit-exact; multi-hop is checked with a tolerance
    because 3*s - 1*s and 2*s differ in the last float32 bits (~6e-8), which is
    accumulation, not a logic error."""
    m = rel_module()
    with torch.no_grad():
        m.step_rel.copy_(torch.tensor([0.31, 0.017]))          # off the init ladder
    a = m.angles(4, rel_path([0, 1, 2, 3]), "cpu")[0][:, REL_SLICE]
    step = m.step_rel.detach()
    assert torch.equal(a[2] - a[1], step), "child - parent must be exactly +step_rel"
    assert torch.equal(a[1] - a[2], -step), "parent - child must be exactly -step_rel"
    torch.testing.assert_close(a[3] - a[1], 2 * step, rtol=0, atol=1e-6)   # grandparent
    torch.testing.assert_close(a[1] - a[0], step, rtol=0, atol=1e-7)       # out-of-tree is depth 0
    assert torch.equal(a[0], torch.zeros(2)), "depth 0 gives angle 0, with no presence offset"


def test_rel_planes_keep_sibling_identity_on_the_rank_planes():
    """depth * delta alone would flatten every node at one depth. The rank planes must still
    separate siblings, or the arm has destroyed the tree coordinate it sits beside."""
    m = rel_module()
    p = rel_path([2, 2], ranks=[[0, 0], [0, 7]])
    a = m.angles(2, p, "cpu")[0]
    assert torch.equal(a[0, REL_SLICE], a[1, REL_SLICE]), "same depth must share the rel angle"
    assert not torch.equal(a[0, 16:60], a[1, 16:60]), "siblings must differ on the rank planes"


@pytest.mark.parametrize("mode", [mm for mm in ROPE_MODES
                                 if mm not in (REL_MODE, DEPTH_MODE)])
def test_rel_planes_are_zero_under_every_other_mode(mode):
    """No existing arm may acquire a relational group, and none may allocate one."""
    kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
    if mode in ("rope_head_split", "rope_layer_split"):
        kw["seq_planes"] = 64; kw["n_head"] = 2
    if mode == PLAN_MODE:
        kw["seq_planes"] = 64; kw["layer_plan"] = "st"; kw["n_layer"] = 2
    if mode == REF_MODE:
        kw["token_planes"] = 0; kw["slot_planes"] = 2
    if mode == REF_PATH_MODE:
        kw["token_planes"] = 0; kw["ref_planes"] = 4
    if mode == DEPTH_MODE:
        kw["token_planes"] = 0; kw["rel_planes"] = 2
    m = GraphRoPE(HD, mode, **kw)
    assert m.use_rel is False and not hasattr(m, "step_rel")


def test_rel_modes_agree_with_what_actually_reads_a_depth():
    """Same drift guard as SLOT_MODES: use_rel must follow from the mode name alone, so a
    future arm cannot read the relational group without the tuple noticing."""
    for mode in ROPE_MODES:
        kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
        if mode in ("rope_head_split", "rope_layer_split"):
            kw["seq_planes"] = 64; kw["n_head"] = 2
        if mode == PLAN_MODE:
            kw["seq_planes"] = 64; kw["layer_plan"] = "st"; kw["n_layer"] = 2
        if mode == REF_MODE:
            kw["token_planes"] = 0; kw["slot_planes"] = 2
        if mode == REL_MODE:
            kw["token_planes"] = 0; kw["rel_planes"] = 2
        if mode == REF_PATH_MODE:
            kw["token_planes"] = 0; kw["ref_planes"] = 4
        if mode == DEPTH_MODE:      # 16 seq + 2 depth, no tree planes, no token planes
            kw["token_planes"] = 0; kw["rel_planes"] = 2
        m = GraphRoPE(HD, mode, **kw)
        assert m.use_rel == (mode in (REL_MODE, DEPTH_MODE)), (
            f"{mode}: use_rel disagrees with the modes that read a depth")


def test_rel_model_trains_one_step_and_gradients_reach_step_rel():
    cfg = GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                    dropout=0.0, bias=False, pos_mode=REL_MODE, rope_seq_planes=16,
                    rope_levels=22, rope_planes_per_level=2, rope_token_planes=0,
                    rope_rel_planes=2, rope_decay=False)
    torch.manual_seed(0)
    m = GPT(cfg)
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 101, (2, 6), generator=g)
    tgt = torch.randint(0, 101, (2, 6), generator=g)
    _, loss = m(idx, tgt, dict(path=rand_path(2, 6, levels=22)))
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("step_rel", "step", "theta"):
        gr = getattr(m.rope, name).grad
        assert gr is not None and torch.isfinite(gr).all(), f"no gradient for rope.{name}"
    assert float(m.rope.step_rel.grad.abs().sum()) > 0, "step_rel got a zero gradient"


def test_rel_rotation_tensors_all_land_in_the_no_decay_group():
    cfg = GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                    dropout=0.0, bias=False, pos_mode=REL_MODE, rope_seq_planes=16,
                    rope_levels=22, rope_planes_per_level=2, rope_token_planes=0,
                    rope_rel_planes=2, rope_decay=False)
    torch.manual_seed(0)
    m = GPT(cfg)
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    decay_ids = {id(p) for gp in opt.param_groups if gp["weight_decay"] > 0 for p in gp["params"]}
    for p in m.rope.parameters():
        assert id(p) not in decay_ids, "a rotation tensor was left in the decay group"


def test_every_mode_survives_a_round_trip_through_gptconfig():
    """THE TRAP, THIRD TIME. rope_slot_planes was set in a config, never added to train.py's
    model_args, and so arrived at GraphRoPE as 0 -- five queued seeds burned their retries
    (d132e3c). Before that the sidecar map drifted the same way (3ca636a). train.py keeps the
    key list in three places by hand, so this asserts the model-side contract: every allocation
    key a mode needs must be a real GPTConfig field that reaches GraphRoPE with its value.
    """
    import dataclasses
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    needed = {
        REF_MODE: {"rope_slot_planes": 2},
        REL_MODE: {"rope_rel_planes": 2},
        # rope_depth REUSES rope_rel_planes rather than adding a key of its own, so what must
        # survive the round trip is the same one -- there is no new chance to forget.
        DEPTH_MODE: {"rope_rel_planes": 2},
        REF_PATH_MODE: {"rope_ref_planes": 4},
        # rope_ref_anchor needs NO group key of its own -- it overwrites path levels rather
        # than allocating planes -- so what must survive the round trip is rope_both's own
        # allocation. That absence is the point: there is nothing new to forget in train.py.
        REF_ANCHOR_MODE: {"rope_token_planes": 4},
        "rope_leaf": {"rope_token_planes": 4},
        PLAN_MODE: {"rope_layer_plan": "st"},
    }
    for mode, keys in needed.items():
        for k in keys:
            assert k in fields, f"{mode} needs GPTConfig.{k}, which does not exist"
        base = dict(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                    dropout=0.0, bias=False, pos_mode=mode, rope_seq_planes=16,
                    rope_levels=22, rope_planes_per_level=2, rope_token_planes=0)
        if mode == PLAN_MODE:
            base["rope_seq_planes"] = 64; base["rope_token_planes"] = 4
        base.update(keys)
        m = GPT(GPTConfig(**base))
        for k, v in keys.items():
            attr = {"rope_slot_planes": "slot_planes", "rope_rel_planes": "rel_planes",
                    "rope_ref_planes": "ref_planes",
                    "rope_token_planes": "token_planes", "rope_layer_plan": "plan"}[k]
            got = getattr(m.rope, attr)
            got = "".join(got) if isinstance(got, tuple) else got
            assert got == v, f"{mode}: GPTConfig.{k}={v!r} arrived at GraphRoPE as {got!r}"


def test_the_anchor_arm_needs_no_new_allocation_key():
    """Every previous pointer arm added a plane-group key and each one broke a launch.

    rope_slot_planes (d132e3c) and the sidecar map before it (3ca636a) both reached GraphRoPE
    as 0 because train.py keeps its key list in three hand-maintained places. rope_ref_anchor
    allocates no group, so it introduces no fourth chance to make that mistake -- and it must
    REFUSE every group key, or a config could silently set one and be ignored.
    """
    import dataclasses
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    base = dict(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                dropout=0.0, bias=False, pos_mode=REF_ANCHOR_MODE, rope_seq_planes=16,
                rope_levels=22, rope_planes_per_level=2, rope_token_planes=4)
    m = GPT(GPTConfig(**base))
    assert m.rope.use_ref_anchor
    # exactly rope_both's parameter set: no step_ref, no theta_ref, no step_slot, no step_rel
    assert sorted(n for n, _ in m.rope.named_parameters()) == ["step", "theta"]
    for key in ("rope_slot_planes", "rope_rel_planes", "rope_ref_planes"):
        assert key in fields                      # the field exists for the arms that use it
        with pytest.raises(ValueError, match="silently ignored"):
            GPT(GPTConfig(**{**base, key: 2}))


# --------------------------------------------------------------------------------------
# R7: rope_ref_path -- THE GRID POINTER (2026-09-28)
#
# The slot group says "are we the same symbol", which is symmetric and carries one bit.
# This group says "I point AT that cell", by rotating by the referent's actual (row,
# column). In a spreadsheet the name IS the location, so no interning is needed.
#
# The property these tests exist to protect: rotary position only ever compares THE SAME
# PLANE INDEX between query and key, so the pointer coordinate and the target cell's own
# coordinate MUST share one group. A private group for references would have the query
# rotating planes the target never writes, and would encode nothing while looking fine.
# --------------------------------------------------------------------------------------
from rope_pos import (REF_PATH_MODE, REF_PATH_MODES, REF_PATH_AXES, REF_PATH_LADDER_RATIO,
                      REF_ROW_LEVEL, REF_COL_LEVEL)  # noqa: E402

#: 16 seq + 22*2 tree + 4 grid-pointer = 64 = HD/2. No token planes, by refusal.
RP_ALLOC = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0, ref_planes=4)
#: Where the pointer group lands in the angle vector.
RP_SLICE = slice(16 + 22 * 2, 16 + 22 * 2 + 4)
RP_TREE_SLICE = slice(16, 16 + 22 * 2)


def rp(**kw):
    a = dict(RP_ALLOC); a.update(kw)
    return GraphRoPE(HD, REF_PATH_MODE, **a)


def sheet_path(cells, levels=22, t=None):
    """A path whose tokens sit in the cells given as ``(row, col)``, or ``None`` for no cell.

    Mirrors the sheets spine: level 0 the sheet, ``REF_ROW_LEVEL`` the row,
    ``REF_COL_LEVEL`` the column, one deeper level so the token is inside the cell.
    """
    t = t or len(cells)
    p = torch.full((1, t, levels), ABSENT, dtype=torch.long)
    for i, c in enumerate(cells):
        if c is None:
            continue
        r, col = c
        p[0, i, 0] = 0
        p[0, i, REF_ROW_LEVEL] = r
        p[0, i, REF_COL_LEVEL] = col
        p[0, i, REF_COL_LEVEL + 1] = 0
    return p


def refs(pairs, t=None):
    """``(1, t, 2)`` of target coordinates; ``None`` means the token carries no reference."""
    t = t or len(pairs)
    r = torch.full((1, t, REF_PATH_AXES), ABSENT, dtype=torch.long)
    for i, pr in enumerate(pairs):
        if pr is not None:
            r[0, i, 0], r[0, i, 1] = pr
    return r


def test_ref_path_is_registered_everywhere_the_loader_looks():
    assert REF_PATH_MODE in ROPE_MODES
    from rope_pos import TREE_MODES, SLOT_MODES
    assert REF_PATH_MODE in TREE_MODES, "it reads the path for a token's own cell coordinate"
    assert REF_PATH_MODE in REF_PATH_MODES, "the loader derives need_ref from this tuple"
    assert REF_PATH_MODE not in SLOT_MODES, "it reads no symbol slot; it must not demand tok_slot"


def test_ref_path_modes_agree_with_what_actually_reads_a_pointer():
    """Same drift guard as SLOT_MODES and the rel tuple: use_ref_path must follow from the
    mode name alone, so a future arm cannot read the pointer without the tuple noticing --
    the class of bug that killed two launches (3ca636a) and burned five seeds (d132e3c)."""
    for mode in ROPE_MODES:
        kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
        if mode in ("rope_head_split", "rope_layer_split"):
            kw["seq_planes"] = 64; kw["n_head"] = 2
        if mode == PLAN_MODE:
            kw["seq_planes"] = 64; kw["layer_plan"] = "st"; kw["n_layer"] = 2
        if mode == REF_MODE:
            kw["token_planes"] = 0; kw["slot_planes"] = 2
        if mode == REL_MODE:
            kw["token_planes"] = 0; kw["rel_planes"] = 2
        if mode == REF_PATH_MODE:
            kw["token_planes"] = 0; kw["ref_planes"] = 4
        if mode == DEPTH_MODE:      # 16 seq + 2 depth, no tree planes, no token planes
            kw["token_planes"] = 0; kw["rel_planes"] = 2
        m = GraphRoPE(HD, mode, **kw)
        assert m.use_ref_path == (mode in REF_PATH_MODES), (
            f"{mode}: use_ref_path disagrees with REF_PATH_MODES")


# ---- THE MECHANISM: coincidence, exactly -------------------------------------------

def test_a_reference_lands_bit_exactly_on_its_targets_angle():
    """THE WHOLE ARM. Token 0 sits in cell (7, 2). Token 1 sits somewhere else entirely but
    REFERS to (7, 2). Their pointer-group angles must be bit-identical, because that zero
    difference is what attention between a reference and its referent sees.
    """
    m = rp()
    path = sheet_path([(7, 2), (99, 40)])
    ref = refs([None, (7, 2)])
    a = m.angles(2, path, "cpu", ref=ref)[0]
    assert torch.equal(a[0, RP_SLICE], a[1, RP_SLICE]), (
        "a reference must be angularly coincident with its target in the pointer group")
    # ...and it must NOT be coincident with a cell it does not name.
    other = m.angles(2, sheet_path([(8, 2), (99, 40)]), "cpu", ref=ref)[0]
    assert not torch.equal(other[0, RP_SLICE], other[1, RP_SLICE]), (
        "a reference to (7,2) must not coincide with cell (8,2)")


def test_both_sides_write_the_same_group_or_the_mechanism_is_vacuous():
    """THE TRAP, pinned. Rotary position compares only MATCHING plane indices, so if the
    pointer coordinate lived in planes the target cell never writes, a reference could never
    reach its referent -- the arm would train, converge and mean nothing.

    This asserts the group is written from BOTH sides: a token with no reference still puts
    its OWN cell's coordinate there. It fails if someone 'tidies' the arm by giving
    references a private group and leaving ordinary tokens at zero in it.
    """
    m = rp()
    path = sheet_path([(7, 2)])
    a = m.angles(1, path, "cpu", ref=refs([None]))[0]
    grp = a[0, RP_SLICE]
    assert not torch.allclose(grp, torch.zeros_like(grp)), (
        "a NON-reference token in cell (7,2) contributes zero to the pointer group, so the "
        "group is written by references alone. Rotary position only compares matching plane "
        "indices, which means a reference's angle could never meet its target's: the "
        "mechanism would be vacuous. The group must be written by both sides."
    )
    # And the two sides must agree numerically, not merely both be non-zero.
    as_target = m.angles(1, sheet_path([(7, 2)]), "cpu", ref=refs([None]))[0][0, RP_SLICE]
    as_pointer = m.angles(1, sheet_path([(0, 0)]), "cpu", ref=refs([(7, 2)]))[0][0, RP_SLICE]
    assert torch.equal(as_target, as_pointer), (
        "the same coordinate must produce the same angle whether it arrives as a token's own "
        "cell or as the target of a reference -- otherwise the two sides use different scales")


def test_the_pointer_group_reuses_one_parameter_set_for_both_sides():
    """Structural counterpart to the test above: there is exactly ONE step/theta pair for the
    group, so the two sides cannot drift apart even in principle."""
    m = rp()
    names = {n for n, _ in m.named_parameters()}
    assert "step_ref" in names and "theta_ref" in names
    assert not any("pointer" in n or "step_ref2" in n or "target" in n for n in names), (
        "a second parameter set for the pointer side would let the two scales diverge")
    assert m.step_ref.shape == (REF_PATH_AXES, RP_ALLOC["ref_planes"] // REF_PATH_AXES)
    assert m.theta_ref.shape == m.step_ref.shape


def test_sibling_identity_survives_the_pointer_group():
    """The group is ADDITIONAL to the per-level rank planes, never a replacement. Two cells in
    one row must still differ on the tree planes, or the pointer would have destroyed the
    structure it is meant to sit beside."""
    m = rp()
    a = m.angles(2, sheet_path([(7, 2), (7, 3)]), "cpu", ref=refs([None, None]))[0]
    assert not torch.equal(a[0, RP_TREE_SLICE], a[1, RP_TREE_SLICE]), (
        "two cells in the same row must differ on the tree rank planes")
    assert not torch.equal(a[0, RP_SLICE], a[1, RP_SLICE]), (
        "two different cells must differ in the pointer group too")


def test_two_references_to_one_cell_are_coincident_with_each_other():
    """Co-reference falls out for free: two formulas naming the same cell agree on the axis."""
    m = rp()
    a = m.angles(3, sheet_path([(1, 1), (2, 2), (5, 5)]), "cpu",
                 ref=refs([(9, 4), (9, 4), None]))[0]
    assert torch.equal(a[0, RP_SLICE], a[1, RP_SLICE])
    assert not torch.equal(a[0, RP_SLICE], a[2, RP_SLICE])


# ---- the absent class, and what it is deliberately not ------------------------------

def test_a_token_with_no_cell_contributes_exactly_zero():
    """Framing, separators and anything outside a cell: EXACTLY zero, as an absent tree level
    is. Measured on the sheets pack this is 19.35% of tokens -- against rope_ref_slot's 57%
    absent, which is what let the model repurpose that group's theta as a presence flag."""
    m = rp()
    a = m.angles(1, sheet_path([None]), "cpu", ref=refs([None]))[0]
    assert torch.equal(a[0, RP_SLICE], torch.zeros(RP_ALLOC["ref_planes"])), (
        "a token with no cell and no reference must contribute exactly zero")


def test_theta_starts_at_exactly_zero_so_cell_zero_and_no_cell_coincide_at_init():
    """The convention every other group follows: theta init is EXACTLY 0, so 'cell (0,0)' and
    'no cell' are indistinguishable at init and are separated only by training."""
    m = rp()
    assert torch.equal(m.theta_ref, torch.zeros_like(m.theta_ref))
    a = m.angles(2, sheet_path([(0, 0), None]), "cpu", ref=refs([None, None]))[0]
    assert torch.equal(a[0, RP_SLICE], a[1, RP_SLICE]), (
        "at init, cell (0,0) and no-cell must coincide; theta separates them later")


def test_theta_cancels_between_two_tokens_that_both_have_coordinates():
    """What theta_ref can and cannot do. With present=1 on both sides it is a constant, so it
    drops out of every query-key difference among tokens that have a coordinate; it can only
    separate those from tokens with no cell at all. Asserted by moving theta and showing the
    DIFFERENCE between two coordinate-bearing tokens is unchanged."""
    m = rp()
    path, ref = sheet_path([(7, 2), (9, 4)]), refs([None, None])
    before = m.angles(2, path, "cpu", ref=ref)[0]
    d_before = before[0, RP_SLICE] - before[1, RP_SLICE]
    with torch.no_grad():
        m.theta_ref.add_(0.37)
    after = m.angles(2, path, "cpu", ref=ref)[0]
    d_after = after[0, RP_SLICE] - after[1, RP_SLICE]
    # Tolerance, not bit-equality: the cancellation is exact in real arithmetic, but adding
    # theta to both angles before subtracting them loses the last bit or two in float32
    # ((a+c) - (b+c) is not bit-identical to a - b). The coincidence tests above ARE
    # bit-exact, because there both sides compute the same number rather than a difference.
    assert torch.allclose(d_before, d_after, atol=1e-6, rtol=0)
    # but it DOES move a coordinate-bearing token relative to a no-cell one
    mixed = m.angles(2, sheet_path([(7, 2), None]), "cpu", ref=ref)[0]
    assert not torch.equal(mixed[0, RP_SLICE], mixed[1, RP_SLICE])


def test_a_whole_row_target_takes_the_targets_absent_column_not_its_own():
    """The selector is keyed on the ROW deliberately. A whole-row reference has col=ABSENT;
    keying per-axis would hand the token the target's row beside ITS OWN column, a pair that
    names no cell that exists."""
    m = rp()
    a = m.angles(1, sheet_path([(3, 9)]), "cpu", ref=refs([(7, ABSENT)]))[0]
    per = RP_ALLOC["ref_planes"] // REF_PATH_AXES
    col_part = a[0, RP_SLICE][per:]
    assert torch.equal(col_part, torch.zeros(per)), (
        "a whole-row target must leave the column planes at zero, not at the token's own column")
    row_part = a[0, RP_SLICE][:per]
    expect = m.angles(1, sheet_path([(7, 0)]), "cpu", ref=refs([None]))[0][0, RP_SLICE][:per]
    assert torch.equal(row_part, expect), "the row axis must still carry the target's row"


# ---- the ladder, and the allocation ------------------------------------------------

def test_the_ladder_is_per_axis_and_starts_from_base_step():
    m = rp(ref_planes=4)
    per = 2
    want = torch.tensor([[BASE_STEP / (REF_PATH_LADDER_RATIO ** p) for p in range(per)]] * 2)
    assert torch.allclose(m.step_ref, want, atol=0, rtol=0)


def test_row_and_column_are_independent_axes():
    """Moving only the row must move only the row half of the group."""
    m = rp()
    per = RP_ALLOC["ref_planes"] // REF_PATH_AXES
    a = m.angles(2, sheet_path([(1, 5), (2, 5)]), "cpu", ref=refs([None, None]))[0]
    assert not torch.equal(a[0, RP_SLICE][:per], a[1, RP_SLICE][:per]), "row half must differ"
    assert torch.equal(a[0, RP_SLICE][per:], a[1, RP_SLICE][per:]), "column half must not"


def test_refuses_the_allocations_that_would_be_silently_wrong():
    with pytest.raises(ValueError, match="needs rope_ref_planes"):
        GraphRoPE(HD, REF_PATH_MODE, seq_planes=16, levels=22, planes_per_level=2,
                  token_planes=0, ref_planes=0)
    with pytest.raises(ValueError, match="multiple of"):
        GraphRoPE(HD, REF_PATH_MODE, seq_planes=16, levels=22, planes_per_level=2,
                  token_planes=0, ref_planes=3)
    with pytest.raises(ValueError, match="no token-leaf planes"):
        GraphRoPE(HD, REF_PATH_MODE, seq_planes=16, levels=22, planes_per_level=2,
                  token_planes=4, ref_planes=4)
    with pytest.raises(ValueError, match="exceeds"):
        GraphRoPE(HD, REF_PATH_MODE, seq_planes=16, levels=30, planes_per_level=2,
                  token_planes=0, ref_planes=4)
    with pytest.raises(ValueError, match="rope_levels >"):
        GraphRoPE(HD, REF_PATH_MODE, seq_planes=16, levels=2, planes_per_level=2,
                  token_planes=0, ref_planes=4)
    # allocated under the wrong arm -- would be silently ignored
    with pytest.raises(ValueError, match="only rope_ref_path"):
        GraphRoPE(HD, "rope_both", seq_planes=16, levels=22, planes_per_level=2,
                  token_planes=4, ref_planes=4)


def test_refuses_a_missing_or_malformed_pointer():
    m = rp()
    with pytest.raises(ValueError, match="needs the grid pointer"):
        m.angles(1, sheet_path([(1, 1)]), "cpu", ref=None)
    with pytest.raises(ValueError, match=r"must be \(B, t, 2\)"):
        m.angles(1, sheet_path([(1, 1)]), "cpu", ref=torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="needs a DOM path"):
        m.angles(1, None, "cpu", ref=refs([(1, 1)]))


@pytest.mark.parametrize("mode", [mm for mm in ROPE_MODES if mm != REF_PATH_MODE])
def test_no_other_arm_acquires_a_pointer_group(mode):
    kw = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
    if mode in ("rope_head_split", "rope_layer_split"):
        kw["seq_planes"] = 64; kw["n_head"] = 2
    if mode == PLAN_MODE:
        kw["seq_planes"] = 64; kw["layer_plan"] = "st"; kw["n_layer"] = 2
    if mode == REF_MODE:
        kw["token_planes"] = 0; kw["slot_planes"] = 2
    if mode == REL_MODE:
        kw["token_planes"] = 0; kw["rel_planes"] = 2
    if mode == DEPTH_MODE:
        kw["token_planes"] = 0; kw["rel_planes"] = 2
    m = GraphRoPE(HD, mode, **kw)
    assert m.use_ref_path is False
    assert not hasattr(m, "step_ref") and not hasattr(m, "theta_ref")


def test_the_pointer_group_sits_where_the_allocation_says():
    """The group must occupy exactly its own slice and leave the rest of the head alone."""
    m = rp()
    a = m.angles(1, sheet_path([(7, 2)]), "cpu", ref=refs([None]))[0]
    assert a.shape == (1, HD // 2)
    used = 16 + 22 * 2 + 4
    assert m.ref_path_used == used == HD // 2, "this allocation fills the head exactly"


def test_ref_path_model_trains_one_step_and_gradients_reach_both_parameters():
    cfg = GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                    dropout=0.0, bias=False, pos_mode=REF_PATH_MODE, rope_seq_planes=16,
                    rope_levels=22, rope_planes_per_level=2, rope_token_planes=0,
                    rope_ref_planes=4, rope_decay=False)
    m = GPT(cfg)
    torch.manual_seed(0)
    x = torch.randint(0, 101, (2, 6))
    y = torch.randint(0, 101, (2, 6))
    path = sheet_path([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), None]).repeat(2, 1, 1)
    ref = refs([None, (1, 1), None, (5, 5), None, None]).repeat(2, 1, 1)
    _, loss = m(x, y, pos=dict(path=path, ref=ref))
    loss.backward()
    assert m.rope.step_ref.grad is not None and m.rope.step_ref.grad.abs().sum() > 0
    assert m.rope.theta_ref.grad is not None


def test_reference_coordinate_reaches_the_model_through_the_real_loader():
    """End to end on the real sheets sidecar: the arrays that had NO consumer before R7 must
    now arrive as pos['ref'] with the right shape and a non-trivial number of live pointers."""
    import os
    pack = f"{PACKS}/sheet_values"
    if not os.path.isdir(pack + "_dompath"):
        pytest.skip("sheets pack not built")
    from topos_data import PackedWindows
    pw = PackedWindows(pack, need_tree=False, sidecar="path", need_ref=True)
    x, y, pos = pw.get_batch_records(list(range(8)), torch.device("cpu"))
    assert "ref" in pos, "the loader must serve the grid pointer under pos['ref']"
    assert pos["ref"].shape == (8, x.shape[1], REF_PATH_AXES)
    assert (pos["ref"][..., 0] >= 0).sum() > 0, "no live pointer in the first 8 records"
    # a column coordinate never appears without a row coordinate
    assert not ((pos["ref"][..., 1] >= 0) & (pos["ref"][..., 0] < 0)).any()


def test_loader_refuses_a_pack_with_no_pointer_arrays():
    """Refusing beats degrading: without the arrays this arm silently becomes rope_both."""
    import os
    from topos_data import PackedWindows
    py = f"{PACKS}/web_c4html_text_gpt2_b4096_v2"
    if not os.path.isdir(py + "_dompath"):
        pytest.skip("web pack not present")
    if os.path.exists(os.path.join(py + "_dompath", "tok_ref_row.npy")):
        pytest.skip("this pack happens to have pointer arrays")
    with pytest.raises(FileNotFoundError, match="tok_ref_row"):
        PackedWindows(py, need_tree=False, sidecar="path", need_ref=True)


def test_on_real_data_a_reference_coincides_with_a_token_in_its_target_cell():
    """THE LOOP CLOSED, on the actual pack rather than a synthetic case.

    ``test_sheet_pack.py::test_reference_coordinate_equals_its_targets_own_path`` asserts the
    invariant on the DATA side: a reference's stored coordinate equals its target cell's own
    path entries. This is its model-side counterpart -- that the invariant survives all the way
    into the angles, so a reference token and a token sitting inside the cell it names really
    do land on the same angle in the pointer group.

    Between them the two tests cover the full chain: builder -> sidecar -> loader -> rotation.
    Either alone would pass while the arm was broken.
    """
    import os
    pack = f"{PACKS}/sheet_values"
    if not os.path.isdir(pack + "_dompath"):
        pytest.skip("sheets pack not built")
    from topos_data import PackedWindows
    pw = PackedWindows(pack, need_tree=False, sidecar="path", need_ref=True)
    x, _y, pos = pw.get_batch_records(list(range(24)), torch.device("cpu"))
    m = rp(levels=pw.levels)
    a = m.angles(x.shape[1], pos["path"], "cpu", ref=pos["ref"])

    path, ref = pos["path"], pos["ref"]
    own_r, own_c = path[..., REF_ROW_LEVEL], path[..., REF_COL_LEVEL]
    has_ref = ref[..., 0] >= 0
    checked = 0
    for bi in range(path.shape[0]):
        # every position in this record that sits inside a cell and carries NO reference
        plain = (~has_ref[bi]) & (own_r[bi] >= 0) & (own_c[bi] >= 0)
        if not plain.any():
            continue
        cells = {}
        for ti in torch.nonzero(plain).flatten().tolist():
            cells.setdefault((int(own_r[bi, ti]), int(own_c[bi, ti])), ti)
        for ti in torch.nonzero(has_ref[bi]).flatten().tolist():
            key = (int(ref[bi, ti, 0]), int(ref[bi, ti, 1]))
            tgt = cells.get(key)
            if tgt is None:
                continue          # the referenced cell was cut by truncation; nothing to compare
            assert torch.equal(a[bi, ti, RP_SLICE], a[bi, tgt, RP_SLICE]), (
                f"record {bi}: reference at {ti} -> cell {key} does not share the pointer "
                f"angle with the token at {tgt} that sits in that cell"
            )
            checked += 1
    assert checked > 50, (
        f"only {checked} reference/target pairs were resolvable in 24 records -- too few to "
        "call this verified"
    )


# =============================================================================================
# rope_ref_anchor (R7b) -- a reference OVERWRITES its own sheet/row/column with its target's.
#
# Why this arm exists: rope_ref_path measured +0.0075 nats/token WORSE than rope_both (worse at
# 20/20 milestones, seed 0). Two faults, both removed here: its coordinate had no SHEET, so
# 17.25% of references aliased onto the wrong cell; and its separate pointer group DUPLICATED a
# token's own coordinate for the 63% carrying no reference, so same-cell pairs agreed on 8
# planes while a pointer and its target agreed on 4 and disagreed on the tree levels.
# =============================================================================================

from rope_pos import (REF_ANCHOR_MODE, REF_ANCHOR_MODES, REF_ANCHOR_AXES,   # noqa: E402
                      REF_ANCHOR_LEVELS, REF_COORD_MODES, REF_SHEET_LEVEL,
                      TREE_MODES, SLOT_MODES)

#: Byte-identical to config/sheet_both.py's allocation. That equality IS the no-confound claim.
RA_ALLOC = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=4)
RA_SEQ = slice(0, 16)
RA_TREE = slice(16, 16 + 44)
#: The three levels the arm overwrites, in plane space: 2 planes per level from plane 16.
RA_HEAD = slice(16, 16 + REF_ANCHOR_AXES * 2)
#: Everything below the column level -- the token's position within its formula.
RA_DEEP = slice(16 + REF_ANCHOR_AXES * 2, 16 + 44)


def ra(**kw):
    a = dict(RA_ALLOC); a.update(kw)
    return GraphRoPE(HD, REF_ANCHOR_MODE, **a)


def anchor_path(cells, deep=None, levels=22):
    """A sheets-shaped path. ``cells`` are ``(sheet, row, col)`` or ``None`` for no cell.

    ``deep[i]`` is an optional coordinate one level BELOW the column -- the token's position
    inside its formula -- so a reference operand and a bare literal can be told apart.
    """
    t = len(cells)
    p = torch.full((1, t, levels + 1), ABSENT, dtype=torch.long)   # +1 = token-leaf column
    for i, c in enumerate(cells):
        if c is None:
            continue
        sh, r, col = c
        p[0, i, REF_SHEET_LEVEL] = sh
        p[0, i, REF_ROW_LEVEL] = r
        p[0, i, REF_COL_LEVEL] = col
        if deep is not None and deep[i] is not None:
            p[0, i, REF_COL_LEVEL + 1] = deep[i]
    return p


def anchor_refs(targets, t=None):
    """``(1, t, 3)`` of target ``(sheet, row, col)``; ``None`` means no usable reference."""
    t = t or len(targets)
    r = torch.full((1, t, REF_ANCHOR_AXES), ABSENT, dtype=torch.long)
    for i, x in enumerate(targets):
        if x is not None:
            r[0, i, 0], r[0, i, 1], r[0, i, 2] = x
    return r


def test_ref_anchor_is_registered_everywhere_the_loader_looks():
    assert REF_ANCHOR_MODE in ROPE_MODES
    assert REF_ANCHOR_MODE in TREE_MODES, "it reads the path -- that is where it writes"
    assert REF_ANCHOR_MODE in REF_ANCHOR_MODES
    assert REF_ANCHOR_MODE in REF_COORD_MODES, "the loader keys need_ref on this tuple"
    assert REF_ANCHOR_MODE not in SLOT_MODES, "it reads no symbol slot"
    assert REF_ANCHOR_MODE not in REF_PATH_MODES, "it is not the separate-group arm"


def test_anchor_rotates_exactly_the_planes_rope_both_rotates():
    """The no-confound claim, made mechanically rather than in prose.

    rope_ref_path had to disclose 64 rotating planes against rope_both's 60, because it spent
    the 4 idle token-leaf planes. This arm adds no group, so the counts must match exactly.
    """
    m = ra()
    mb = GraphRoPE(HD, "rope_both", **RA_ALLOC)
    path = anchor_path([(0, 7, 2)])
    zero = anchor_path([None])
    a = m.angles(1, path, "cpu", ref=anchor_refs([None]))[0, 0]
    b = mb.angles(1, path, "cpu")[0, 0]
    az = m.angles(1, zero, "cpu", ref=anchor_refs([None]))[0, 0]
    bz = mb.angles(1, zero, "cpu")[0, 0]
    assert int((a != az).sum()) == int((b != bz).sum()), (
        "rope_ref_anchor rotates a different number of planes than rope_both, so sheet_both "
        "is no longer a confound-free paired baseline"
    )
    assert sorted(n for n, _ in m.named_parameters()) == sorted(n for n, _ in mb.named_parameters()), (
        "rope_ref_anchor holds different parameters than rope_both; it is supposed to add none"
    )


def test_with_no_reference_the_anchor_arm_IS_rope_both():
    """Nothing is written for a token carrying no reference -- no duplicate, no new class."""
    m = ra()
    mb = GraphRoPE(HD, "rope_both", **RA_ALLOC)
    path = anchor_path([(0, 7, 2), (1, 9, 5), None], deep=[3, None, None])
    a = m.angles(3, path, "cpu", ref=anchor_refs([None, None, None]))
    b = mb.angles(3, path, "cpu")
    assert torch.equal(a, b), (
        "with no references this arm must be bit-identical to rope_both -- any difference is a "
        "coordinate being written where none was asked for"
    )


def test_a_reference_lands_on_its_targets_angle():
    """The coincidence, on the three overwritten levels."""
    m = ra()
    # token 0: a Ref operand in cell (sheet 0, row 7, col 2), naming (sheet 0, row 9, col 5)
    # token 1: a literal sitting in that target cell
    path = anchor_path([(0, 7, 2), (0, 9, 5)], deep=[7, None])
    a = m.angles(2, path, "cpu", ref=anchor_refs([(0, 9, 5), None]))[0]
    assert torch.equal(a[0, RA_HEAD], a[1, RA_HEAD]), (
        "a reference does not share its target's sheet/row/column angle -- the mechanism is "
        "not connected"
    )


def test_a_reference_keeps_its_formula_position():
    """Only sheet/row/column change. The levels below hold which operand the token is."""
    m = ra()
    path = anchor_path([(0, 7, 2), (0, 9, 5)], deep=[7, None])
    a = m.angles(2, path, "cpu", ref=anchor_refs([(0, 9, 5), None]))[0]
    assert not torch.equal(a[0, RA_DEEP], a[1, RA_DEEP]), (
        "the reference operand and the target's literal agree on the deeper levels too, so the "
        "overwrite reached past the column level and erased the token's position in its formula"
    )
    # and the deep levels must equal what they would be with no reference at all
    plain = m.angles(2, path, "cpu", ref=anchor_refs([None, None]))[0]
    assert torch.equal(a[0, RA_DEEP], plain[0, RA_DEEP])


def test_cross_sheet_reference_does_not_alias_onto_the_own_sheet_cell():
    """FAULT 1 of rope_ref_path, pinned. 17.25% of references cross sheets.

    With a (row, col) pointer and no sheet axis, a reference to Sheet2!F10 coincided with
    whatever cell held row 10 column F on the reference's OWN sheet. Both halves are asserted:
    the wrong cell must NOT match and the right one must.
    """
    m = ra()
    #  0: a Ref on sheet 0 naming (sheet 1, row 9, col 5)
    #  1: a token in (sheet 0, row 9, col 5)  -- the decoy the old design matched
    #  2: a token in (sheet 1, row 9, col 5)  -- the real target
    path = anchor_path([(0, 7, 2), (0, 9, 5), (1, 9, 5)], deep=[7, None, None])
    a = m.angles(3, path, "cpu", ref=anchor_refs([(1, 9, 5), None, None]))[0]
    assert not torch.equal(a[0, RA_HEAD], a[1, RA_HEAD]), (
        "a cross-sheet reference still coincides with the same (row, col) on its OWN sheet -- "
        "the sheet axis is not reaching the rotation"
    )
    assert torch.equal(a[0, RA_HEAD], a[2, RA_HEAD]), (
        "a cross-sheet reference does not coincide with its actual target"
    )


def test_a_truncated_away_target_leaves_the_token_alone():
    """The invariant most likely to be broken by a later refactor.

    Measured on the v1 pack's val split: 24.4% of pointer-bearing tokens named a cell that is
    not in the window, because the record is cut at block_size. The builder suppresses those to
    ABSENT on all three axes. If the arm then overwrote anyway, such a token would lose its real
    position AND gain a coordinate matching nothing -- worse than leaving it alone, on a quarter
    of all pointers, and indistinguishable from the mechanism simply not working.
    """
    m = ra()
    path = anchor_path([(0, 7, 2), (1, 9, 5)], deep=[7, 3])
    suppressed = m.angles(2, path, "cpu", ref=anchor_refs([(ABSENT, ABSENT, ABSENT), None]))
    untouched = m.angles(2, path, "cpu", ref=anchor_refs([None, None]))
    assert torch.equal(suppressed, untouched), (
        "an all-ABSENT pointer changed the angles, so a reference whose target was truncated "
        "away loses its own sheet/row/column instead of keeping them"
    )


def test_a_whole_row_pointer_is_refused_not_half_applied():
    """SUPERSEDES an earlier test that asserted the opposite, and was wrong.

    That test required a whole-row target (``column = ABSENT``) to be APPLIED, reproducing the
    row node's own angle. Measured over all 52,801,353 token positions of the shipped pack, the
    population it protected is empty -- 0 tokens have a row without a column -- while 16 have a
    column without a row and were being silently discarded. The builder now refuses a whole-row
    target outright, and a partial pointer must leave the token alone: an absent level above
    present ones is a malformed path.
    """
    m = ra()
    path = anchor_path([(0, 7, 2)], deep=[7])
    got = m.angles(1, path, "cpu", ref=anchor_refs([(0, 9, ABSENT)]))
    untouched = m.angles(1, path, "cpu", ref=anchor_refs([None]))
    assert torch.equal(got, untouched), (
        "a whole-row pointer was applied, leaving level 2 ABSENT below two present levels"
    )


def test_two_references_to_one_cell_coincide_with_each_other():
    m = ra()
    path = anchor_path([(0, 7, 2), (0, 8, 3)], deep=[7, 4])
    a = m.angles(2, path, "cpu", ref=anchor_refs([(0, 9, 5), (0, 9, 5)]))[0]
    assert torch.equal(a[0, RA_HEAD], a[1, RA_HEAD])


def test_anchor_refuses_a_two_axis_pointer():
    """A v1 pack has no sheet axis, and silently using it is fault 1 all over again."""
    m = ra()
    path = anchor_path([(0, 7, 2)])
    two = torch.full((1, 1, 2), ABSENT, dtype=torch.long)
    with pytest.raises(ValueError, match="sheet"):
        m.angles(1, path, "cpu", ref=two)


def test_anchor_refuses_a_missing_pointer():
    m = ra()
    with pytest.raises(ValueError, match="pos\\['ref'\\]"):
        m.angles(1, anchor_path([(0, 7, 2)]), "cpu", ref=None)


def test_anchor_refuses_too_few_levels():
    with pytest.raises(ValueError, match="rope_levels"):
        GraphRoPE(HD, REF_ANCHOR_MODE, seq_planes=16, levels=2, planes_per_level=2,
                  token_planes=4)


def test_anchor_levels_are_the_contiguous_prefix():
    """``_anchor_path`` overwrites one slice, which is only correct while this holds."""
    assert REF_ANCHOR_LEVELS == tuple(range(REF_ANCHOR_AXES))
    assert REF_ANCHOR_LEVELS == (REF_SHEET_LEVEL, REF_ROW_LEVEL, REF_COL_LEVEL)


def test_anchor_takes_no_plane_allocation_key():
    """It adds no group, so it must reject every group allocation the other arms use."""
    for key in ("slot_planes", "rel_planes", "ref_planes"):
        with pytest.raises(ValueError, match="silently ignored"):
            ra(**{key: 2})


def test_anchor_writes_the_group_from_both_sides_or_it_is_vacuous():
    """THE TRAP, inherited from rope_ref_path and still worth pinning.

    Rotary position compares only matching plane indices. Here both sides write the same planes
    automatically -- the target writes its coordinate by simply being where it is -- but a
    future 'tidy-up' that moved the pointer into planes of its own would silently break the
    mechanism while the arm still trained and converged. This fails if the coordinate a
    reference writes is not the very same thing an ordinary token writes.
    """
    m = ra()
    as_target = m.angles(1, anchor_path([(0, 9, 5)], deep=[None]), "cpu",
                         ref=anchor_refs([None]))[0, 0, RA_HEAD]
    as_pointer = m.angles(1, anchor_path([(0, 0, 0)], deep=[None]), "cpu",
                          ref=anchor_refs([(0, 9, 5)]))[0, 0, RA_HEAD]
    assert torch.equal(as_target, as_pointer), (
        "the same coordinate produces a different angle depending on whether it arrives as a "
        "token's own cell or as a reference target. The two sides must run through the SAME "
        "per-level step/theta -- if a pointer group has been given its own parameters, a "
        "reference can never meet its referent and the arm encodes nothing."
    )


def test_a_pointer_applies_only_when_every_axis_is_present():
    """A PARTIAL overwrite would be a malformed path -- a level claiming absence above levels
    holding real values -- so any missing axis means the token keeps its own coordinates.

    This replaces keying on the row alone. Measured over the whole shipped pack, row-keying
    protected 0 token positions (row present, column absent) and silently discarded 16 (column
    present, row absent).
    """
    m = ra()
    path = anchor_path([(0, 7, 2)], deep=[7])
    untouched = m.angles(1, path, "cpu", ref=anchor_refs([None]))
    for partial in ((ABSENT, 9, 5), (0, ABSENT, 5), (0, 9, ABSENT),
                    (ABSENT, ABSENT, 5), (0, ABSENT, ABSENT)):
        got = m.angles(1, path, "cpu", ref=anchor_refs([partial]))
        assert torch.equal(got, untouched), (
            f"pointer {partial} with a missing axis was applied anyway; the token's path is now "
            "malformed -- an absent level above present ones"
        )
    # and a fully present pointer IS applied
    full = m.angles(1, path, "cpu", ref=anchor_refs([(0, 9, 5)]))
    assert not torch.equal(full, untouched)


# --------------------------------------------------------------------------------------------
# R11b -- rope_ref_delta: a reference points at its target by a CONSTANT VECTOR, not by
# coincidence. The whole arm is rope_ref_anchor plus one learned constant on the referencing
# side, so these tests are mostly about the two ways it could be silently vacuous: applied to
# both sides (cancels), or applied to the wrong token set (overwrite and offset disagree).
# --------------------------------------------------------------------------------------------

from rope_pos import REF_DELTA_MODE, REF_DELTA_MODES, read_ref_delta  # noqa: E402


def rd(**kw):
    a = dict(RA_ALLOC); a.update(kw)
    return GraphRoPE(HD, REF_DELTA_MODE, **a)


def set_delta(m, rows=((0.3, -0.2), (0.5, 0.1), (-0.4, 0.25))):
    """Put a distinct nonzero constant on every (axis, plane) so a mix-up cannot pass."""
    with torch.no_grad():
        m.delta_ref.copy_(torch.tensor(rows, dtype=torch.float32))
    return m


def test_ref_delta_is_registered_everywhere_the_loader_looks():
    assert REF_DELTA_MODE in ROPE_MODES
    assert REF_DELTA_MODE in TREE_MODES, "it overwrites path levels, so it reads the path"
    assert REF_DELTA_MODE in REF_ANCHOR_MODES, (
        "it overwrites identically to the anchor, so it owes the loader the same three-axis "
        "pointer including the sheet -- keying need_ref_sheet on this tuple is what supplies it"
    )
    assert REF_DELTA_MODE in REF_COORD_MODES, "the loader keys need_ref on this tuple"
    assert REF_DELTA_MODE in REF_DELTA_MODES
    assert REF_DELTA_MODE not in SLOT_MODES, "it reads no symbol slot"
    assert REF_DELTA_MODE not in REF_PATH_MODES, "it is not the separate-group arm"


def test_at_delta_zero_this_arm_is_bit_identical_to_the_anchor():
    """The cleanest statement that this is ref_anchor plus one degree of freedom.

    delta_ref initialises to EXACTLY zero, so before any training step the two arms cannot
    differ anywhere. If they do, the delta path is perturbing something it should not touch --
    and the paired comparison against ref_anchor would then not isolate the constant.
    """
    m, ma = rd(), ra()
    assert float(m.delta_ref.detach().abs().max()) == 0.0, (
        "delta_ref must initialise to exactly zero")
    path = anchor_path([(0, 7, 2), (0, 9, 5), None, (1, 3, 4)], deep=[7, None, None, 2])
    refs = anchor_refs([(0, 9, 5), None, (1, 3, 4), None])
    a = m.angles(4, path, "cpu", ref=refs)
    b = ma.angles(4, path, "cpu", ref=refs)
    assert torch.equal(a, b), "at delta=0 rope_ref_delta must be bit-identical to rope_ref_anchor"
    assert torch.equal(a.cos(), b.cos()) and torch.equal(a.sin(), b.sin())
    x = torch.randn(1, 2, 4, HD, generator=torch.Generator().manual_seed(4))
    for dt in (torch.float32, torch.bfloat16):
        ra_ = apply_rotary(x.to(dt), a.cos(), a.sin())
        rb_ = apply_rotary(x.to(dt), b.cos(), b.sin())
        assert torch.equal(ra_, rb_), f"rotation differs at delta=0 in {dt}"


def test_reference_minus_target_equals_delta_and_the_reverse_is_minus_delta():
    """The defining property, and the DIRECTION property that coincidence cannot express.

    Exact equality is not available: both angles carry ``rank * step`` of order several radians,
    so the float32 subtraction that recovers delta loses the low bits. The tolerance is on the
    SUBTRACTION, not on the mechanism -- the negation below is exact because it is the same
    two numbers in the other order.
    """
    m = set_delta(rd())
    # token 0: a reference in cell (0,7,2) naming (0,9,5). token 1: a literal in that cell.
    path = anchor_path([(0, 7, 2), (0, 9, 5)], deep=[7, None])
    a = m.angles(2, path, "cpu", ref=anchor_refs([(0, 9, 5), None]))[0]
    want = m.delta_ref.detach().flatten()
    fwd = a[0, RA_HEAD] - a[1, RA_HEAD]
    assert torch.allclose(fwd, want, atol=1e-5), (
        f"reference - target on the anchor planes is {fwd.tolist()}, not delta_ref "
        f"{want.tolist()} -- the constant is not landing where the arm claims"
    )
    rev = a[1, RA_HEAD] - a[0, RA_HEAD]
    assert torch.equal(rev, -fwd), "direction is not antisymmetric"
    assert not torch.allclose(fwd, torch.zeros_like(fwd)), (
        "the difference is zero, which is the anchor's behaviour -- with delta set, a referent "
        "must NOT be at the same angle as its target, or it stays confusable with a neighbour"
    )


def test_the_target_is_untouched_or_the_offset_would_cancel():
    """THE trap. Applied to both sides the constant cancels in every query-key difference and
    the arm silently reduces to rope_ref_anchor -- training, converging, meaning nothing. That
    is the same shape of failure as giving references a private plane group.

    So: a token carrying NO reference must have bit-identical angles whatever delta_ref is.
    """
    path = anchor_path([(0, 9, 5), (0, 7, 2), None], deep=[None, 3, None])
    refs = anchor_refs([None, None, None])           # nobody references anything
    zero = rd().angles(3, path, "cpu", ref=refs)
    big = set_delta(rd(), rows=((2.0, -3.0), (1.5, 0.75), (-1.25, 4.0))).angles(
        3, path, "cpu", ref=refs)
    assert torch.equal(zero, big), (
        "delta_ref changed the angles of tokens that carry no reference. It must apply to the "
        "REFERENCING side only: applied to both sides it cancels in every query-key difference "
        "and this arm becomes rope_ref_anchor with extra steps -- vacuous but still trainable."
    )


def test_the_offset_uses_the_same_selector_as_the_overwrite():
    """Overwrite and offset must fire on exactly the same tokens.

    If they drifted apart, some token would take its target's coordinate WITHOUT the constant
    (indistinguishable from a neighbour again) or the constant without the coordinate (pointing
    at nothing, delta away from its own cell). Both would train fine.
    """
    m = set_delta(rd())
    path = anchor_path([(0, 7, 2)], deep=[7])
    base = ra().angles(1, path, "cpu", ref=anchor_refs([None]))   # anchor, no pointer
    for partial in ((ABSENT, 9, 5), (0, ABSENT, 5), (0, 9, ABSENT)):
        got = m.angles(1, path, "cpu", ref=anchor_refs([partial]))
        assert torch.equal(got, base), (
            f"pointer {partial} is incomplete so the overwrite is refused, but the offset was "
            "still applied -- the token now sits delta away from its OWN cell, naming nothing"
        )


def test_composition_two_hops_is_twice_delta():
    """A reference whose target is itself a reference sits at 2*delta from the final referent.

    Nothing in the arm represents transitivity; the group structure supplies it. Tolerance
    again covers the float32 subtraction, not the mechanism.
    """
    m = set_delta(rd())
    # token 0 -> cell (0,9,5); token 1 IS in (0,9,5) and itself references (0,2,1);
    # token 2 is a literal sitting in (0,2,1), the end of the chain.
    path = anchor_path([(0, 7, 2), (0, 9, 5), (0, 2, 1)], deep=[7, 4, None])
    refs = anchor_refs([(0, 9, 5), (0, 2, 1), None])
    a = m.angles(3, path, "cpu", ref=refs)[0]
    want = m.delta_ref.detach().flatten()
    # hop 1: token 0 was overwritten to (0,9,5)+delta; token 1 to (0,2,1)+delta.
    # So token 0 - token 2 is (angle(0,9,5) - angle(0,2,1)) + delta, not 2*delta directly --
    # the composition shows up between token 1 and token 2, and between token 0 and a token
    # sitting in (0,9,5) with no reference. Build that fourth token to close the chain.
    path4 = anchor_path([(0, 7, 2), (0, 9, 5), (0, 2, 1), (0, 9, 5)], deep=[7, 4, None, None])
    refs4 = anchor_refs([(0, 9, 5), (0, 2, 1), None, None])
    b = m.angles(4, path4, "cpu", ref=refs4)[0]
    one_hop = b[0, RA_HEAD] - b[3, RA_HEAD]          # reference -> its target (no ref itself)
    assert torch.allclose(one_hop, want, atol=1e-5)
    # token 1 references (0,2,1); token 2 sits there with no reference: another single hop.
    hop2 = b[1, RA_HEAD] - b[2, RA_HEAD]
    assert torch.allclose(hop2, want, atol=1e-5)
    # Composition: chaining the two hops accumulates delta twice.
    assert torch.allclose(one_hop + hop2, 2 * want, atol=2e-5), (
        "two hops do not accumulate to 2*delta -- composition is supposed to fall out of the "
        "algebra with nothing representing transitivity"
    )


def test_ref_delta_adds_no_plane_and_rotates_what_rope_both_rotates():
    """No plane-count confound, asserted mechanically.

    It adds ONE parameter table (delta_ref) and no planes, so the rotating-plane count must
    match rope_both exactly -- the same claim ref_anchor makes, which rope_ref_path could not.
    """
    m, mb = rd(), GraphRoPE(HD, "rope_both", **RA_ALLOC)
    path = anchor_path([(0, 7, 2)])
    zero = anchor_path([None])
    refs = anchor_refs([None])
    a = m.angles(1, path, "cpu", ref=refs)[0, 0]
    az = m.angles(1, zero, "cpu", ref=refs)[0, 0]
    b = mb.angles(1, path, "cpu")[0, 0]
    bz = mb.angles(1, zero, "cpu")[0, 0]
    assert int((a != az).sum()) == int((b != bz).sum())
    assert sorted(n for n, _ in m.named_parameters()) == ["delta_ref", "step", "theta"], (
        "rope_ref_delta should hold exactly rope_both's parameters plus delta_ref"
    )
    assert m.delta_ref.shape == (REF_ANCHOR_AXES, RA_ALLOC["planes_per_level"])
    assert m.delta_ref.requires_grad, "delta_ref must be learned, not a fixed constant"


def test_delta_receives_gradient_or_it_can_never_move():
    """The diagnostic is 'did delta move', which presupposes it CAN. A detached or
    unreachable parameter would leave the reading at zero forever and look like a null result.
    """
    m = set_delta(rd(), rows=((0.1, 0.1), (0.1, 0.1), (0.1, 0.1)))
    path = anchor_path([(0, 7, 2), (0, 9, 5)], deep=[7, None])
    a = m.angles(2, path, "cpu", ref=anchor_refs([(0, 9, 5), None]))
    a.sum().backward()
    assert m.delta_ref.grad is not None, "delta_ref got no gradient"
    assert float(m.delta_ref.grad.abs().sum()) > 0, "delta_ref's gradient is identically zero"


def test_delta_needs_no_new_allocation_key():
    """Fifth time for the plumbing trap. delta_ref is a parameter, not an allocation, so there
    is nothing for train.py to forget -- and the arm must still REFUSE every group key.
    """
    base = dict(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                dropout=0.0, bias=False, pos_mode=REF_DELTA_MODE, rope_seq_planes=16,
                rope_levels=22, rope_planes_per_level=2, rope_token_planes=4)
    m = GPT(GPTConfig(**base))
    assert m.rope.use_ref_anchor and m.rope.use_ref_delta
    assert sorted(n for n, _ in m.rope.named_parameters()) == ["delta_ref", "step", "theta"]
    for key in ("rope_slot_planes", "rope_rel_planes", "rope_ref_planes"):
        with pytest.raises(ValueError, match="silently ignored"):
            GPT(GPTConfig(**{**base, key: 2}))


def test_ref_delta_refuses_too_few_levels():
    with pytest.raises(ValueError, match="overwrites path levels"):
        GraphRoPE(HD, REF_DELTA_MODE, seq_planes=16, levels=2, planes_per_level=2,
                  token_planes=4)


def test_read_ref_delta_round_trips_a_checkpoint(tmp_path):
    """The diagnostic must work on a real checkpoint layout, including the torch.compile
    '_orig_mod.' prefix every run in this project writes.
    """
    m = set_delta(rd())
    p = tmp_path / "ckpt.pt"
    torch.save({"model": {"_orig_mod.rope." + k: v for k, v in m.state_dict().items()},
                "iter_num": 8240, "val_nats_tok": 0.81}, p)
    got = read_ref_delta(str(p))
    assert got["shape"] == [REF_ANCHOR_AXES, RA_ALLOC["planes_per_level"]]
    assert got["moved_off_zero"] is True
    assert abs(got["max_abs"] - 0.5) < 1e-6
    assert got["iter"] == 8240
    # an untrained arm must read as NOT moved, or the diagnostic cannot answer its own question
    p2 = tmp_path / "fresh.pt"
    torch.save({"model": {"rope." + k: v for k, v in rd().state_dict().items()}}, p2)
    assert read_ref_delta(str(p2))["moved_off_zero"] is False


def test_read_ref_delta_refuses_a_checkpoint_from_another_arm(tmp_path):
    p = tmp_path / "anchor.pt"
    torch.save({"model": {"rope." + k: v for k, v in ra().state_dict().items()}}, p)
    with pytest.raises(KeyError, match="no delta_ref"):
        read_ref_delta(str(p))


# ===================================================================================
# rope_depth (2026-09-29) -- sequence + tree DEPTH and nothing else. 16 + 2 = 18 of 64.
# ===================================================================================

#: 16 seq + 2 depth = 18 of 64; the per-level planes are NOT allocated. That is the arm.
DEPTH_ALLOC = dict(seq_planes=16, levels=22, planes_per_level=2, token_planes=0, rel_planes=2)
DEPTH_SLICE = slice(16, 18)
DEPTH_TREE_WOULD_BE = slice(18, 18 + 22 * 2)


def rdep(**kw):
    a = dict(DEPTH_ALLOC); a.update(kw)
    return GraphRoPE(HD, DEPTH_MODE, **a)


def _path_at_depths(depths, levels=22):
    """A path whose token *i* sits at ``depths[i]``, with DIFFERENT sibling ranks throughout,
    so any test that passes cannot be passing because the subtrees happened to match."""
    p = torch.full((1, len(depths), levels), ABSENT, dtype=torch.long)
    for i, d in enumerate(depths):
        for k in range(d):
            p[0, i, k] = (i * 7 + k * 3) % 11      # deliberately unequal across tokens
    return p


def test_depth_arm_allocates_18_planes_and_no_tree_parameters():
    m = rdep()
    assert m.depth_used == 18
    assert m.use_seq is True and m.use_tree is False and m.use_token is False
    assert m.use_rel is True
    names = {n for n, _ in m.named_parameters()}
    assert names == {"step_rel"}, (
        f"rope_depth should own only the depth ladder, got {sorted(names)} -- a per-level "
        "step/theta table means the tree planes are still being allocated"
    )


def test_depth_is_the_count_of_non_absent_levels():
    """Including the two boundary cases: outside the tree entirely, and at the deepest level."""
    m = rdep()
    depths = [0, 1, 2, 5, 22]
    p = _path_at_depths(depths)
    a = m.angles(len(depths), p, "cpu")[0]
    step = m.step_rel.detach()
    for i, d in enumerate(depths):
        want = step * float(d)
        assert torch.allclose(a[i, DEPTH_SLICE], want, atol=0, rtol=0), (
            f"token at depth {d}: got {a[i, DEPTH_SLICE].tolist()}, want {want.tolist()}"
        )
    # depth 0 is a REAL value, not a sentinel: an out-of-tree token sits at angle 0 and is
    # therefore distinct from a depth-1 child of the root, which sits one step away.
    assert torch.equal(a[0, DEPTH_SLICE], torch.zeros(2))
    assert not torch.equal(a[1, DEPTH_SLICE], a[0, DEPTH_SLICE])


def test_depth_arm_does_not_rotate_the_per_level_planes():
    """THE DEFINING PROPERTY, and simultaneously the arm's weakness stated out loud.

    Two tokens at the SAME depth in DIFFERENT subtrees are indistinguishable on these planes.
    The full path would separate them on the per-level planes; this arm has none, so everything
    the tree said has to be recovered from depth plus emission order instead. If this test ever
    fails, some per-level rotation has crept back in and the arm is no longer the 18-plane arm.
    """
    m = rdep()
    p = torch.full((1, 2, 22), ABSENT, dtype=torch.long)
    p[0, 0, :3] = torch.tensor([0, 1, 2])          # depth 3, one subtree
    p[0, 1, :3] = torch.tensor([0, 9, 4])          # depth 3, a DIFFERENT subtree
    a = m.angles(2, p, "cpu")[0]
    assert torch.equal(a[0, DEPTH_SLICE], a[1, DEPTH_SLICE]), (
        "two tokens at one depth must be identical on the depth planes"
    )
    # and the planes the tree would have used are genuinely unrotated, not merely unequal
    assert bool((a[:, DEPTH_TREE_WOULD_BE] == 0).all())
    assert bool((a[:, 18:] == 0).all()), "everything above the depth group must be unrotated"


def test_depth_difference_is_linear_in_hops():
    """One level apart is exactly one step, two levels exactly two. The whole idea rests on it."""
    m = rdep()
    p = _path_at_depths([3, 4, 5])
    a = m.angles(3, p, "cpu")[0]
    step = m.step_rel.detach()
    one = a[1, DEPTH_SLICE] - a[0, DEPTH_SLICE]
    two = a[2, DEPTH_SLICE] - a[0, DEPTH_SLICE]
    assert torch.allclose(one, step, atol=1e-6), f"one hop: {one.tolist()} vs {step.tolist()}"
    assert torch.allclose(two, 2 * step, atol=1e-6), f"two hops: {two.tolist()}"
    # signed, so the reverse direction is the negative -- a parent is -one step from its child
    assert torch.allclose(a[0, DEPTH_SLICE] - a[1, DEPTH_SLICE], -step, atol=1e-6)


def test_depth_arm_needs_no_new_allocation_key():
    """It reuses rope_rel_planes, so there is no fifth chance at the plumbing trap that has
    already killed two launches (3ca636a) and burned five seeds (d132e3c)."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    assert "rope_rel_planes" in fields
    cfg = GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=2 * HD,
                    dropout=0.0, bias=False, pos_mode=DEPTH_MODE, rope_seq_planes=16,
                    rope_levels=22, rope_planes_per_level=2, rope_token_planes=0,
                    rope_rel_planes=2)
    m = GPT(cfg)
    assert m.rope.rel_planes == 2, "rope_rel_planes did not reach GraphRoPE"
    assert m.rope.depth_used == 18


def test_depth_arm_refuses_token_planes_and_a_zero_group():
    with pytest.raises(ValueError, match="no token-leaf planes"):
        rdep(token_planes=4)
    with pytest.raises(ValueError, match="rope_rel_planes > 0"):
        rdep(rel_planes=0)


def test_depth_and_order_reconstruct_the_tree():
    """THE JUSTIFICATION FOR THE ARM, on real data rather than in a shell history.

    A prefix (DFS pre-order) serialisation is fully determined by (emission order, depth): the
    parent of any node is the nearest PRECEDING node one level shallower. If that holds, depth
    is a LOSSLESS encoding of the tree even though the tree is never made explicit -- which is
    the entire reason to spend 2 planes instead of 44.
    """
    src = (f"{SOURCES}/"
           "rep23_textpack_wholefile_apps_kodcode_423k_astcompact_ident32k_treepos_v3")
    if not os.path.isdir(src):
        pytest.skip("Python source sidecar not present")
    parent = np.load(os.path.join(src, "node_parent.npy"), mmap_mode="r")
    depth = np.load(os.path.join(src, "node_depth.npy"), mmap_mode="r")
    noff = np.load(os.path.join(src, "node_off.npy"))
    ok = bad = 0
    for r in range(1500):
        lo, hi = int(noff[r]), int(noff[r + 1])
        d = np.asarray(depth[lo:hi])
        par = np.asarray(parent[lo:hi])
        last_at: dict[int, int] = {}
        for i in range(hi - lo):
            di = int(d[i])
            if last_at.get(di - 1, -1) == int(par[i]):
                ok += 1
            else:
                bad += 1
            last_at[di] = i
        if ok > 150_000:
            break
    assert ok > 100_000, f"only {ok} nodes checked"
    assert bad == 0, (
        f"{bad} of {ok + bad} parents NOT recoverable from (order, depth) -- the render is not "
        "a prefix serialisation and the losslessness argument for rope_depth does not hold"
    )
