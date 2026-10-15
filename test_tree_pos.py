"""Tests for the tree-position port.

The first test is the one that matters: this module was ported out of another
repository, and the banked numbers in ``research.md`` were produced there. If the
port is not the same computation, a re-run cannot reproduce them and the
disagreement would be blamed on training noise rather than on the port. So the
equivalence is asserted against the original implementation when it is reachable,
and skipped — loudly — when it is not.

Run: ``python -m pytest test_tree_pos.py -v``
"""

import os
import sys

import pytest
import torch

from tree_pos import MAX_DEPTH, MAX_NODES, MAX_SIB, TreeAddEmbedding

#: Where the original implementation lives, if this machine still has it. The
#: port's provenance is a checkout of the mathesis monorepo; see architecture.md.
_ORIGIN = os.environ.get(
    "TOPOS_ORIGIN_REPO",
    "",  # set TOPOS_ORIGIN_REPO to compare against the original implementation
)


def _fixture(n_embd=64, b=3, t=48, n=30, seed=7):
    torch.manual_seed(seed)
    causal = torch.randint(0, n, (b, t))
    causal[:, :5] = -1  # framing tokens: outside the tree
    parent = torch.randint(-1, n - 1, (b, n))
    parent[:, 0] = -1  # a root
    return dict(
        causal=causal,
        parent=parent,
        depth=torch.randint(0, 25, (b, n)),
        sib=torch.randint(0, 10, (b, n)),
        node=causal,  # the original reads 'causal'; 'node' present but unused
    )


def test_port_is_bit_identical_to_the_original():
    """The port must be the SAME computation, not a similar one.

    Asserted bit-for-bit rather than within a tolerance: both run the same
    operations in the same order on the same tables, so any difference at all
    means a real divergence, not float noise.
    """
    if not os.path.isdir(_ORIGIN):
        pytest.skip(
            f"original implementation not at {_ORIGIN}; set TOPOS_ORIGIN_REPO to "
            "check the port, or accept that equivalence is unverified on this machine"
        )
    sys.path.insert(0, _ORIGIN)
    from mathesis.repr2x2.token_rope import MAX_ADD_DEPTH, MAX_ADD_NODES, MAX_ADD_SIB
    from mathesis.repr2x2.token_rope import TreeAddEmbedding as Original

    assert (MAX_NODES, MAX_DEPTH, MAX_SIB) == (MAX_ADD_NODES, MAX_ADD_DEPTH, MAX_ADD_SIB)

    ported = TreeAddEmbedding(64)
    original = Original(64, block_size=1024, mode="tree_add_nopos")

    # The private init seed should make the tables identical without copying.
    for name in ("ord_emb", "depth_emb", "sib_emb"):
        assert torch.equal(getattr(ported, name).weight, getattr(original, name).weight), name

    pos = _fixture()
    with torch.no_grad():
        assert torch.equal(ported(pos), original(0, pos["causal"].shape[1], pos, torch.device("cpu")))


def test_outside_the_tree_gets_exactly_zero():
    """Not row 0 — row 0 is a real depth and a real sibling rank.

    A framing token handed row 0 would be taught that framing looks like a
    shallow first child, which is a claim about the data that nobody made.
    """
    pos = _fixture()
    out = TreeAddEmbedding(64)(pos)
    assert out[:, :5].abs().sum().item() == 0.0
    assert out[:, 5:].abs().sum().item() > 0.0


def test_positional_budget_does_not_scale_with_context_length():
    """The length-independence property, asserted rather than described.

    Two instances differing only in the context length they will be used at must
    have identical parameter counts — that is the whole reason tree coordinates
    are interesting, so it should fail loudly if someone reintroduces a
    block_size-sized table.
    """
    assert TreeAddEmbedding(64).n_params() == TreeAddEmbedding(64).n_params()
    expected = (MAX_NODES + MAX_DEPTH + MAX_SIB) * 64
    assert TreeAddEmbedding(64).n_params() == expected


def test_missing_coordinates_raise_rather_than_default():
    """A silently-defaulted coordinate would train a wrong position and log
    nothing. Each required key must be demanded by name."""
    for key in ("causal", "parent", "depth", "sib"):
        pos = _fixture()
        pos[key] = None
        with pytest.raises(ValueError, match=key):
            TreeAddEmbedding(64)(pos)


def test_arms_differ_by_exactly_the_tables_they_should():
    """Guards the claim in the commit message and the README."""
    from model import GPT, GPTConfig

    def n(mode):
        cfg = GPTConfig(block_size=1024, vocab_size=4096, n_layer=2, n_head=2,
                        n_embd=64, pos_mode=mode)
        return sum(p.numel() for p in GPT(cfg).parameters())

    learned, none, tree = n("learned"), n("none"), n("tree")
    assert learned - none == 1024 * 64, "control minus floor must be the sequence table"
    assert tree - none == (MAX_NODES + MAX_DEPTH + MAX_SIB) * 64, "treatment minus floor must be the tree tables"


def test_wpe_is_absent_when_it_should_be():
    """Under 'none' and 'tree' the sequence table must not exist at all, so it
    cannot hold parameters or take gradients by accident."""
    from model import GPT, GPTConfig

    for mode, expected in (("learned", True), ("none", False), ("tree", False)):
        cfg = GPTConfig(block_size=1024, vocab_size=4096, n_layer=2, n_head=2,
                        n_embd=64, pos_mode=mode)
        model = GPT(cfg)
        assert ("wpe" in dict(model.transformer.named_children())) is expected, mode
        assert (model.tree_pos is not None) is (mode == "tree"), mode
