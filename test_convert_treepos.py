"""The treepos_v3 -> dompath conversion must reproduce the web packs' path rule exactly."""
import json

import numpy as np
import pytest

from convert_treepos import ABSENT, build_node_path, convert


def _write_pack(tmp_path, parent, sib, depth, node_off, n_tokens=None):
    """A minimal pack + treepos_v3 sidecar. Returns the pack dir."""
    pack = tmp_path / "src"
    side = tmp_path / "src_treepos_v3"
    pack.mkdir()
    side.mkdir()
    n_rec = len(node_off) - 1
    n_tokens = n_tokens if n_tokens is not None else 4 * n_rec
    per = n_tokens // n_rec
    np.save(pack / "tokens.npy", np.arange(n_tokens, dtype=np.uint16))
    np.save(pack / "tok_off.npy", np.arange(n_rec + 1, dtype=np.int64) * per)
    json.dump(dict(block_size=16, vocab_size=64, eot=0, n_train=max(1, n_rec - 2), n_val=1, n_test=1),
              open(pack / "meta.json", "w"))
    # one token per node, pointing at it, so tok_causal is meaningful and in range
    causal = np.concatenate([np.arange(node_off[r + 1] - node_off[r], dtype=np.int32)[:per]
                             for r in range(n_rec)])
    causal = np.resize(causal, n_tokens).astype(np.int32)
    np.save(side / "tok_causal.npy", causal)
    np.save(side / "tok_node.npy", causal + 1)          # gold map: deliberately DIFFERENT
    np.save(side / "tok_within.npy", (np.arange(n_tokens) % 3).astype(np.int16))
    np.save(side / "node_parent.npy", np.asarray(parent, dtype=np.int32))
    np.save(side / "node_sib.npy", np.asarray(sib, dtype=np.int16))
    np.save(side / "node_depth.npy", np.asarray(depth, dtype=np.int16))
    np.save(side / "node_off.npy", np.asarray(node_off, dtype=np.int64))
    return str(pack)


# record 0: root, two children, two grandchildren under the first and one under the second.
# S0 deliberately equals the FLAT ranks, so the default path is checkable against it.
# record 1: ids restart at 0, with distinctive ranks so a leak across the boundary would show
P0 = [-1, 0, 1, 1, 0, 4]
S0 = [0, 0, 0, 1, 1, 0]
D0 = [0, 1, 2, 2, 1, 2]
P1 = [-1, 0, 1]
S1 = [0, 3, 2]
D1 = [0, 1, 2]
OFF = [0, 6, 9]


def _tree():
    return P0 + P1, S0 + S1, D0 + D1, OFF


def test_three_level_tree_gets_the_expected_rows():
    parent, sib, depth, off = _tree()
    path, _ = build_node_path(parent, sib, depth, off, levels=4)
    assert path[1].tolist() == [0, -1, -1, -1]
    assert path[2].tolist() == [0, 0, -1, -1]
    assert path[3].tolist() == [0, 1, -1, -1]
    assert path[4].tolist() == [1, -1, -1, -1]
    assert path[5].tolist() == [1, 0, -1, -1]


def test_the_root_is_not_a_level():
    """Depth 0 is the Module node, the same in every program, so its row is all ABSENT."""
    parent, sib, depth, off = _tree()
    path, _ = build_node_path(parent, sib, depth, off, levels=4)
    assert path[0].tolist() == [ABSENT] * 4
    assert path[6].tolist() == [ABSENT] * 4          # record 1's root


def test_paths_are_record_local():
    """Record 1's ids restart at 0; its paths must come from ITS parents, not record 0's.

    Pinned to rank="field" so the supplied ranks 3 and 2 survive into the path and a
    leak across the record boundary would be unmistakable."""
    parent, sib, depth, off = _tree()
    path, _ = build_node_path(parent, sib, depth, off, levels=4, rank="field")
    assert path[7].tolist() == [3, -1, -1, -1]
    assert path[8].tolist() == [3, 2, -1, -1]


def test_deep_paths_truncate_at_the_leaf_end():
    """A node deeper than `levels` keeps its ANCESTORS' ranks and loses its own.

    rank="field" so the distinct ranks 7/8/9/5 are visible; a flat rank would make
    every node in this single chain 0 and the assertion vacuous."""
    parent = [-1, 0, 1, 2, 3]
    sib = [0, 7, 8, 9, 5]
    depth = [0, 1, 2, 3, 4]
    off = [0, 5]
    path, stats = build_node_path(parent, sib, depth, off, levels=2, rank="field")
    assert path[1].tolist() == [7, -1]
    assert path[2].tolist() == [7, 8]
    assert path[3].tolist() == [7, 8]                # rank 9 truncated away
    assert path[4].tolist() == [7, 8]                # rank 5 truncated away
    assert stats["nodes_deeper_than_levels"] == 2    # depths 3 and 4 exceed levels=2
    assert stats["records_with_a_deep_node"] == 1


def test_refuses_a_sidecar_that_is_not_in_dfs_preorder():
    """A parent after its child would make the fill read unfilled rows -- fail loudly."""
    parent = [-1, 2, 0]      # node 1's parent is node 2, which comes later
    sib = [0, 0, 0]
    depth = [0, 2, 1]
    with pytest.raises(ValueError, match="DFS pre-order"):
        build_node_path(parent, sib, depth, [0, 3], levels=4)


def test_convert_copies_the_token_arrays_verbatim(tmp_path):
    parent, sib, depth, off = _tree()
    pack = _write_pack(tmp_path, parent, sib, depth, off, n_tokens=8)
    out = str(tmp_path / "dst")
    convert(pack, out, levels=4)
    for f in ("tok_within.npy", "node_off.npy"):
        assert np.array_equal(np.load(f"{out}_dompath/{f}"), np.load(f"{pack}_treepos_v3/{f}"))
    # tok_node comes from tok_causal -- the lookahead-free map -- not from the gold tok_node
    assert np.array_equal(np.load(f"{out}_dompath/tok_node.npy"),
                          np.load(f"{pack}_treepos_v3/tok_causal.npy"))
    assert not np.array_equal(np.load(f"{out}_dompath/tok_node.npy"),
                              np.load(f"{pack}_treepos_v3/tok_node.npy"))


def test_convert_writes_a_loadable_pack(tmp_path):
    """The result must load through PackedWindows(sidecar='path') with levels+1 columns."""
    from topos_data import PackedWindows

    parent, sib, depth, off = _tree()
    pack = _write_pack(tmp_path, parent, sib, depth, off, n_tokens=8)
    out = str(tmp_path / "dst")
    m = convert(pack, out, levels=4)
    assert m["levels"] == 4 and m["absent"] == ABSENT

    pw = PackedWindows(out, need_tree=False, sidecar="path")
    assert pw.n_records == 2
    toks, pos = pw.record(0)
    assert pos["path"].shape == (len(toks), 5)       # levels + the within-node column
    # every in-tree token's row is its node's path
    npth = np.load(f"{out}_dompath/node_path.npy")
    tn = np.load(f"{out}_dompath/tok_node.npy")[: len(toks)]
    for i, node in enumerate(tn):
        if node >= 0:
            assert pos["path"][i, :4].tolist() == npth[node].tolist()


# Two children of ONE parent, both with source rank 0 -- what node_sib does across
# fields (an If's body[0] and orelse[0]). Flat ranking must separate them.
_SAME_RANK = dict(parent=[-1, 0, 0], sib=[0, 0, 0], depth=[0, 1, 1], off=[0, 3])


def test_field_ranks_collide_and_flat_ranks_do_not():
    p, s, d, o = (_SAME_RANK[k] for k in ("parent", "sib", "depth", "off"))
    field, _ = build_node_path(p, s, d, o, levels=3, rank="field")
    assert field[1].tolist() == field[2].tolist(), "the source ranks really do collide"

    flat, stats = build_node_path(p, s, d, o, levels=3, rank="flat")
    assert flat[1].tolist() == [0, -1, -1]
    assert flat[2].tolist() == [1, -1, -1]
    assert stats["rank"] == "flat"


def test_flat_is_the_default():
    p, s, d, o = (_SAME_RANK[k] for k in ("parent", "sib", "depth", "off"))
    default, _ = build_node_path(p, s, d, o, levels=3)
    flat, _ = build_node_path(p, s, d, o, levels=3, rank="flat")
    assert np.array_equal(default, flat)


def test_flat_rank_counts_in_document_order_across_depths():
    """A parent's children are ranked 0,1,2... in id order even when subtrees
    intervene, so the rank is a position among siblings, not among all descendants."""
    # 0 root; 1 child(rank0); 2 grandchild; 3 child(rank1); 4 grandchild; 5 child(rank2)
    parent = [-1, 0, 1, 0, 3, 0]
    sib = [0, 0, 0, 0, 0, 0]
    depth = [0, 1, 2, 1, 2, 1]
    path, _ = build_node_path(parent, sib, depth, [0, 6], levels=3, rank="flat")
    assert [path[i][0] for i in (1, 3, 5)] == [0, 1, 2]
    assert path[2].tolist() == [0, 0, -1]
    assert path[4].tolist() == [1, 0, -1]


def test_rejects_an_unknown_rank():
    p, s, d, o = (_SAME_RANK[k] for k in ("parent", "sib", "depth", "off"))
    with pytest.raises(ValueError, match="rank must be"):
        build_node_path(p, s, d, o, levels=3, rank="sideways")


# --------------------------------------------------------------------------------------
# THE SYMBOL SLOT. node_ref is RECORD-LOCAL (verified on the real pack: all 9,707,041
# refs satisfy 0 <= ref < n_nodes_in_record and none satisfies the global form), so the
# builder must rebase it through node_off. Using it unrebased would index a different
# record's nodes and the loss curve would look entirely normal.
# --------------------------------------------------------------------------------------
from convert_treepos import SLOT_GATES, build_tok_slot  # noqa: E402


def test_slot_groups_a_use_with_its_binder_and_numbers_in_first_mention_order():
    # two records of 4 nodes. rec0: node1 -> node0, node3 -> node2  (two symbols)
    #                         rec1: node2 -> node1, node3 -> node1  (one symbol, two uses)
    node_off = np.array([0, 4, 8], dtype=np.int64)
    ref = np.array([-1, 0, -1, 2,   -1, -1, 1, 1], dtype=np.int32)
    tok_off = np.array([0, 4, 8], dtype=np.int64)
    causal = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    # every name resolves immediately: ident_at strictly below char_end
    ident_at = np.zeros(8, dtype=np.int32)
    char_end = np.ones(8, dtype=np.int32)
    slot, st = build_tok_slot(ref, node_off, causal, tok_off, ident_at, char_end, gate="strict")
    # rec0: class{0,1} is first mention -> slot 0; class{2,3} -> slot 1
    assert slot[:4].tolist() == [0, 0, 1, 1]
    # rec1: node0 has no ref and is nobody's target, so it carries NO symbol
    assert slot[4] == ABSENT
    # class{1,2,3} is rec1's only symbol -> slot 0, numbered per record not globally
    assert slot[5:8].tolist() == [0, 0, 0]
    assert st["slots_per_record_max"] == 2 and st["records_over_318_slots"] == 0


def test_slot_refuses_a_forward_edge():
    """First-mention slots assume every edge is backward. A forward edge would make a
    token's slot depend on a binder the reader has not reached."""
    node_off = np.array([0, 3], dtype=np.int64)
    ref = np.array([2, -1, -1], dtype=np.int32)          # node0 -> node2: FORWARD
    tok_off = np.array([0, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="FORWARD edge"):
        build_tok_slot(ref, node_off, np.array([0, 1, 2], dtype=np.int32), tok_off,
                       np.zeros(3, dtype=np.int32), np.ones(3, dtype=np.int32))


def test_slot_refuses_a_ref_that_is_not_record_local():
    node_off = np.array([0, 2, 4], dtype=np.int64)
    ref = np.array([-1, -1, -1, 3], dtype=np.int32)      # rec1 node1 -> local 3, but n=2
    tok_off = np.array([0, 2, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="not record-local"):
        build_tok_slot(ref, node_off, np.array([0, 1, 0, 1], dtype=np.int32), tok_off,
                       np.zeros(4, dtype=np.int32), np.ones(4, dtype=np.int32))


def test_slot_gate_strict_needs_the_delimiter_consumed():
    """node_ident_at is ALREADY one past the name, so equality means the name's last
    character has been read but not its delimiter. strict refuses that; complete allows
    it and thereby takes one character of lookahead."""
    node_off = np.array([0, 2], dtype=np.int64)
    ref = np.array([-1, 0], dtype=np.int32)
    tok_off = np.array([0, 2], dtype=np.int64)
    causal = np.array([0, 1], dtype=np.int32)
    ident_at = np.array([5, 5], dtype=np.int32)
    char_end = np.array([5, 5], dtype=np.int32)          # equal: name read, delimiter not
    strict, _ = build_tok_slot(ref, node_off, causal, tok_off, ident_at, char_end, gate="strict")
    complete, _ = build_tok_slot(ref, node_off, causal, tok_off, ident_at, char_end, gate="complete")
    assert strict.tolist() == [ABSENT, ABSENT], "strict must not claim the identity yet"
    assert complete.tolist() == [0, 0], "complete claims it one character early"


def test_slot_gate_carry_forward_fills_but_never_across_a_record():
    node_off = np.array([0, 2, 4], dtype=np.int64)
    # both records: local node1 -> local node0, so each has one symbol class {0,1}
    ref = np.array([-1, 0, -1, 0], dtype=np.int32)
    tok_off = np.array([0, 3, 6], dtype=np.int64)
    causal = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
    # rec0's first token resolves (ident_at 0 < char_end 1); nothing in rec1 ever does
    ident_at = np.array([0, 9, 9, 9, 9, 9], dtype=np.int32)
    char_end = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    carry, _ = build_tok_slot(ref, node_off, causal, tok_off, ident_at, char_end, gate="carry")
    # rec0 token0 resolves (0 < 1) and the slot carries forward over tokens 1,2
    assert carry[:3].tolist() == [0, 0, 0]
    # rec1 never resolves (9 < 1 is false), and rec0's slot must NOT leak across
    assert carry[3:].tolist() == [ABSENT, ABSENT, ABSENT], "a slot leaked across records"


def test_slot_gate_names_are_validated():
    node_off = np.array([0, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="gate must be one of"):
        build_tok_slot(np.array([-1, 0], dtype=np.int32), node_off,
                       np.array([0, 1], dtype=np.int32), np.array([0, 2], dtype=np.int64),
                       np.zeros(2, dtype=np.int32), np.ones(2, dtype=np.int32), gate="nope")
    assert SLOT_GATES[0] == "carry", "carry is the leak-free default"
