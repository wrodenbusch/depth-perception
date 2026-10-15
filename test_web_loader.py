"""The ``path`` sidecar kind of ``PackedWindows`` (the web cell's rope arms).

Kept apart from ``test_topos_data.py`` because that file is skipped as a whole
when the Python pack is absent, and these tests need no pack on disk.
"""
import json

import numpy as np
import pytest
import torch

from topos_data import PackedWindows


def _tiny_path_pack(tmp_path):
    d = tmp_path / "pack"
    s = tmp_path / "pack_dompath"
    d.mkdir()
    s.mkdir()
    toks = np.array([50256, 1, 2, 3, 50256, 50256, 4, 5, 50256], dtype=np.uint16)
    np.save(d / "tokens.npy", toks)
    np.save(d / "tok_off.npy", np.array([0, 5, 9], dtype=np.int64))
    json.dump(dict(block_size=8, vocab_size=50304, eot=50256, n_train=1, n_val=1, n_test=0),
              open(d / "meta.json", "w"))
    np.save(s / "tok_node.npy", np.array([-1, 0, 0, 1, -1, -1, 0, 0, -1], dtype=np.int32))
    np.save(s / "node_path.npy",
            np.array([[0] + [-1] * 23, [0, 1] + [-1] * 22, [2] + [-1] * 23], dtype=np.int16))
    np.save(s / "node_off.npy", np.array([0, 2, 3], dtype=np.int64))
    return str(d)


def test_path_sidecar_gives_each_token_its_node_path(tmp_path):
    pw = PackedWindows(_tiny_path_pack(tmp_path), sidecar="path")
    assert pw.sidecar == "path" and pw.levels == 24
    toks, pos = pw.record(0)
    assert list(pos) == ["path"]
    assert pos["path"].shape == (5, 24) and pos["path"].dtype == np.int64
    assert pos["path"][0].tolist() == [-1] * 24  # framing
    assert pos["path"][1].tolist() == [0] + [-1] * 23
    assert pos["path"][2].tolist() == [0] + [-1] * 23
    assert pos["path"][3].tolist() == [0, 1] + [-1] * 22
    assert pos["path"][4].tolist() == [-1] * 24
    toks, pos = pw.record(1)
    assert pos["path"][1].tolist() == [2] + [-1] * 23  # node ids are record-local
    x, y, pb = pw.get_batch("train", 1, torch.device("cpu"), generator=torch.Generator().manual_seed(0))
    assert list(pb) == ["path"]
    assert pb["path"].shape == (1, 4, 24) and pb["path"].dtype == torch.long
    assert pb["path"][0, 0].tolist() == [-1] * 24
    assert pb["path"][0, 3].tolist() == [0, 1] + [-1] * 22


def test_path_batches_pad_with_absent(tmp_path):
    pw = PackedWindows(_tiny_path_pack(tmp_path), sidecar="path")
    x, y, pb = pw.get_batch_records([0, 1], torch.device("cpu"))
    assert x.shape == (2, 4) and pb["path"].shape == (2, 4, 24)
    # record 1 has 4 tokens -> 3 targets; its 4th row is padding: -1 target, absent path
    assert y[1, 3] == -1 and x[1, 3] == 50256
    assert pb["path"][1, 3].tolist() == [-1] * 24


def test_no_sidecar_returns_no_positions(tmp_path):
    pw = PackedWindows(_tiny_path_pack(tmp_path), need_tree=False)
    assert pw.sidecar is None and not pw.need_tree
    _, pos = pw.record(0)
    assert pos is None
    _, _, pb = pw.get_batch("val", 1, torch.device("cpu"))
    assert pb is None


def test_path_sidecar_refuses_when_absent(tmp_path):
    d = _tiny_path_pack(tmp_path)
    import shutil
    shutil.rmtree(d + "_dompath")
    with pytest.raises(FileNotFoundError, match="DOM-path sidecar"):
        PackedWindows(d, sidecar="path")


def test_need_tree_still_selects_the_tree_sidecar(tmp_path):
    d = _tiny_path_pack(tmp_path)
    with pytest.raises(FileNotFoundError, match="tree sidecar"):
        PackedWindows(d, need_tree=True)
    with pytest.raises(ValueError):
        PackedWindows(d, sidecar="bogus")


def _tiny_leaf_pack(tmp_path):
    """v2 sidecar: 22 levels plus tok_within."""
    d = tmp_path / "pack2"
    s = tmp_path / "pack2_dompath"
    d.mkdir()
    s.mkdir()
    toks = np.array([50256, 1, 2, 3, 50256, 50256, 4, 5, 50256], dtype=np.uint16)
    np.save(d / "tokens.npy", toks)
    np.save(d / "tok_off.npy", np.array([0, 5, 9], dtype=np.int64))
    json.dump(dict(block_size=8, vocab_size=50304, eot=50256, n_train=1, n_val=1, n_test=0),
              open(d / "meta.json", "w"))
    np.save(s / "tok_node.npy", np.array([-1, 0, 0, 1, -1, -1, 0, 0, -1], dtype=np.int32))
    np.save(s / "tok_within.npy", np.array([-1, 0, 1, 0, -1, -1, 0, 1, -1], dtype=np.int16))
    np.save(s / "node_path.npy",
            np.array([[0] + [-1] * 21, [0, 1] + [-1] * 20, [2] + [-1] * 21], dtype=np.int16))
    np.save(s / "node_off.npy", np.array([0, 2, 3], dtype=np.int64))
    return str(d)


def test_v2_sidecar_appends_the_within_index_as_a_23rd_column(tmp_path):
    pw = PackedWindows(_tiny_leaf_pack(tmp_path), sidecar="path")
    assert pw.levels == 22 and pw.has_within
    toks, pos = pw.record(0)
    assert pos["path"].shape == (5, 23)
    assert pos["path"][0].tolist() == [-1] * 23                      # framing: absent everywhere
    assert pos["path"][1].tolist() == [0] + [-1] * 21 + [0]          # node 0, first token
    assert pos["path"][2].tolist() == [0] + [-1] * 21 + [1]          # node 0, second token
    assert pos["path"][3].tolist() == [0, 1] + [-1] * 20 + [0]       # node 1 restarts at 0
    x, y, pb = pw.get_batch_records([0, 1], torch.device("cpu"))
    assert pb["path"].shape == (2, 4, 23)
    assert pb["path"][1, 3].tolist() == [-1] * 23                    # padding row
    assert pb["path"][1, 2].tolist() == [2] + [-1] * 21 + [1]


def test_v1_sidecar_has_no_within_column(tmp_path):
    pw = PackedWindows(_tiny_path_pack(tmp_path), sidecar="path")
    assert not pw.has_within
    _, pos = pw.record(0)
    assert pos["path"].shape[1] == 24


def test_step_batches_groups_by_length_and_respects_the_budget(tmp_path):
    """The bucketer must return every sampled record exactly once, in groups that
    fit the token budget, with the group's padded length rounded to PAD_TO."""
    import torch
    from topos_data import PackedWindows

    pw = PackedWindows(_tiny_path_pack(tmp_path), sidecar="path")
    g = torch.Generator().manual_seed(0)
    groups, total = pw.step_batches("train", 8, token_budget=4096, generator=g)
    flat = [r for grp in groups for r in grp]
    assert len(flat) == 8
    lo, hi = pw.splits["train"]
    assert all(lo <= r < hi for r in flat)
    for grp in groups:
        lens = [int(pw.tok_off[r + 1]) - int(pw.tok_off[r]) - 1 for r in grp]
        t = min(pw.block_size, ((max(lens) + pw.PAD_TO - 1) // pw.PAD_TO) * pw.PAD_TO)
        assert len(grp) * t <= 4096 or len(grp) == 1
    assert total == sum(int(pw.tok_off[r + 1]) - int(pw.tok_off[r]) - 1 for r in flat)


def test_step_batches_draws_the_same_records_as_a_plain_sampler(tmp_path):
    """Only the GROUPING may change: the multiset of records for a step must be
    what torch.randint would have given with the same generator."""
    import torch
    from topos_data import PackedWindows

    pw = PackedWindows(_tiny_path_pack(tmp_path), sidecar="path")
    lo, hi = pw.splits["train"]
    want = sorted(torch.randint(lo, hi, (8,), generator=torch.Generator().manual_seed(7)).tolist())
    groups, _ = pw.step_batches("train", 8, token_budget=4096,
                                generator=torch.Generator().manual_seed(7))
    assert sorted(r for grp in groups for r in grp) == want
