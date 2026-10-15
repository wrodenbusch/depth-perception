"""Tests for the packed-window loader.

The first test exists because I got this wrong. The sidecar manifest says node ids
are "shifted by the window's running node base", which describes how a window is
*built* from its members — not how the arrays are *stored*. Stored, they are
record-local. I rebased them anyway on the first pass, which drove every id
negative, made every token read as outside the tree, and turned the ``tree`` arm
into the ``none`` arm.

**Nothing failed.** Both arms sit at ``ln(vocab)`` at initialisation, so the loss
looked exactly right. A 17-hour run would have returned "tree position does
nothing", which is a plausible result, and the bug would have been attributed to
the mechanism. The test below is the cheap check that catches it.

Run: ``python -m pytest test_topos_data.py -v``
"""
from topos_paths import PACKS  # noqa: E402

import os

import pytest
import torch

from topos_data import DEFAULT_PACK, PackedWindows

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_PACK),
    reason=f"pack not present at {DEFAULT_PACK}; set TOPOS_PACK",
)


@pytest.fixture(scope="module")
def pw():
    return PackedWindows()


def test_node_ids_are_record_local_and_index_real_nodes(pw):
    """THE bug. A record's largest causal id must be exactly ``n_nodes - 1``.

    Too large and the coordinates index another record's nodes. Negative-only and
    every token reads as outside the tree and the arm silently becomes ``none``.
    """
    for r in (0, 1, 7, 100, 1000):
        toks, pos = pw.record(r)
        n_nodes = len(pos["parent"])
        assert pos["causal"].max() == n_nodes - 1, (
            f"record {r}: max causal id {pos['causal'].max()} but {n_nodes} nodes — "
            "ids are not record-local, or have been rebased"
        )
        assert pos["parent"].max() < n_nodes
        assert (pos["causal"] >= 0).mean() > 0.5, "most tokens should be inside the tree"


def test_a_window_is_a_forest_not_a_tree(pw):
    """One root per member program. A single root would mean the window had been
    given a synthetic parent, which would leak which program a token is in —
    exactly the information the experiment withholds."""
    for r in (0, 1, 7):
        _, pos = pw.record(r)
        roots = (pos["parent"] == -1).sum()
        assert roots > 1, f"record {r} has {roots} roots; a window should be a forest"


def test_splits_do_not_overlap_and_cover_every_record(pw):
    tr, va, te = pw.splits["train"], pw.splits["val"], pw.splits["test"]
    assert tr[0] == 0 and te[1] == pw.n_records
    assert tr[1] == va[0] and va[1] == te[0], "splits must be contiguous and disjoint"
    assert tr[1] - tr[0] == int(pw.meta["n_train"])
    assert va[1] - va[0] == int(pw.meta["n_val"])


def test_padding_is_ignored_by_the_loss_not_learned(pw):
    """Windows are variable length (only ~0.9% are exactly block_size). Padded
    target positions must be -1, which nanoGPT's cross_entropy ignores. Padding
    with 0 would train the model to predict EOT on several percent of positions
    and would never surface as an error."""
    x, y, _ = pw.get_batch("train", 8, torch.device("cpu"), generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape
    pad = (y == -1)
    assert pad.any(), "a batch of 8 variable-length windows should contain padding"
    assert (y[~pad] >= 0).all()


def test_batch_coordinates_stay_in_range_after_padding(pw):
    """Padding a batch to the longest window must not let one record's ids point
    into the padded region of the node tables."""
    _, _, pos = pw.get_batch("train", 8, torch.device("cpu"), generator=torch.Generator().manual_seed(1))
    assert pos["causal"].max().item() < pos["parent"].shape[1]
    assert pos["causal"].min().item() == -1, "padding and framing should read as outside the tree"


def test_tree_arm_refuses_when_the_sidecar_is_absent(tmp_path):
    """A missing sidecar must raise, not yield an arm with no coordinates."""
    import json
    import shutil

    fake = tmp_path / "pack"
    fake.mkdir()
    shutil.copy(os.path.join(DEFAULT_PACK, "meta.json"), fake / "meta.json")
    for name in ("tokens.npy", "tok_off.npy"):
        os.symlink(os.path.join(DEFAULT_PACK, name), fake / name)
    with pytest.raises(FileNotFoundError, match="tree sidecar"):
        PackedWindows(str(fake), need_tree=True)
    # ... but the position-free arms are fine without it
    assert PackedWindows(str(fake), need_tree=False).n_records > 0


def test_all_three_arms_run_on_real_data(pw):
    """End to end, on the actual pack."""
    from model import GPT, GPTConfig

    x, y, pos = pw.get_batch("val", 2, torch.device("cpu"), generator=torch.Generator().manual_seed(2))
    uniform = torch.log(torch.tensor(float(pw.vocab_size))).item()
    for mode in ("learned", "none", "tree"):
        cfg = GPTConfig(block_size=1024, vocab_size=pw.vocab_size, n_layer=2,
                        n_head=4, n_embd=128, pos_mode=mode)
        torch.manual_seed(0)
        model = GPT(cfg).eval()
        with torch.no_grad():
            _, loss = model(x, targets=y, pos=pos if mode == "tree" else None)
        # An untrained model must sit at the uniform floor; far from it means the
        # targets or the padding are wrong, not that the model is good.
        assert abs(loss.item() - uniform) < 0.5, f"{mode}: loss {loss.item()} vs uniform {uniform}"


# --------------------------------------------------------------------------------------
# THE SYMBOL SLOT (rope_ref_slot). The loader must REFUSE when the arm needs tok_slot and
# the sidecar lacks it: without the refusal the arm silently degrades to rope_both, whose
# loss curve is indistinguishable at a glance.
# --------------------------------------------------------------------------------------
PY_PACK = f"{PACKS}/py_ast_423k"


@pytest.mark.skipif(not os.path.isdir(PY_PACK + "_dompath"), reason="py dompath sidecar absent")
def test_slot_arm_loads_and_every_slot_is_in_range():
    pw = PackedWindows(PY_PACK, need_tree=False, sidecar="path", need_slot=True)
    assert pw.has_slot and pw.need_slot
    x, y, pos = pw.get_batch_records([0, 1, 5, 17], torch.device("cpu"))
    assert "slot" in pos and pos["slot"].shape == x.shape
    s = pos["slot"]
    assert s.min() >= -1, "a slot below ABSENT would index the step table negatively"
    # 2 planes reach ~318 before wrapping; the whole pack peaks far below that
    assert s.max() < 318, f"slot {int(s.max())} would alias on the slow plane"
    # Coverage is BIMODAL: only 38.29% of records have any slot-bearing token (median
    # per-record coverage 0.000, p90 0.968), because a record needs a name whose
    # delimiter is consumed while tok_causal still points at its node. So a specific
    # small batch may legitimately be all-ABSENT; record 1000 is one that does resolve.
    _, _, pos2 = pw.get_batch_records([1000], torch.device("cpu"))
    assert (pos2["slot"] >= 0).any(), "record 1000 is known to carry slots"
    assert pos2["slot"].max() < 318


@pytest.mark.skipif(not os.path.isdir(PY_PACK + "_dompath"), reason="py dompath sidecar absent")
def test_slot_padding_is_absent_not_zero():
    """A pad position carries no symbol, so its slot must be ABSENT. Padding with 0 would
    make every pad token claim to be symbol 0 and rotate the slot planes accordingly."""
    pw = PackedWindows(PY_PACK, need_tree=False, sidecar="path", need_slot=True)
    ix = [0, 1, 2, 3]
    lens = [int(pw.tok_off[r + 1]) - int(pw.tok_off[r]) - 1 for r in ix]
    x, y, pos = pw.get_batch_records(ix, torch.device("cpu"))
    for i, t in enumerate(lens):
        if t < x.shape[1]:
            assert (pos["slot"][i, t:] == -1).all(), "padding must be ABSENT in the slot"


def test_slot_arm_refuses_when_tok_slot_is_absent(tmp_path):
    import json
    import shutil

    fake = tmp_path / "pack"
    side = tmp_path / "pack_dompath"
    fake.mkdir()
    side.mkdir()
    shutil.copy(os.path.join(PY_PACK, "meta.json"), fake / "meta.json")
    for name in ("tokens.npy", "tok_off.npy"):
        os.symlink(os.path.join(PY_PACK, name), fake / name)
    for name in ("tok_node.npy", "node_path.npy", "node_off.npy", "tok_within.npy"):
        os.symlink(os.path.join(PY_PACK + "_dompath", name), side / name)
    # the path arms are content with this sidecar ...
    assert PackedWindows(str(fake), need_tree=False, sidecar="path").n_records > 0
    # ... but the reference arm must refuse rather than train without its coordinate
    with pytest.raises(FileNotFoundError, match="tok_slot"):
        PackedWindows(str(fake), need_tree=False, sidecar="path", need_slot=True)
