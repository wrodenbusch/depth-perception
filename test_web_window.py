"""``--window`` must keep every token and move no token's tree coordinate.

WHY THIS TEST EXISTS. Cropping long pages threw away 11.4% of all text tokens at a 4,096
context and would have thrown away 42% at 1,024, so the builder gained a windowing mode. The
risk in windowing is not that it loses text -- that is easy to see -- but that a window past
the first silently *shifts* which node row a token resolves to, because each record carries
its own trimmed node table. A shifted row is invisible: the pack builds, the model trains, and
every token in the back half of a long page is simply assigned somebody else's tree position.

So the invariant under test is not "the same number of tokens" but **the same (token, tree
path) pairs, in the same order, bit for bit** -- compared against the unwindowed render of the
same page, which is the thing the existing packs were built from.
"""
from __future__ import annotations

import numpy as np
import pytest

from build_web_pack import BLOCK, split_windows, tokenise_page, trim_nodes

# A synthetic page: 40 text nodes at varied depths, long enough to need several windows.
PAGE = [(f"Sentence number {i} about the subject of paragraph {i // 4}, with enough words in "
         f"it to make a few dozen tokens all on its own line here.", [i // 4, i % 4])
        for i in range(40)]


def resolve(ids, tn, npth):
    """(token id, its tree path or None) for every token -- what the model actually sees."""
    npth = trim_nodes(tn, npth)
    out = []
    for t, node in zip(ids, tn):
        out.append((int(t), None if node < 0 else tuple(npth[node].tolist())))
    return out


@pytest.fixture(scope="module")
def enc():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def test_split_windows_is_lossless_and_bounded():
    ids = np.arange(1100, dtype=np.uint16)
    wins = split_windows(ids, ids.astype(np.int32), ids.astype(np.int16), 1024)
    assert len(wins) == 2
    assert np.array_equal(np.concatenate([w[0] for w in wins]), ids)
    assert all(0 < len(w[0]) <= 1024 for w in wins)
    # near-equal, not 1024 + a 76-token runt
    assert abs(len(wins[0][0]) - len(wins[1][0])) <= 1


def test_a_short_page_is_one_window_unchanged():
    ids = np.arange(300, dtype=np.uint16)
    wins = split_windows(ids, ids.astype(np.int32), ids.astype(np.int16), 1024)
    assert len(wins) == 1 and np.array_equal(wins[0][0], ids)


@pytest.mark.parametrize("block", [256, 512, 1024])
def test_windowing_preserves_every_token_and_its_tree_path(enc, block):
    """THE invariant: windowed records concatenate to exactly the unwindowed page."""
    ids, tn, npth, tw, st = tokenise_page(PAGE, enc, block=BLOCK, levels=24, window=True)
    assert not st["cropped"] and len(ids) == st["full_len"]
    whole = resolve(ids, tn, npth)

    wins = split_windows(ids, tn, tw, block)
    assert len(wins) > 1, "the fixture page must be long enough to window"
    pieced = [pair for w_ids, w_tn, _ in wins for pair in resolve(w_ids, w_tn, npth)]

    assert len(pieced) == len(whole), f"{len(whole) - len(pieced)} tokens lost"
    assert pieced == whole, "a token's tree path moved when the page was windowed"


@pytest.mark.parametrize("block", [256, 512, 1024])
def test_within_node_index_travels_with_its_token(enc, block):
    """``tok_within`` is sliced alongside the ids; rope_leaf reads it, so it must stay aligned."""
    ids, tn, npth, tw, _ = tokenise_page(PAGE, enc, block=BLOCK, levels=24, window=True)
    wins = split_windows(ids, tn, tw, block)
    assert np.array_equal(np.concatenate([w[2] for w in wins]), tw)


def test_cropping_is_still_the_default_and_still_crops(enc):
    """The old behaviour must be untouched, or existing packs stop reproducing."""
    ids, tn, npth, tw, st = tokenise_page(PAGE, enc, block=512, levels=24)
    assert st["cropped"] and len(ids) == 512
    assert st["full_len"] > 512


def test_trim_keeps_a_prefix_so_ids_never_move():
    npth = np.arange(50, dtype=np.int16).reshape(50, 1)
    tn = np.array([-1, 7, 30, 12, -1], dtype=np.int32)      # a mid-page window's id range
    kept = trim_nodes(tn, npth)
    assert len(kept) == 31
    for node in (7, 30, 12):
        assert kept[node][0] == npth[node][0], "a node id resolved to a different row"


def test_no_tokens_means_no_node_rows():
    assert len(trim_nodes(np.array([-1, -1], dtype=np.int32),
                          np.zeros((9, 3), dtype=np.int16))) == 0


# --- the guard: windowing is only safe for a stream of leaves -------------------------------
# The web TEXT render keeps text nodes only, whose ancestors are elements that emit no tokens,
# so no parent-child pair is in the stream and windowing cannot corrupt depth. An AST corpus is
# the opposite: interior nodes throughout. The builder must refuse to window that shape.

#: A page whose nodes DO contain each other -- [0] is an ancestor of [0,0], as in an AST.
NESTED = [("the outer node's own text, long enough to be several tokens", [0]),
          ("a child of the outer node with its own words here", [0, 0]),
          ("a grandchild deeper still, also carrying text", [0, 0, 0])]


def test_interior_nodes_are_counted(enc):
    _, _, _, _, st = tokenise_page(NESTED, enc, block=BLOCK, levels=24, window=True)
    assert st["nodes_interior"] == 2, "[0] and [0,0] each have a descendant in the stream"


def test_a_leaf_only_page_reports_no_interior_nodes(enc):
    """The web text render's actual shape: sibling text nodes, none containing another."""
    _, _, _, _, st = tokenise_page(PAGE, enc, block=BLOCK, levels=24, window=True)
    assert st["nodes_interior"] == 0
