"""Tests for build_web_pack.py: the visible-text render and the DOM-path sidecar rules."""
import numpy as np
import tiktoken

from build_web_pack import extract_page, tokenise_page, too_deep, LEVELS, ABSENT, EOT, SKIP_TAGS

HTML = ("<html><head><title>T</title><style>p{}</style></head>"
        "<body><div class='a'>Hello <b>bold</b>world<script>var x;</script>"
        "<ul><li>One<li>Two</ul><!-- c --></div><p>  spaced   out </p></body></html>")


def test_extract_paths_start_below_body_and_skip_dropped_subtrees():
    nodes = extract_page(HTML)
    texts = [t for t, _ in nodes]
    assert texts == ["Hello", "bold", "world", "One", "Two", "spaced out"]
    paths = [p for _, p in nodes]
    # body > div(0) > text(0), b(1) > text(0), text(2), [script: dropped, takes no rank],
    #            ul(3) > li(0) > text(0), li(1) > text(0);  body > p(1) > text(0)
    assert paths[0] == [0, 0]
    assert paths[1] == [0, 1, 0]
    assert paths[2] == [0, 2]
    assert paths[3] == [0, 3, 0, 0]
    assert paths[4] == [0, 3, 1, 0]
    assert paths[5] == [1, 0]


def test_head_is_a_skip_tag():
    assert "head" in SKIP_TAGS and "script" in SKIP_TAGS


def test_extract_returns_none_without_body_text():
    assert extract_page("") in (None, [])


def test_tokenise_assigns_tokens_and_newlines_to_nodes():
    enc = tiktoken.get_encoding("gpt2")
    nodes = [("Hello", [0, 0]), ("world", [0, 2])]
    ids, tok_node, node_path, tok_within, stats = tokenise_page(nodes, enc)
    assert ids[0] == EOT and ids[-1] == EOT
    assert tok_node[0] == ABSENT and tok_node[-1] == ABSENT
    text = enc.decode([int(i) for i in ids[1:-1]])
    assert text == "Hello\nworld"
    body = ids[1:-1]
    dec = [enc.decode([int(i)]) for i in body]
    for d, n in zip(dec, tok_node[1:-1]):
        assert n == (0 if d.strip() in ("Hello", "") or d == "\n" else 1), (d, n)
    assert node_path.shape == (2, LEVELS)
    assert node_path[0].tolist() == [0, 0] + [ABSENT] * (LEVELS - 2)
    assert stats["tokens_kept_in_tree"] == len(body)


def test_tokenise_truncates_deep_paths_at_leaf_end_and_crops():
    enc = tiktoken.get_encoding("gpt2")
    deep = list(range(30))
    nodes = [("x " * 50, deep)]
    ids, tok_node, node_path, tok_within, stats = tokenise_page(nodes, enc, block=16)
    assert node_path[0].tolist() == deep[:LEVELS]
    assert stats["nodes_truncated_path"] == 1
    assert len(ids) == 16 and stats["cropped"] is True
    assert len(tok_node) == len(ids) == len(tok_within)


def test_within_index_restarts_at_every_node_and_counts_the_newline_last():
    enc = tiktoken.get_encoding("gpt2")
    nodes = [("Hello big world", [0, 0]), ("again", [0, 2])]
    ids, tok_node, node_path, tok_within, stats = tokenise_page(nodes, enc)
    assert tok_within.dtype == np.int16
    assert tok_within[0] == ABSENT and tok_within[-1] == ABSENT     # framing, like tok_node
    body_node, body_within = tok_node[1:-1], tok_within[1:-1]
    n0 = body_within[body_node == 0].tolist()
    n1 = body_within[body_node == 1].tolist()
    assert n0 == list(range(len(n0))) and len(n0) >= 4          # "Hello"," big"," world","\n"
    assert n1 == list(range(len(n1)))                            # restarts at 0
    dec = [enc.decode([int(i)]) for i in ids[1:-1]]
    nl = dec.index("\n")
    assert tok_node[1 + nl] == 0 and tok_within[1 + nl] == n0[-1]  # the newline is node 0's LAST token
    assert (tok_within >= 0).sum() == (tok_node >= 0).sum()


def test_too_deep_drops_a_page_past_the_level_cap():
    assert too_deep([("x", list(range(23)))], 22) is True
    assert too_deep([("x", list(range(22))), ("y", [0])], 22) is False


def test_levels_22_with_drop_deep_truncates_nothing():
    enc = tiktoken.get_encoding("gpt2")
    nodes = [("x", list(range(22)))]
    assert not too_deep(nodes, 22)
    ids, tok_node, node_path, tok_within, stats = tokenise_page(nodes, enc, levels=22)
    assert node_path.shape == (1, 22) and stats["nodes_truncated_path"] == 0
    assert node_path[0].tolist() == list(range(22))


# ---------------------------------------------------------------- html render

HTML_MARKUP = (
    "<html><head><title>T</title></head><body>"
    "<div class='a' id='d1'>Hi<br><img src='x' alt='q&amp;r'/>"
    "<script>var x;</script><span>deep</span></div>"
    "</body></html>"
)


def test_html_pieces_concatenate_to_the_serialisation():
    from build_web_pack import extract_html_page
    pieces, paths, kinds = extract_html_page(HTML_MARKUP)
    doc = "".join(s for s, _ in pieces)
    assert doc == (
        '<div class="a" id="d1">'
        'Hi'
        '<br/>'
        '<img src="x" alt="q&amp;r"/>'
        '<span>deep</span>'
        '</div>'
    )


def test_html_childless_elements_self_close_and_parents_wrap():
    from build_web_pack import extract_html_page
    pieces, paths, kinds = extract_html_page(HTML_MARKUP)
    strings = [s for s, _ in pieces]
    assert "/>" in strings                      # br and img close themselves
    assert ">" in strings and "</div>" in strings


def test_html_attribute_ranks_precede_child_ranks():
    from build_web_pack import extract_html_page
    pieces, paths, kinds = extract_html_page(HTML_MARKUP)
    # div is the first child of body -> path [0]; its two attributes are its
    # children 0 and 1, so its first CONTENT child ("Hi") is rank 2.
    assert paths[0] == [0]                      # div
    assert paths[1] == [0, 0]                   # class
    assert paths[2] == [0, 1]                   # id
    assert paths[3] == [0, 2]                   # text "Hi"
    assert kinds[0] == "element" and kinds[1] == "attr" and kinds[3] == "text"


def test_html_skipped_subtree_contributes_no_node_and_no_token():
    from build_web_pack import extract_html_page
    pieces, paths, kinds = extract_html_page(HTML_MARKUP)
    doc = "".join(s for s, _ in pieces)
    assert "script" not in doc and "var x" not in doc
    # the span after the dropped script keeps the rank the script did not take
    span = [p for p, k in zip(paths, kinds) if k == "element" and p[:1] == [0] and len(p) == 2]
    assert [0, 5] in span, span                 # class,id,Hi,br,img,span -> rank 5


def test_html_tokens_land_on_the_element_and_its_attribute():
    import tiktoken
    from build_web_pack import extract_html_page, tokenise_html_page
    enc = tiktoken.get_encoding("gpt2")
    pieces, paths, kinds = extract_html_page(HTML_MARKUP)
    ids, tok_node, node_path, tok_within, stats = tokenise_html_page(pieces, paths, kinds, enc)
    dec = [enc.decode([int(i)]) for i in ids]
    # the '<div' token belongs to the div element node (id 0), ' class="a"' to node 1
    first = next(i for i, d in enumerate(dec) if "div" in d)
    assert tok_node[first] == 0
    cls = next(i for i, d in enumerate(dec) if "class" in d)
    assert tok_node[cls] == 1
    assert 0.0 < stats["tokens_markup"] / stats["tokens_kept_in_tree"] <= 1.0


def test_html_markup_fraction_dominates_a_tag_heavy_page():
    import tiktoken
    from build_web_pack import extract_html_page, tokenise_html_page
    enc = tiktoken.get_encoding("gpt2")
    tags = "".join(f'<p class="c{i}">t</p>' for i in range(30))
    pieces, paths, kinds = extract_html_page(f"<html><body>{tags}</body></html>")
    ids, tok_node, node_path, tok_within, stats = tokenise_html_page(pieces, paths, kinds, enc)
    assert stats["tokens_markup"] / stats["tokens_kept_in_tree"] > 0.5


def test_html_within_counts_per_node_not_distance_from_first():
    """An element's pieces are split by its children, so within must COUNT the
    node's own tokens -- a distance-from-first would give '</div>' a huge index
    that wraps the token-leaf planes."""
    import tiktoken
    from build_web_pack import extract_html_page, tokenise_html_page
    enc = tiktoken.get_encoding("gpt2")
    body = "".join(f"<span>word{i}</span>" for i in range(20))
    pieces, paths, kinds = extract_html_page(f"<html><body><div>{body}</div></body></html>")
    ids, tok_node, node_path, tok_within, stats = tokenise_html_page(pieces, paths, kinds, enc)
    inside = tok_within[tok_node >= 0]
    assert inside.max() < 20, inside.max()


def test_text_render_is_unchanged_by_the_html_work():
    """Pin the text path: same nodes, same tokens, same sidecar as before."""
    enc = tiktoken.get_encoding("gpt2")
    nodes = extract_page(HTML)
    assert [t for t, _ in nodes] == ["Hello", "bold", "world", "One", "Two", "spaced out"]
    ids, tok_node, node_path, tok_within, stats = tokenise_page(nodes, enc)
    assert enc.decode([int(i) for i in ids[1:-1]]) == "Hello\nbold\nworld\nOne\nTwo\nspaced out"
    assert tok_node[0] == ABSENT and tok_node[-1] == ABSENT
    assert stats["tokens_markup"] == 0
    assert node_path[1].tolist() == [0, 1, 0] + [ABSENT] * (LEVELS - 3)


def test_emit_trims_node_rows_the_tokens_cannot_reach(tmp_path):
    """A cropped record must not carry node rows no token references.

    The node array covers the whole page; the token array is cropped at block_size.
    Every node id in tok_node must still resolve, and nothing beyond the largest
    referenced id should be stored.
    """
    import subprocess, sys, json
    import numpy as np

    out = str(tmp_path / "pk")
    r = subprocess.run([sys.executable, "build_web_pack.py", "--out", out, "--render", "html",
                        "--shards", "4", "--lines-per-shard", "1500", "--pages", "5000",
                        "--workers", "4", "--levels", "22", "--drop-deep"],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    side = out + "_dompath"
    toff = np.load(out + "/tok_off.npy"); noff = np.load(side + "/node_off.npy")
    tn = np.load(side + "/tok_node.npy", mmap_mode="r")
    npth = np.load(side + "/node_path.npy", mmap_mode="r")
    assert len(npth) == noff[-1]
    for rec in range(0, len(toff) - 1, 37):
        t = np.asarray(tn[toff[rec]:toff[rec + 1]]); t = t[t >= 0]
        n_have = int(noff[rec + 1] - noff[rec])
        if len(t) == 0:
            continue
        assert int(t.max()) < n_have, "a token references a node row that was trimmed away"
        assert n_have == int(t.max()) + 1, f"record {rec} keeps {n_have} rows, needs {int(t.max())+1}"


def test_drop_alias_removes_only_pages_the_rotation_cannot_separate(tmp_path):
    """--drop-alias must leave no rank >= MAX_RANK and no within >= MAX_WITHIN,
    and must not fire when the page is inside those bounds."""
    import subprocess, sys
    import numpy as np
    from build_web_pack import MAX_RANK, MAX_WITHIN

    out = str(tmp_path / "pk")
    r = subprocess.run([sys.executable, "build_web_pack.py", "--out", out, "--render", "html",
                        "--shards", "8", "--lines-per-shard", "2000", "--pages", "9000",
                        "--workers", "4", "--levels", "22", "--drop-deep", "--drop-alias"],
                       capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, r.stderr[-2000:]
    side = out + "_dompath"
    npth = np.load(side + "/node_path.npy", mmap_mode="r")
    tw = np.load(side + "/tok_within.npy", mmap_mode="r")
    assert int(np.asarray(npth[:2_000_000]).max()) < MAX_RANK
    assert int(np.asarray(tw[:2_000_000]).max()) < MAX_WITHIN
    import json
    m = json.load(open(side + "/manifest.json"))
    assert m["drop_alias"] is True and m["max_rank"] == MAX_RANK and m["max_within"] == MAX_WITHIN


def test_drop_alias_is_off_by_default(tmp_path):
    """Without the flag the builder must behave exactly as before."""
    import json, subprocess, sys
    out = str(tmp_path / "pk2")
    r = subprocess.run([sys.executable, "build_web_pack.py", "--out", out,
                        "--shards", "8", "--lines-per-shard", "2000", "--pages", "9000",
                        "--workers", "4", "--levels", "22", "--drop-deep"],
                       capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, r.stderr[-2000:]
    m = json.load(open(out + "_dompath/manifest.json"))
    assert m["drop_alias"] is False and m["max_rank"] is None


def test_drop_long_leaves_every_record_a_whole_page(tmp_path):
    """``--drop-long``: a page that does not fit is left OUT, never cut down.

    The author's requirement was that nothing be truncated. Cropping is the builder's default and
    discards 11.4% of text at 4,096; ``--window`` keeps everything by splitting pages, which is
    unsafe for a tree whose interior nodes are tokenised; this mode is the third answer -- keep
    whole pages only. The assertion that matters is not the page count but that NO record was
    cropped, since a silently cropped record is indistinguishable from a short page.
    """
    import json, subprocess, sys
    out = str(tmp_path / "whole")
    r = subprocess.run([sys.executable, "build_web_pack.py", "--out", out,
                        "--block", "1024", "--drop-long",
                        # MORE INPUT THAN THE OTHER END-TO-END TESTS ON PURPOSE: dropping long
                        # pages at 1,024 keeps only ~34% of pages against ~55% under cropping,
                        # so the usual budget never reaches the 4,096 held-out pages the
                        # builder requires and the build refuses before it can be checked.
                        "--shards", "8", "--lines-per-shard", "5000", "--pages", "12000",
                        "--workers", "4", "--levels", "22", "--drop-deep"],
                       capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, r.stderr[-2000:]
    m = json.load(open(out + "/meta.json"))
    assert m["drop_long"] is True and m["block_size"] == 1024
    assert m["records_cropped_frac"] == 0.0
    assert m["text_tokens_lost_to_crop_frac"] == 0.0
    assert m["pages_dropped_long"] > 0, "no page was long enough to exercise the mode"
    off = np.load(out + "/tok_off.npy")
    assert int((off[1:] - off[:-1]).max()) <= 1024


def test_window_and_drop_long_are_refused_together(tmp_path):
    """They are alternatives: one keeps every token, the other keeps every record whole."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "build_web_pack.py", "--out", str(tmp_path / "x"),
                        "--block", "1024", "--drop-long", "--window",
                        "--shards", "1", "--lines-per-shard", "50", "--pages", "50",
                        "--workers", "2", "--levels", "22"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    assert "alternatives, not a pair" in (r.stdout + r.stderr)
