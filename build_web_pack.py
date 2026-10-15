#!/usr/bin/env python
"""Raw C4 HTML pages -> visible-text pack + per-token DOM-path sidecar.

    .venv/bin/python build_web_pack.py \
        --out $TOPOS_PACKS/web_c4html_text_gpt2_b4096 --pages 600000 --workers 64

THE TOKEN STREAM IS THE PAGE'S VISIBLE TEXT AND NOTHING ELSE. Tags, attributes,
scripts, styles are gone. Where a token sits in the DOM is carried only by the
sidecar, which only the graph arms read. The spec
(docs/superpowers/specs/2026-09-21-web-graph-rotation-design.md) states every
rule below; the manifest restates them next to the numbers they produce.

WHAT A NODE IS
--------------
Element nodes not under a skipped tag, and text nodes with any non-whitespace
character. A node's RANK is its index among its parent's RETAINED children
(elements and kept text nodes, flat, document order). Its PATH is the list of
ranks from the child of <body> down to itself: <html> and <body> are the same
on every page and are not levels. Paths longer than LEVELS are truncated at the
LEAF end, so level k always means tree level k.

WHAT A TOKEN IS
---------------
Kept text nodes are joined by a single newline, framed [EOT] text [EOT], GPT-2
BPE. A token belongs to the text node whose byte span holds its first
non-whitespace byte; the newline after node i belongs to node i. Framing EOTs
are ABSENT on every level. Records are cropped to BLOCK tokens -- or, under
``--window``, a long page becomes several records and nothing is discarded.

EVERY TOKEN IS A LEAF (sidecar v2)
----------------------------------
``tok_within.npy`` gives each token its index within its text node (0 for the
node's first token; the node's trailing newline is its last token; -1 for the
framing EOTs). Appended to the node path it makes every token's coordinate
unique -- the ``rope_leaf`` arm reads it, the other arms ignore it.

DEEP PAGES ARE DROPPED, NOT TRUNCATED (``--drop-deep``)
-------------------------------------------------------
With ``--drop-deep`` a page whose deepest node exceeds ``--levels`` is left out
of the pack entirely, so no coordinate in the pack is ever truncated. Measured
on the cell-1 pack: 10.5% of pages exceed 22 levels below <body>, 8.9% exceed 23.

NO LOOKAHEAD, BY CONSTRUCTION
-----------------------------
Emission is document order and a rank counts earlier siblings only, so every
coordinate is known from the prefix. There is nothing to measure.
"""
from __future__ import annotations
from topos_paths import HF_HOME  # noqa: E402

import argparse
import glob
import hashlib
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from npy_stream import StreamNpy

SNAPSHOT_GLOB = (f"{HF_HOME}/hub/datasets--bs-modeling-metadata--"
                 "c4-en-html-with-metadata/snapshots/*/c4-en-html_*.jsonl.gz")
SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "iframe", "head"})
LEVELS = 24
ABSENT = -1
EOT = 50256
BLOCK = 4096
MIN_TEXT_TOKENS = 64
VOCAB_PADDED = 50304
SPLIT_SEED = 0
N_VAL = 2048
N_TEST = 2048
FAST_WRAP_RANK = 19      # rank at which the fast plane (step 0.316) has wrapped
#: A level has TWO planes, the slow one 16x slower, so a rank stays UNIQUE until the
#: slow plane also turns over -- at 2*pi / (0.3162/16) = 318. The token level has FOUR
#: planes, its slowest 4**3 times slower, so a within-node index is unique to 1,272.
#: Past those two numbers the coordinate genuinely collides with one a period earlier.
MAX_RANK = 318
MAX_WITHIN = 1272
SLOW_WRAP_RANK = 318     # rank at which the slow plane (step 0.316/16) has wrapped
_WS = b" \t\r\n\x0b\x0c"


def extract_page(html: str):
    """Kept text nodes in document order: [(text, rank_path_from_below_body)]. None if no <body>.

    Backend: selectolax's MODEST engine, not lexbor. On shard 1 line 659 of the
    C4 snapshot lexbor builds a tree whose child pointers segfault the process
    on traversal (reproduced with iter() and with child/next alike, selectolax
    0.4.12); Modest parses the same page cleanly. The walk is iterative so a
    pathological nesting depth cannot overflow the C stack either.
    """
    from selectolax.parser import HTMLParser
    try:
        tree = HTMLParser(html)
    except Exception:
        return None
    body = tree.body
    if body is None:
        return None
    out = []
    # Explicit stack of (node, path). Children are pushed in reverse so they pop
    # in document order; a child's rank is decided when its parent is expanded,
    # counting only retained siblings before it.
    stack = [(body, [])]
    while stack:
        node, path = stack.pop()
        if isinstance(node, str):             # a deferred text item: emit in order
            out.append((node, path))
            continue
        rank = 0
        pending = []
        for c in node.iter(include_text=True):
            tag = c.tag
            if tag == "-text":
                t = " ".join(c.text(deep=False).split())
                if t:
                    pending.append((t, path + [rank]))
                    rank += 1
                continue
            if tag is None or tag.startswith("-") or tag in SKIP_TAGS:
                continue                      # comments, doctype, dropped subtrees: no rank
            pending.append((c, path + [rank]))
            rank += 1
        # Push this node's items in reverse so they pop in document order; a
        # child element's subtree is fully emitted before the next sibling pops.
        for item in reversed(pending):
            stack.append(item)
    return out


def _tokenise(pieces, node_of_piece, paths, markup, enc, block, levels, join, window=False):
    """The shared core of both renders.

    ``pieces``         the strings that concatenate (through ``join``) to the record
    ``node_of_piece``  which node each piece belongs to -- an element contributes
                       SEVERAL pieces (``<div``, ``>``, ``</div>``), so this is not
                       the identity outside the text render
    ``paths``          node id -> rank path
    ``markup``         node id -> True when the node is an element or an attribute
    """
    text = join.join(pieces)
    body = enc.encode_ordinary(text)
    # byte span starts: piece i owns [start_i, start_{i+1}), including its separator
    starts = np.zeros(len(pieces) + 1, dtype=np.int64)
    acc = 0
    sep = len(join.encode("utf-8"))
    for i, t in enumerate(pieces):
        starts[i] = acc
        acc += len(t.encode("utf-8")) + sep
    starts[-1] = acc
    # first non-whitespace byte of each token (or its first byte if all whitespace)
    firsts = np.empty(len(body), dtype=np.int64)
    off = 0
    for j, tid in enumerate(body):
        b = enc.decode_single_token_bytes(tid)
        k = 0
        while k < len(b) and b[k] in _WS:
            k += 1
        firsts[j] = off + (k if k < len(b) else 0)
        off += len(b)
    piece_of_tok = np.searchsorted(starts[:-1], firsts, side="right") - 1
    tok_node = np.asarray(node_of_piece, dtype=np.int32)[piece_of_tok] if len(body)\
        else np.zeros(0, dtype=np.int32)
    # WITHIN-NODE INDEX: the token's ORDINAL among its own node's tokens, counted.
    # On the text render a node's tokens are contiguous, so this equals the old
    # distance-from-the-node's-first-token and the text pack is unchanged. On the
    # html render an element's tokens are split apart by its children, and a
    # distance would hand `</div>` an index of thousands -- far past the wrap of
    # the token-leaf planes that rope_leaf rotates on.
    if len(tok_node):
        order = np.argsort(tok_node, kind="stable")
        grouped = tok_node[order]
        grp_start = np.searchsorted(grouped, grouped, side="left")
        within = np.empty(len(order), dtype=np.int64)
        within[order] = np.arange(len(order)) - grp_start
        within = np.minimum(within, np.iinfo(np.int16).max).astype(np.int16)
    else:
        within = np.zeros(0, dtype=np.int16)
    node_path = np.full((len(paths), levels), ABSENT, dtype=np.int16)
    n_trunc = 0
    for i, p in enumerate(paths):
        if len(p) > levels:
            n_trunc += 1
        q = np.minimum(np.asarray(p[:levels], dtype=np.int64), np.iinfo(np.int16).max)
        node_path[i, :len(q)] = q
    ids = np.concatenate([[EOT], body, [EOT]]).astype(np.uint16)
    tn = np.concatenate([[ABSENT], tok_node, [ABSENT]]).astype(np.int32)
    tw = np.concatenate([[ABSENT], within, [ABSENT]]).astype(np.int16)
    full = len(ids)
    cropped = (full > block) and not window
    # WINDOW RATHER THAN CROP (2026-09-30). Cropping is one line, and it is expensive: the
    # 4,096 pack's own manifest records 11.4% of all text tokens discarded, and at 1,024 it
    # would be 42%. Under ``--window`` the page is left whole here and split into records by
    # ``emit``, which keeps every token.
    #
    # WHY THE SPLIT IS SAFE FOR BOTH ARMS, which is not obvious. Rotary attention depends
    # only on the DIFFERENCE of two positions, so a window that restarts its sequence
    # index at 0 is indistinguishable from one carrying its true document offset -- the
    # sequence arm loses nothing. And ``node_path`` holds each node's absolute rank path
    # from the root, so a window's tree coordinates stay globally valid however deep into
    # the page it begins. Neither arm is advantaged by where a boundary falls.
    if not window:
        ids, tn, tw = ids[:block], tn[:block], tw[:block]
    # INTERIOR NODES: a node with a descendant that is ALSO in the token stream. The count
    # gates ``--window`` (see the refusal in main), because windowing is only safe for a tree
    # whose stream holds no parent-child pair. Sorted order puts a path immediately before its
    # descendants, so testing the next entry finds every interior node in one pass.
    sorted_paths = sorted(tuple(p) for p in paths)
    n_interior = sum(1 for i, q in enumerate(sorted_paths[:-1])
                     if sorted_paths[i + 1][:len(q)] == q and len(sorted_paths[i + 1]) > len(q))
    ranks = np.array([r for p in paths for r in p], dtype=np.int64)
    is_markup = np.asarray(markup, dtype=bool)
    inside = tn >= 0
    stats = dict(n_text_tokens=len(body), full_len=full, cropped=cropped,
                 nodes_interior=n_interior,
                 nodes=len(paths), nodes_truncated_path=n_trunc,
                 max_depth=max((len(p) for p in paths), default=0),
                 ranks=len(ranks), ranks_fast_wrap=int((ranks >= FAST_WRAP_RANK).sum()),
                 ranks_slow_wrap=int((ranks >= SLOW_WRAP_RANK).sum()),
                 tokens_kept_in_tree=int(inside.sum()),
                 tokens_markup=int(is_markup[tn[inside]].sum()) if len(is_markup) else 0)
    return ids, tn, node_path, tw, stats


#: Attribute values take all five XML entities. TEXT content takes only the three
#: that can be mistaken for markup: escaping quotes and apostrophes in prose would
#: turn every ordinary apostrophe into ``&#39;`` and inflate the stream for nothing.
_ATTR_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;"), ("'", "&#39;"))
_TEXT_ESCAPES = _ATTR_ESCAPES[:3]
_UNWRITABLE = set("<>/=\"'")


def _escape(value: str, escapes) -> str:
    for raw, ent in escapes:
        value = value.replace(raw, ent)
    return value


def _writable(name: str) -> bool:
    """A tag or attribute name that can be written without ambiguity."""
    return bool(name) and not any(c.isspace() or c in _UNWRITABLE for c in name)


def _child_items(el, path, start_rank):
    """Work items for ``el``'s RETAINED children, in document order.

    Ranks continue from ``start_rank``, which is the number of attribute nodes the
    element already used up -- attributes are ranked BEFORE child elements and text.
    """
    items, rank = [], start_rank
    for c in el.iter(include_text=True):
        tag = c.tag
        if tag == "-text":
            t = " ".join(c.text(deep=False).split())
            if t:
                items.append(("text", t, path + [rank]))
                rank += 1
            continue
        if tag is None or tag.startswith("-") or tag in SKIP_TAGS:
            continue                          # comments, doctype, dropped subtrees: no rank
        items.append(("elem", c, path + [rank]))
        rank += 1
    return items


def extract_html_page(html: str):
    """The page as MARKUP: ``(pieces, paths, kinds)``, or None if unrenderable.

    ``pieces`` is ``[(string, node_id)]`` in emission order and concatenates to the
    whole serialisation; ``paths[node_id]`` is the rank path; ``kinds[node_id]`` is
    ``"element"`` / ``"attr"`` / ``"text"``.

    THE FORM IS XML-STYLE, NOT HTML5 VOID SYNTAX. Scraped markup contains
    void-named elements carrying children -- ``<source>`` and ``<wbr>`` with a child
    appear on ~1.4% of pages -- so ``<br>`` without a closing tag cannot be read back
    without unbounded lookahead. Every childless element is written ``<tag/>`` and
    everything else ``<tag>...</tag>``, so the form is decided at the open tag.
    (mathesis scripts/dom_render_html.py reached the same conclusion for the same
    corpus; this is an independent implementation, not an import.)

    A NODE IS an element, an attribute, or a kept text node. Attributes are children
    of their element and take the first ranks, so an element's first content child
    has rank ``n_attributes``.
    """
    from selectolax.parser import HTMLParser
    try:
        tree = HTMLParser(html)
    except Exception:
        return None
    body = tree.body
    if body is None:
        return None
    pieces, paths, kinds = [], [], []

    def new_node(path, kind):
        paths.append(path)
        kinds.append(kind)
        return len(paths) - 1

    # Body is not a level: its children start the path. Its own attributes are not
    # nodes, exactly as in the text render.
    stack = list(reversed(_child_items(body, [], 0)))
    while stack:
        kind, obj, arg = stack.pop()
        if kind == "piece":                   # a deferred closing tag
            pieces.append((obj, arg))
            continue
        if kind == "text":
            nid = new_node(arg, "text")
            pieces.append((_escape(obj, _TEXT_ESCAPES), nid))
            continue
        el, path = obj, arg
        if not _writable(el.tag):
            return None                       # a tag we cannot write back: drop the page
        nid = new_node(path, "element")
        pieces.append(("<" + el.tag, nid))
        rank = 0
        for name, value in (el.attributes or {}).items():
            if not _writable(name):
                continue                      # unwritable name: no node, no rank, no token
            anid = new_node(path + [rank], "attr")
            rank += 1
            pieces.append((f' {name}="{_escape(value or "", _ATTR_ESCAPES)}"', anid))
        items = _child_items(el, path, rank)
        if not items:
            pieces.append(("/>", nid))
            continue
        pieces.append((">", nid))
        stack.append(("piece", f"</{el.tag}>", nid))
        stack.extend(reversed(items))
    return pieces, paths, kinds


def tokenise_page(nodes, enc, block: int = BLOCK, levels: int = LEVELS, window: bool = False):
    """TEXT render: every node contributes one piece, pieces joined by newlines.

    -> (ids uint16 [EOT..EOT], tok_node int32, node_path int16 (n_nodes, levels), tok_within int16, stats).
    """
    return _tokenise([t for t, _ in nodes], list(range(len(nodes))), [p for _, p in nodes],
                     [False] * len(nodes), enc, block, levels, "\n", window)


def tokenise_html_page(pieces, paths, kinds, enc, block: int = BLOCK, levels: int = LEVELS,
                       window: bool = False):
    """HTML render: a node may contribute several pieces, and nothing joins them."""
    return _tokenise([s for s, _ in pieces], [n for _, n in pieces], paths,
                     [k != "text" for k in kinds], enc, block, levels, "", window)


def trim_nodes(tn, npth):
    """Keep only the node rows a record's tokens can reach: the prefix up to the largest id.

    Node ids are record-local and ``npth`` is indexed by them, so a PREFIX preserves every id
    unchanged. See the long note at the call site for why that matters.
    """
    keep = int(tn.max()) + 1 if len(tn) and tn.max() >= 0 else 0
    return npth[:keep] if keep < len(npth) else npth


def split_windows(ids, tn, tw, block: int):
    """One page -> a list of ``(ids, tok_node, tok_within)`` records, none longer than ``block``.

    Windows are of NEAR-EQUAL length rather than ``block`` followed by a remainder: a page of
    1,100 tokens becomes two records of 550, not one of 1,024 and one of 76. Fixed-size chunking
    leaves a runt tail on most pages, and a runt still occupies a padded batch slot, so it costs
    the GPU a full window's work to carry a few dozen tokens. Equal windows keep every token and
    keep the bucket batcher's padding waste down.
    """
    n = len(ids)
    if n <= block:
        return [(ids, tn, tw)]
    w = -(-n // block)                              # ceil: the fewest windows that fit
    size = -(-n // w)
    return [(ids[i:i + size], tn[i:i + size], tw[i:i + size]) for i in range(0, n, size)]


def too_deep(nodes, levels: int) -> bool:
    """True when any node's path is longer than ``levels`` (the ``--drop-deep`` rule)."""
    return any(len(p) > levels for _, p in nodes)


def _process_shard(args):
    """Worker: one shard -> (shard_idx, name, n_read, n_nobody, n_short, n_deep, n_alias, n_long, [rec])."""
    import tiktoken
    path, shard_idx, limit, levels, drop_deep, render, drop_alias, block, window, drop_long = args
    enc = tiktoken.get_encoding("gpt2")
    out, n_read, n_nobody, n_short, n_deep, n_alias, n_long = [], 0, 0, 0, 0, 0, 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            if limit and n_read >= limit:
                break
            n_read += 1
            try:
                html = json.loads(line)["html"]
            except Exception:
                n_nobody += 1
                continue
            if render == "html":
                got = extract_html_page(html)
                if not got or not got[0]:
                    n_nobody += 1
                    continue
                pieces, paths, kinds = got
                if drop_deep and any(len(p) > levels for p in paths):
                    n_deep += 1
                    continue
                ids, tn, npth, tw, st = tokenise_html_page(pieces, paths, kinds, enc, block=block,
                                                           levels=levels, window=window)
            else:
                nodes = extract_page(html)
                if not nodes:
                    n_nobody += 1
                    continue
                if drop_deep and too_deep(nodes, levels):
                    n_deep += 1
                    continue
                ids, tn, npth, tw, st = tokenise_page(nodes, enc, block=block, levels=levels,
                                                      window=window)
            if st["n_text_tokens"] < MIN_TEXT_TOKENS:
                n_short += 1
                continue
            # DROP RATHER THAN CROP (2026-09-30). With --drop-long a page that does not fit the
            # context is left out entirely, so every record in the pack is a WHOLE document and
            # nothing in it -- text or coordinate -- is truncated. This is the alternative to
            # --window for a cell that must not contain fragments; the two are mutually
            # exclusive and main refuses the pair. The cost is pages, and a bias toward shorter
            # ones, which the manifest records.
            if drop_long and st["full_len"] > block:
                n_long += 1
                continue
            # ALIASING (2026-09-25). Drop a page whose coordinates the rotation cannot
            # tell apart, so the corpus matches the encoding's unambiguous capacity and
            # the arm is measured on the mechanism rather than on collisions. Measured
            # cost: ~1.5% of text pages, ~2.2% of html pages. (Dropping at the FAST
            # plane's wrap of 20 instead would discard 55% / 99% -- the fast plane
            # turning over is not a collision, since the slow plane still separates.)
            if drop_alias:
                mx = int(npth.max()) if npth.size else -1
                mw = int(tw.max()) if tw.size else -1
                if mx >= MAX_RANK or mw >= MAX_WITHIN:
                    n_alias += 1
                    continue
            out.append((f"{os.path.basename(path)}:{line_no}", ids, tn, npth, tw, st))
    return shard_idx, os.path.basename(path), n_read, n_nobody, n_short, n_deep, n_alias, n_long, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=600_000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--shards", type=int, default=0, help="0 = as many as --pages needs")
    ap.add_argument("--lines-per-shard", type=int, default=0,
                    help="read only this many lines of each shard (smoke builds); 0 = all")
    ap.add_argument("--shard-timeout", type=float, default=3600,
                    help="seconds to wait for the next shard result before declaring the rest lost")
    ap.add_argument("--block", type=int, default=BLOCK, help="record length cap")
    ap.add_argument("--window", action="store_true",
                    help="split a long page into several records instead of cropping it. "
                         "Keeps 100%% of text tokens; the 4,096 crop discards 11.4%%, a 1,024 "
                         "crop would discard 42%%. Costs ~1.9x records and ~1.9x node rows at "
                         "block 1024. Off by default so existing packs rebuild byte-identically.")
    ap.add_argument("--drop-long", action="store_true",
                    help="drop a page longer than --block instead of cropping it, so EVERY "
                         "record is a whole page and no coordinate anywhere in the pack is "
                         "truncated. The alternative to --window when the cell must not contain "
                         "document fragments. Costs pages: at 2,048 it keeps 77% of pages and "
                         "~46% of the corpus's text, and biases the pack toward shorter pages.")
    ap.add_argument("--allow-window-interior", action="store_true",
                    help="window even when the stream contains a node's own descendants. Only "
                         "correct if NO arm in the cell reads depth or reconstructs the tree "
                         "from it -- read the refusal message before reaching for this.")
    ap.add_argument("--levels", type=int, default=LEVELS, help="tree levels below <body> in the path")
    ap.add_argument("--drop-alias", action="store_true",
                    help=f"drop a page with any sibling rank >= {MAX_RANK} or any within-node "
                         f"index >= {MAX_WITHIN} -- the points where BOTH planes of a level wrap "
                         f"and the coordinate collides with one a period earlier")
    ap.add_argument("--drop-deep", action="store_true",
                    help="drop pages deeper than --levels instead of truncating their paths")
    ap.add_argument("--render", choices=("text", "html"), default="text",
                    help="text = the page's visible text only (structure lives only in the sidecar); "
                         "html = the markup INCLUDING the text, so structure is also spelled out in tokens")
    a = ap.parse_args()
    shards = sorted(glob.glob(SNAPSHOT_GLOB))
    if not shards:
        sys.exit(f"no shards at {SNAPSHOT_GLOB}")
    est_per_shard = 16384 * (0.8 if a.drop_deep else 0.9)
    # --shards caps how many shards may be scanned; the page target decides how many
    # are actually submitted, so a big --shards no longer means a 1,185-task queue.
    n_shards = min(a.shards or len(shards), int(a.pages / est_per_shard) + 4, len(shards))
    t0 = time.time()
    shard_log = []
    # STREAMING (2026-09-24). The previous build collected every page and called
    # np.concatenate at the end: fine at 600k pages / 13 GB, impossible at the full
    # corpus (~17.5M pages, ~373 GB) on a 220 GB machine. Now each shard's pages are
    # written straight through, and only the held-out candidates are kept in memory.
    #
    # A page is a held-out CANDIDATE when a stable hash of its id falls under a
    # threshold sized to yield ~3x the val+test target. At the end the candidates are
    # sorted by page id, the first N_VAL become val and the next N_TEST become test,
    # and every leftover is appended to train -- so no page is discarded and the split
    # does not depend on how many pages the build happened to read, nor on the order
    # the workers returned them.
    os.makedirs(a.out, exist_ok=True)
    side = a.out + "_dompath"
    os.makedirs(side, exist_ok=True)
    w_tok = StreamNpy(os.path.join(a.out, "tokens.npy"), np.uint16)
    w_tn = StreamNpy(os.path.join(side, "tok_node.npy"), np.int32)
    w_np = StreamNpy(os.path.join(side, "node_path.npy"), np.int16, cols=a.levels)
    w_tw = StreamNpy(os.path.join(side, "tok_within.npy"), np.int16)
    agg = dict(cropped=0, full_len=0, n_text_tokens=0, nodes=0, nodes_truncated_path=0,
               ranks=0, ranks_fast_wrap=0, ranks_slow_wrap=0, tokens_kept_in_tree=0,
               tokens_markup=0, tokens_stored=0, windows=0, nodes_interior=0)
    page_ids, tok_off, node_off = [], [0], [0]
    n_pages = [0]                       # emit() calls; under --window one page is several records
    holdout = []        # candidate pages, kept in memory (~3x (N_VAL+N_TEST) records)
    n_written = [0]

    def emit(rec):
        """Append one page to the pack's streams and fold its stats in.

        Under ``--window`` a page becomes SEVERAL records. Windowing lives here, after the
        split has been decided by hashing the page id, so every window of a page lands in the
        same split -- windowing before that point would put the first half of a page in train
        and the second in test, which is a leak of the most damaging kind.
        """
        pid, ids, tn, npth, tw, st = rec
        # WHY WINDOWING IS GATED ON THE TREE'S SHAPE (2026-09-30, at the author's warning).
        # Windowing is only safe for a stream of LEAVES. The depth encoding is lossless because
        # a node's parent is the nearest PRECEDING node one level shallower -- so a window that
        # starts mid-document does not merely lose the ancestors, it makes that rule name the
        # WRONG parent: some shallower node inside the window. The arm would train and encode a
        # corrupted tree.
        #
        # The web TEXT render is safe by construction: it keeps text nodes only, and their
        # ancestors are elements that emit no tokens, so no parent-child pair is in the stream.
        # Measured on the shipped pack: 0 of 53,487 nodes has a descendant in the stream.
        # The HTML render and any AST corpus are NOT safe -- they are interior nodes throughout.
        if a.window and st is not None and st["nodes_interior"] and not a.allow_window_interior:
            sys.exit(f"[web] REFUSED --window: page {pid} has {st['nodes_interior']} nodes whose "
                     f"DESCENDANTS are also in the token stream. Windowing breaks the depth "
                     f"encoding for such a tree -- a window starting mid-document makes "
                     f"'parent = nearest preceding node one level shallower' resolve to the "
                     f"wrong parent. Crop instead, or pass --allow-window-interior if no arm "
                     f"in this cell reads depth.")
        wins = split_windows(ids, tn, tw, a.block) if a.window else [(ids, tn, tw)]
        for k, (w_ids, w_tnk, w_twk) in enumerate(wins):
            _emit_one(pid if k == 0 else f"{pid}#w{k}", w_ids, w_tnk, npth, w_twk,
                      st if k == 0 else None)
        agg["windows"] += len(wins)
        n_pages[0] += 1

    def _emit_one(pid, ids, tn, npth, tw, st):
        """One RECORD. ``st`` is the page's stats, folded in once per page, not per window."""
        # TRIM UNREACHABLE NODES (2026-09-25). `npth` covers the WHOLE page, but a
        # record's tokens are cropped at block_size -- and on the html render 82% of
        # records are cropped, so most node rows are never referenced by any token.
        # Measured on a 6,000-page html smoke: only 31.4% of stored node rows are
        # reachable, i.e. node_path was 3.2x larger than it needs to be (62.5 ->
        # 19.7 KB/page). Node ids are record-local and the array is indexed by them,
        # so keeping the prefix up to the largest referenced id preserves every id.
        #
        # THIS IS ALSO WHAT MAKES ``--window`` FREE OF ID REMAPPING. A window past the first
        # references a RANGE of node ids rather than a prefix, but keeping the prefix 0..max
        # leaves every id pointing at the same row it always did. Slicing a suffix instead
        # would save disk and require renumbering every `tok_node` -- the exact class of
        # offset-remapping code that has produced the worst bugs in this project. The waste
        # is measured: 1.88x node rows at block 1024, against keeping 42% more text.
        #
        # This matters beyond disk: cell 4's graph arms went DISK-BOUND on a sidecar
        # too large for the page cache and lost 1.8x against the sequence arm, which
        # reads no sidecar at all.
        npth = trim_nodes(tn, npth)
        page_ids.append(pid)
        w_tok.append(ids); tok_off.append(tok_off[-1] + len(ids))
        w_tn.append(tn); w_np.append(npth); w_tw.append(tw)
        node_off.append(node_off[-1] + len(npth))
        # tokens_stored counts every window; the PAGE stats are folded in once (st is None on
        # the second and later windows of a page) so per-page rates stay per-page.
        agg["tokens_stored"] += len(ids)
        if st is not None:
            for k in agg:
                if k in ("tokens_stored", "windows"): continue
                elif k == "cropped": agg[k] += int(st["cropped"])
                else: agg[k] += st[k]
        n_written[0] += 1

    def is_candidate(pid, target, expected):
        h = int(hashlib.blake2b(pid.encode(), digest_size=8, key=str(SPLIT_SEED).encode()).hexdigest(), 16)
        return (h % expected) < target
    # imap_unordered + a per-result timeout: a worker that dies (a parser
    # segfault) would otherwise hang Pool.imap forever with no error. A shard
    # whose result never arrives is logged as lost and the build goes on.
    # ~3x the val+test target, so truncating to exactly N_VAL/N_TEST is always possible.
    hold_expected = max(a.pages, N_VAL + N_TEST + 1)
    hold_target = min(3 * (N_VAL + N_TEST), hold_expected)
    if a.window and a.drop_long:
        sys.exit("[web] REFUSED: --window and --drop-long are alternatives, not a pair. "
                 "--window keeps every token by splitting long pages into records; "
                 "--drop-long keeps every record whole by leaving long pages out.")
    tasks = [(s, i, a.lines_per_shard, a.levels, a.drop_deep, a.render, a.drop_alias,
              a.block, a.window, a.drop_long)
             for i, s in enumerate(shards[:n_shards])]
    seen = set()
    # BACKPRESSURE (2026-09-24). ``imap_unordered`` over all 1,185 tasks hands the pool
    # every task at once and buffers each completed result in the parent until the loop
    # consumes it. With 58 workers each returning ~15k pages of arrays (~0.4 GB/shard),
    # the parent grew ~40 -> ~101 GB in two hours and would have exhausted 220 GB long
    # before the build finished: the workers produce far faster than one writer consumes.
    #
    # Submitting in WAVES of ``workers`` tasks bounds the unconsumed results to one wave.
    # DEADLOCK, 2026-09-25. Breaking out of `imap_unordered` early without draining it
    # leaves results queued; `Pool.__exit__` calls terminate(), and terminate() then hung
    # for good with the parent in futex_wait and all 48 workers defunct -- taking a
    # finished 76 GB pack with it, because the offsets and metadata live in the parent.
    #
    # Two defences. (1) Below, the task list is trimmed to the number of shards the
    # requested page count actually needs, so the common path never breaks early.
    # (2) If we do break, terminate the pool EXPLICITLY inside the loop and hard-exit
    # the context, rather than leaving it to __exit__.
    if not a.shards:
        pass                                   # n_shards already derived from --pages
    with Pool(a.workers) as pool:
        pending = list(range(len(tasks)))
        while len(seen) < len(tasks):
            wave = [tasks[i] for i in pending[:a.workers]]
            pending = pending[a.workers:]
            if not wave:
                break
            it = pool.imap_unordered(_process_shard, wave)
            for _ in range(len(wave)):
                try:
                    (shard_idx, name, n_read, n_nobody, n_short, n_deep, n_alias, n_long,
                     out) = it.next(timeout=a.shard_timeout)
                except StopIteration:
                    break
                except Exception as exc:  # multiprocessing.TimeoutError or a worker exception
                    lost = sorted(set(range(len(tasks))) - seen)
                    print(f"[web] LOST shards {lost}: {type(exc).__name__}: {exc}", flush=True)
                    shard_log.append(dict(lost_shards=[os.path.basename(shards[i]) for i in lost],
                                          error=f"{type(exc).__name__}: {exc}"))
                    break
                seen.add(shard_idx)
                shard_log.append(dict(shard=name, read=n_read, no_body=n_nobody, short=n_short,
                                      deep=n_deep, alias=n_alias, long=n_long, kept=len(out)))
                out.sort(key=lambda r: r[0])       # within a shard, page-id order is deterministic
                for rec in out:
                    if is_candidate(rec[0], hold_target, hold_expected):
                        holdout.append(rec)
                    else:
                        emit(rec)
                    if n_written[0] + len(holdout) >= a.pages:
                        break
                print(f"[web] shard {shard_idx} {name}: read {n_read} kept {len(out)} "
                      f"(total {n_written[0] + len(holdout)}) {time.time()-t0:.0f}s", flush=True)
                out = None                      # release the wave's pages promptly
                if n_written[0] + len(holdout) >= a.pages:
                    break
            if n_written[0] + len(holdout) >= a.pages:
                print(f"[web] page target reached; terminating the pool", flush=True)
                pool.terminate()
                pool.join()
                break
    # Finalise: train is what streamed through, plus any surplus candidates; then
    # exactly N_VAL val records and N_TEST test records, so the three splits occupy
    # contiguous record ranges in that order (what PackedWindows assumes).
    holdout.sort(key=lambda r: r[0])
    if len(holdout) < N_VAL + N_TEST:
        sys.exit(f"[web] REFUSED: only {len(holdout)} held-out candidates, need {N_VAL + N_TEST}")
    val_recs = holdout[:N_VAL]
    test_recs = holdout[N_VAL:N_VAL + N_TEST]
    surplus = holdout[N_VAL + N_TEST:]
    for rec in surplus:
        emit(rec)
    n_train = n_written[0]
    for rec in val_recs:
        emit(rec)
    for rec in test_recs:
        emit(rec)
    if n_train < 1:
        sys.exit("[web] REFUSED: no training records")
    for w in (w_tok, w_tn, w_np, w_tw):
        w.close()
    np.save(os.path.join(a.out, "tok_off.npy"), np.array(tok_off, dtype=np.int64))
    np.save(os.path.join(side, "node_off.npy"), np.array(node_off, dtype=np.int64))
    train, val, test = range(n_train), val_recs, test_recs
    order = page_ids
    n = len(order)
    n_read_total = sum(x.get("read", 0) for x in shard_log)
    n_deep_total = sum(x.get("deep", 0) for x in shard_log)
    n_alias_total = sum(x.get("alias", 0) for x in shard_log)
    n_long_total = sum(x.get("long", 0) for x in shard_log)
    # Under --drop-long nothing may have been cropped; if anything was, the mode did not work
    # and the pack silently contains truncated pages after all.
    if a.drop_long and agg["cropped"]:
        sys.exit(f"[web] BUG: --drop-long but {agg['cropped']} records were cropped")
    meta = dict(text_pack_version=1, block_size=a.block, windowed=a.window, vocab_size=VOCAB_PADDED, eot=EOT,
                tokenizer="gpt2", tokenizer_is_pretrained=True,
                renders_with=("web_html" if a.render == "html" else "web_text"), render=a.render,
                n_records=n, n_train=len(train), n_val=len(val), n_test=len(test),
                split_seed=SPLIT_SEED, split_rule="by page; records stored train, then val, then test",
                source="bs-modeling-metadata/c4-en-html-with-metadata", shards=shard_log,
                min_text_tokens=MIN_TEXT_TOKENS, levels=a.levels, drop_deep=a.drop_deep, drop_alias=a.drop_alias,
                max_rank=MAX_RANK if a.drop_alias else None,
                max_within=MAX_WITHIN if a.drop_alias else None,
                pages_dropped_deep=n_deep_total, pages_dropped_alias=n_alias_total, pages_read=n_read_total,
                drop_long=a.drop_long, pages_dropped_long=n_long_total,
                total_tokens=int(agg["tokens_stored"]),
                tokens_per_record_mean=agg["tokens_stored"] / n,
                records_cropped_frac=agg["cropped"] / n,
                text_tokens_lost_to_crop_frac=1 - agg["tokens_stored"] / (agg["full_len"] or 1),
                n_pages=n_pages[0], windows_per_page=n / (n_pages[0] or 1),
                nodes_interior=agg["nodes_interior"],
                markup_token_frac=agg["tokens_markup"] / (agg["tokens_kept_in_tree"] or 1),
                build_wall_s=round(time.time() - t0, 1))
    with open(os.path.join(a.out, "meta.json"), "w") as f: json.dump(meta, f, indent=1)
    with open(os.path.join(a.out, "page_ids.json"), "w") as f: json.dump(page_ids, f)
    html_rules = (
        "every childless element is written `<tag/>` and every other `<tag>...</tag>`, so the form is "
        "decided at the open tag (scraped markup has void-named elements WITH children, which makes "
        "HTML5 void syntax unreadable without lookahead); attribute values are double-quoted with the "
        "five XML entities escaped; text content escapes only & < > (escaping quotes would turn every "
        "apostrophe in prose into `&#39;`); attributes keep source order"
    )
    manifest = dict(dompath_version=2, text_pack=a.out, levels=a.levels, absent=ABSENT,
                    render=a.render,
                    markup_token_frac=agg["tokens_markup"] / (agg["tokens_kept_in_tree"] or 1),
                    html_render_rule=(html_rules if a.render == "html" else None),
                    drop_deep=a.drop_deep, drop_alias=a.drop_alias,
                max_rank=MAX_RANK if a.drop_alias else None,
                max_within=MAX_WITHIN if a.drop_alias else None, pages_dropped_deep=n_deep_total, pages_dropped_alias=n_alias_total, pages_read=n_read_total,
                    pages_dropped_deep_frac=n_deep_total / (n_read_total or 1),
                    pages_dropped_alias_frac=n_alias_total / (n_read_total or 1),
                    drop_rule=("a page whose deepest node exceeds `levels` is DROPPED from the pack, so no path is truncated"
                               if a.drop_deep else "deep paths are truncated at the LEAF end (cell 1 behaviour)"),
                    within_rule=("tok_within = the token's ORDINAL among its own node's tokens, 0 at the node's "
                                 "first token; -1 for the framing EOTs (tok_node == -1). On the html render an "
                                 "element's tokens are split apart by its children, so this is a count, not a distance."
                                 if a.render == "html" else
                                 "tok_within = the token's index within its text node, 0 at the node's first token, "
                                 "the node's trailing newline is its last token; -1 for the framing EOTs (tok_node == -1)"),
                    parser="selectolax.parser (Modest engine); lexbor segfaults on shard 1 line 659",
                    skip_tags=sorted(SKIP_TAGS),
                    node_rule=("elements, ATTRIBUTES and kept text nodes, none under a skipped tag; an element's "
                               "attributes are its first children, so its first content child has rank n_attributes; "
                               "path starts at the children of <body>"
                               if a.render == "html" else
                               "element nodes not under a skipped tag, and text nodes with any non-whitespace char; "
                               "rank = index among the parent's RETAINED children; path starts at the children of <body>; "
                               "paths longer than `levels` are truncated at the LEAF end"),
                    token_rule=("a token belongs to the node whose byte span holds its first non-whitespace byte: "
                                "`<tag`, `>`, `/>` and `</tag>` to the ELEMENT, ` name=\"value\"` to the ATTRIBUTE, "
                                "character data to the TEXT node; framing EOTs are ABSENT"
                                if a.render == "html" else
                                "a token belongs to the text node whose byte span holds its first non-whitespace byte; "
                                "the newline after node i belongs to node i; framing EOTs are ABSENT"),
                    lookahead="0 by construction: emission is document order and a rank counts earlier siblings only",
                    nodes_total=int(agg["nodes"]),
                    nodes_truncated_path_frac=agg["nodes_truncated_path"] / (agg["nodes"] or 1),
                    ranks_fast_wrap_frac=agg["ranks_fast_wrap"] / (agg["ranks"] or 1),
                    ranks_slow_wrap_frac=agg["ranks_slow_wrap"] / (agg["ranks"] or 1),
                    tokens_in_tree_frac=agg["tokens_kept_in_tree"] / (agg["tokens_stored"] or 1))
    with open(os.path.join(side, "manifest.json"), "w") as f: json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "shards"}, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
