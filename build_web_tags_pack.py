#!/usr/bin/env python
"""The Web-S pages again, with the TAGS left in: a baseline that can read the structure as tokens.

    .venv/bin/python build_web_tags_pack.py --text-pack $TOPOS_PACKS/web_whole_b2048 \
        --out $TOPOS_PACKS/web_tags_b4096 --block 4096 --workers 64

WHY THIS PACK EXISTS. A referee's objection to the web result: the text pack keeps only the
visible text, so the structure reaches a model ONLY through the rotated tree coordinate. The
sequence-only baseline is therefore "no structure", not "structure written as text", and the gain
could be "don't delete the markup" rather than anything about rotary position. This pack is the
control: the SAME pages, the SAME split, the SAME visible text, with the element tags written into
the token stream, so a sequence-only model can read every open, close and nesting level.

WHAT THE RENDER IS -- AND WHY IT IS NOT ``build_web_pack.py --render html``.
  * TAGS, NO ATTRIBUTES. ``<div>`` and ``</div>``, never ``class=".."`` or ``href=".."``. The
    question is whether the TREE, written as tokens, substitutes for the tree as a rotation;
    attributes are content the text render never had, and they triple the stream. A childless
    element is ``<tag/>``, so the form is decided at the open tag (the html render's rule).
  * With no attribute nodes, ranks are exactly the text render's: a node's rank is its index
    among its parent's retained element and text children, starting at 0. A text node's path
    here is IDENTICAL to its path in the text pack, which is what lets the depth arm on this
    pack be compared with the depth arm on the text pack.
  * TEXT IS WRITTEN RAW, not entity-escaped, so the visible text is byte-identical to the text
    pack's. A literal ``<`` in prose is then ambiguous to an HTML parser -- but nothing parses
    this stream; a model reads it. The manifest counts the pages where it happens.
  * EACH PIECE IS BPE-ENCODED ON ITS OWN, then the ids are concatenated. Encoding the joined
    string would let GPT-2's pre-tokeniser glue punctuation across a boundary (``.</`` is one
    pre-token), and a token would then hold some text and some markup. Piecewise, every token
    belongs to exactly one piece. And because the text pack's newline separator always splits a
    pre-token, a text node's tokens here are EXACTLY that node's tokens in the text pack minus
    its trailing newline -- checked page by page at build time (``--verify``), not assumed.

THE PAGE SET AND SPLIT ARE COPIED, NOT RE-DERIVED. Only pages in ``--text-pack`` are read (its
``page_ids.json`` names shard and line), and each keeps its split: the text pack's val pages are
this pack's val pages, in the same order, and likewise test. A page whose tagged render is longer
than ``--block`` is DROPPED -- never cropped (whole pages only) -- and the manifest records how
many, per split. Evaluation on visible text aligns records by page id, so a dropped val page is
simply absent from both sides of that comparison.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from build_web_pack import (ABSENT, EOT, MAX_RANK, MAX_WITHIN, SKIP_TAGS, SNAPSHOT_GLOB,
                            _writable, extract_page, trim_nodes)
from npy_stream import StreamNpy


def extract_tags_page(html: str):
    """``(pieces, paths, kinds)``: tags without attributes, raw text. None if unrenderable.

    ``pieces`` is ``[(string, node_id)]`` in document order; ``kinds[node_id]`` is
    ``"element"`` or ``"text"``.
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

    def child_items(el, path):
        items, rank = [], 0
        for c in el.iter(include_text=True):
            tag = c.tag
            if tag == "-text":
                t = " ".join(c.text(deep=False).split())
                if t:
                    items.append(("text", t, path + [rank]))
                    rank += 1
                continue
            if tag is None or tag.startswith("-") or tag in SKIP_TAGS:
                continue
            items.append(("elem", c, path + [rank]))
            rank += 1
        return items

    stack = list(reversed(child_items(body, [])))
    while stack:
        kind, obj, arg = stack.pop()
        if kind == "piece":
            pieces.append((obj, arg))
            continue
        nid = len(paths)
        paths.append(arg)
        if kind == "text":
            kinds.append("text")
            pieces.append((obj, nid))
            continue
        kinds.append("element")
        el = obj
        if not _writable(el.tag):
            return None
        items = child_items(el, arg)
        if not items:
            pieces.append((f"<{el.tag}/>", nid))
            continue
        pieces.append((f"<{el.tag}>", nid))
        stack.append(("piece", f"</{el.tag}>", nid))
        stack.extend(reversed(items))
    return pieces, paths, kinds


def tokenise_tags(pieces, paths, kinds, enc, levels: int):
    """Piecewise BPE. -> ids, tok_node, node_path, tok_within, text_mask, stats."""
    ids, tn = [EOT], [ABSENT]
    for s, nid in pieces:
        t = enc.encode_ordinary(s)
        ids.extend(t)
        tn.extend([nid] * len(t))
    ids.append(EOT)
    tn.append(ABSENT)
    ids = np.asarray(ids, dtype=np.uint16)
    tn = np.asarray(tn, dtype=np.int32)
    # within-node ORDINAL, as in build_web_pack (an element's tokens are split by its children)
    within = np.full(len(tn), ABSENT, dtype=np.int16)
    inside = tn >= 0
    if inside.any():
        sub = tn[inside]
        order = np.argsort(sub, kind="stable")
        grouped = sub[order]
        start = np.searchsorted(grouped, grouped, side="left")
        w = np.empty(len(sub), dtype=np.int64)
        w[order] = np.arange(len(sub)) - start
        within[inside] = np.minimum(w, np.iinfo(np.int16).max)
    node_path = np.full((len(paths), levels), ABSENT, dtype=np.int16)
    for i, p in enumerate(paths):
        q = np.minimum(np.asarray(p[:levels], dtype=np.int64), np.iinfo(np.int16).max)
        node_path[i, :len(q)] = q
    is_text = np.asarray([k == "text" for k in kinds], dtype=bool)
    text_mask = np.zeros(len(tn), dtype=bool)
    text_mask[inside] = is_text[tn[inside]]
    stats = dict(n_tokens=len(ids), n_text_tokens=int(text_mask.sum()),
                 n_markup_tokens=int(inside.sum() - text_mask.sum()),
                 max_depth=max((len(p) for p in paths), default=0),
                 lt_in_text=int(any(("<" in s or ">" in s) for s, n in pieces if kinds[n] == "text")))
    return ids, tn, node_path, within, text_mask, stats


def text_tokens_of_text_pack(ids, tn, newline_id: int):
    """The text pack's visible-text tokens: framing EOTs and the node-separating newlines removed."""
    keep = (tn >= 0) & (ids != newline_id)
    return ids[keep]


def _work(args):
    """One shard: the requested lines -> list of (pid, record arrays, stats) or (pid, None, why)."""
    import tiktoken
    path, lines, levels, block, verify = args
    enc = tiktoken.get_encoding("gpt2")
    nl = enc.encode_ordinary("\n")[0]
    want = set(lines)
    out = []
    name = os.path.basename(path)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        last = max(want) if want else -1
        for line_no, line in enumerate(fh):
            if line_no > last:
                break
            if line_no not in want:
                continue
            pid = f"{name}:{line_no}"
            html = json.loads(line)["html"]
            got = extract_tags_page(html)
            if not got or not got[0]:
                out.append((pid, None, "unrenderable"))
                continue
            pieces, paths, kinds = got
            if any(len(p) > levels for p in paths):
                out.append((pid, None, "deep"))
                continue
            ids, tn, npth, tw, tmask, st = tokenise_tags(pieces, paths, kinds, enc, levels)
            if len(ids) > block:
                out.append((pid, None, "long"))
                continue
            if npth.size and int(npth.max()) >= MAX_RANK or int(tw.max()) >= MAX_WITHIN:
                out.append((pid, None, "alias"))
                continue
            if verify:
                # the text pack's tokens for this page, re-derived the way build_web_pack does
                from build_web_pack import tokenise_page
                nodes = extract_page(html)
                tids, ttn, _, _, _ = tokenise_page(nodes, enc, block=1 << 30, levels=levels)
                a = text_tokens_of_text_pack(tids, ttn, nl)
                b = ids[tmask]
                if len(a) != len(b) or not np.array_equal(a, b):
                    out.append((pid, None, "MISALIGNED"))
                    continue
            out.append((pid, (ids, tn, trim_nodes(tn, npth), tw, tmask), st))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--text-pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--levels", type=int, default=22)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit-pages", type=int, default=0, help="smoke: only the first N pages per split")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the per-page alignment check against the text render")
    a = ap.parse_args()

    import glob
    shards = {os.path.basename(p): p for p in glob.glob(SNAPSHOT_GLOB)}
    tmeta = json.load(open(os.path.join(a.text_pack, "meta.json")))
    tpids = json.load(open(os.path.join(a.text_pack, "page_ids.json")))
    n_tr, n_va = tmeta["n_train"], tmeta["n_val"]
    split_of = {}
    order = {"val": [], "test": []}
    for i, pid in enumerate(tpids):
        s = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        split_of[pid] = s
        if s != "train":
            order[s].append(pid)
    if a.limit_pages:
        keep = set(order["val"][:a.limit_pages]) | set(order["test"][:a.limit_pages]) |\
            {p for p in tpids[:n_tr][:a.limit_pages * 4]}
        split_of = {p: s for p, s in split_of.items() if p in keep}
        order = {k: [p for p in v if p in keep] for k, v in order.items()}
    by_shard: dict = {}
    for pid in split_of:
        sh, ln = pid.rsplit(":", 1)
        by_shard.setdefault(sh, []).append(int(ln))
    missing = [s for s in by_shard if s not in shards]
    if missing:
        sys.exit(f"source shards missing from {SNAPSHOT_GLOB}: {missing[:3]}")
    tasks = [(shards[s], sorted(ls), a.levels, a.block, a.verify) for s, ls in sorted(by_shard.items())]

    os.makedirs(a.out, exist_ok=True)
    side = a.out + "_dompath"
    os.makedirs(side, exist_ok=True)
    w_tok = StreamNpy(os.path.join(a.out, "tokens.npy"), np.uint16)
    w_tn = StreamNpy(os.path.join(side, "tok_node.npy"), np.int32)
    w_np = StreamNpy(os.path.join(side, "node_path.npy"), np.int16, cols=a.levels)
    w_tw = StreamNpy(os.path.join(side, "tok_within.npy"), np.int16)
    w_tm = StreamNpy(os.path.join(side, "tok_is_text.npy"), np.uint8)
    page_ids, tok_off, node_off = [], [0], [0]
    held: dict = {}
    dropped = {"train": {}, "val": {}, "test": {}}
    agg = dict(n_tokens=0, n_text_tokens=0, n_markup_tokens=0, lt_in_text=0, max_depth=0)
    t0 = time.time()

    def write(pid, rec, st):
        ids, tn, npth, tw, tm = rec
        page_ids.append(pid)
        w_tok.append(ids); tok_off.append(tok_off[-1] + len(ids))
        w_tn.append(tn); w_np.append(npth); w_tw.append(tw); w_tm.append(tm.astype(np.uint8))
        node_off.append(node_off[-1] + len(npth))
        for k in ("n_tokens", "n_text_tokens", "n_markup_tokens", "lt_in_text"):
            agg[k] += st[k]
        agg["max_depth"] = max(agg["max_depth"], st["max_depth"])

    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_work, tasks)):
            for pid, rec, st in res:
                s = split_of[pid]
                if rec is None:
                    dropped[s][st] = dropped[s].get(st, 0) + 1
                    continue
                if s == "train":
                    write(pid, rec, st)
                else:
                    held[pid] = (rec, st)
            print(f"[tags] shard {k + 1}/{len(tasks)} train written {len(page_ids)} "
                  f"held {len(held)} {time.time() - t0:.0f}s", flush=True)
    n_train = len(page_ids)
    for s in ("val", "test"):
        for pid in order[s]:
            if pid in held:
                write(pid, *held[pid])
    n_val = sum(1 for p in order["val"] if p in held)
    n_test = sum(1 for p in order["test"] if p in held)
    for w in (w_tok, w_tn, w_np, w_tw, w_tm):
        w.close()
    np.save(os.path.join(a.out, "tok_off.npy"), np.array(tok_off, dtype=np.int64))
    np.save(os.path.join(side, "node_off.npy"), np.array(node_off, dtype=np.int64))
    misaligned = sum(d.get("MISALIGNED", 0) for d in dropped.values())
    meta = dict(text_pack_version=1, block_size=a.block, windowed=False, vocab_size=tmeta["vocab_size"],
                eot=EOT, tokenizer="gpt2", tokenizer_is_pretrained=True, renders_with="web_tags",
                render="tags", n_records=len(page_ids), n_train=n_train, n_val=n_val, n_test=n_test,
                split_rule=f"copied page by page from {a.text_pack}; val and test in its order",
                source_text_pack=a.text_pack, levels=a.levels, drop_long=True,
                pages_dropped=dropped, total_tokens=agg["n_tokens"],
                text_tokens=agg["n_text_tokens"], markup_tokens=agg["n_markup_tokens"],
                markup_token_frac=agg["n_markup_tokens"] / max(agg["n_text_tokens"] + agg["n_markup_tokens"], 1),
                tokens_per_record_mean=agg["n_tokens"] / max(len(page_ids), 1),
                pages_with_angle_bracket_in_text=agg["lt_in_text"],
                alignment_verified=a.verify, pages_misaligned=misaligned,
                build_wall_s=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    json.dump(page_ids, open(os.path.join(a.out, "page_ids.json"), "w"))
    json.dump(dict(dompath_version=2, render="tags", levels=a.levels, absent=ABSENT,
                   node_rule="elements and kept text nodes, no attributes; rank = index among the "
                             "parent's retained children from 0, identical to the text render",
                   token_rule="each piece BPE-encoded separately: `<tag>`, `</tag>`, `<tag/>` belong to "
                              "the ELEMENT, text pieces to the TEXT node; framing EOTs are ABSENT",
                   tok_is_text="1 where the token belongs to a text node: the visible-text tokens, "
                               "equal page by page to the text pack's tokens minus its node-separating "
                               "newlines (verified at build when alignment_verified)"),
              open(os.path.join(side, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "pages_dropped"}, indent=1))
    print("dropped:", json.dumps(dropped))
    if misaligned:
        sys.exit(f"[tags] {misaligned} pages failed the text alignment check -- the pack is not usable "
                 f"for the visible-text comparison")


if __name__ == "__main__":
    main()
