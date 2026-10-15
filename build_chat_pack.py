#!/usr/bin/env python
"""WildChat-1M conversations -> chat pack + per-token conversation/Markdown-tree sidecar.

    HF_HOME=$HF_HOME .venv/bin/python build_chat_pack.py \
        --out $TOPOS_PACKS/chat_wildchat_b2048 --workers 96

PRIVACY. WildChat is REAL users' conversations with ChatGPT and can contain personal data even
after its own redaction pass. Nothing from it may be quoted into docs, commits, reports, logs or
figures -- this builder prints aggregate statistics only, and so must anything downstream.

THE SAME FORMAT AS THE WEB PACKS, so every arm, the loader and the derived-pack tools work
unchanged: ``tokens.npy`` + ``tok_off.npy`` + ``meta.json`` + ``page_ids.json`` (conversation
hashes), and a ``_dompath`` sidecar with ``tok_node`` / ``node_path`` / ``node_off`` /
``tok_within`` (levels = 22, record-local node ids, prefix order). Records are stored train, then
val, then test, as PackedWindows assumes.

WHAT A RECORD IS
----------------
One WHOLE conversation: English (WildChat's conversation-level language field), not flagged
toxic. Each message is rendered as a role line, ``User:`` or ``Assistant:``, then its content,
and messages are joined by a blank line; framed ``[EOT] text [EOT]``, GPT-2 BPE as Web-S. The
content is verbatim except that ``\\r\\n`` / ``\\r`` become ``\\n`` (what the Markdown parser does
internally; normalising first keeps the parser's line numbers and the byte map in agreement).
A conversation longer than ``--block`` tokens is DROPPED, never cropped or windowed -- the
project's whole-documents rule -- and the count is recorded. Exact-duplicate renders are kept
once, so the same text cannot land in train and test.

WHAT A NODE IS (the tree)
-------------------------
    root (not a level) -> turn -> [section ->] Markdown blocks -> nested blocks
- **turn**: one per message, rank = message index in the conversation (0 = first user message).
  The role line belongs to the turn node, as does the blank line after the message.
- **Markdown blocks**, parsed with markdown-it-py (CommonMark + tables): headings, paragraphs,
  bullet / ordered lists, list items, block quotes, fenced / indented code, horizontal rules,
  HTML blocks, tables -> thead / tbody -> rows. Rank = index among the parent's children.
- **Sections**: a heading's following top-level blocks become its children until the next heading
  of the same or a higher level, so ``#`` / ``##`` / ``###`` give real nesting depth, as a
  document outline does. Without this, Markdown headings are flat siblings.
- A tight list's paragraphs are hidden by the parser and are NOT nodes; the item text belongs to
  the item, so a tight list costs one level per nesting, not two.
- Table cells share a line, and the parser maps blocks to LINES, so a row is the finest table node.

WHAT A TOKEN IS
---------------
A token belongs to the node owning the byte that holds its first non-whitespace character (the
web rule). A content line belongs to the deepest node whose line range covers it; a line no block
covers (a blank line between blocks) belongs to the node of the line before it, or to the turn.
Framing EOTs are ABSENT. ``tok_within`` = the token's ordinal among its node's tokens.

DROPS, all counted: conversations deeper than ``levels`` (no path is ever truncated), any rank at
or past 318 or within-node index past 1,272 (where the rotary planes wrap -- as the web packs'
``--drop-alias``), longer than ``--block``, under 64 text tokens.

INTERIOR NODES ARE NORMAL HERE: a turn's role line and its blocks are both in the stream, as in the
Python AST. That is why this pack never windows (and must not be windowed): depth's "parent =
nearest preceding node one level shallower" holds only on whole conversations.

NO LOOKAHEAD for the turn and block structure: ranks count earlier siblings, nodes are emitted in
document order. One caveat, recorded in the manifest: a Markdown block's TYPE can depend on later
lines (a paragraph followed by ``---`` becomes a heading; a table needs its delimiter row), as an
incremental Markdown parser would also have to revise. It changes which node a few tokens land in,
not any rank already emitted for an earlier sibling.
"""
from __future__ import annotations
from topos_paths import HF_HOME  # noqa: E402

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from npy_stream import StreamNpy

SNAPSHOT_GLOB = (f"{HF_HOME}/hub/datasets--allenai--WildChat-1M/snapshots/*/"
                 "data/train-*.parquet")
LEVELS = 22
ABSENT = -1
EOT = 50256
BLOCK = 2048
MIN_TEXT_TOKENS = 64
VOCAB_PADDED = 50304
SPLIT_SEED = 0
N_VAL = 2048
N_TEST = 2048
MAX_RANK = 318           # the slow plane (step 0.316/16) wraps here
MAX_WITHIN = 1272        # the token-leaf planes' slowest wraps here
_WS = b" \t\r\n\x0b\x0c"
ROLE = {"user": "User", "assistant": "Assistant"}

_ENC = None
_MD = None


MARKER_IDS: dict = {}


def _tools():
    global _ENC, _MD
    if _ENC is None:
        import tiktoken
        from markdown_it import MarkdownIt
        _ENC = tiktoken.get_encoding("gpt2")
        _MD = MarkdownIt("commonmark").enable("table")
        for role in ROLE.values():
            MARKER_IDS[role] = _ENC.encode_ordinary(f"{role}:")
    return _ENC, _MD


def normalise(content: str) -> str:
    return (content or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "�")


def block_nodes(content: str, md):
    """Markdown -> ``[(parent_local, line_lo, line_hi)]`` in document (pre-) order.

    ``parent_local`` is an index into the returned list, or -1 for the turn itself. Line ranges are
    half-open, in the content's own line numbering.
    """
    out = []
    stack = []            # open container nodes (local indices), -1 for "a hidden paragraph"
    sections = []         # (heading level, local index) at the message's top level
    for t in md.parse(content):
        if t.type == "inline" or t.map is None and t.nesting != -1:
            continue
        if t.nesting == -1:
            if t.type in ("th_close", "td_close"):
                continue
            if stack:
                stack.pop()
            continue
        hidden_para = t.type == "paragraph_open" and t.hidden
        if hidden_para:
            stack.append(-1)
            continue
        live = [s for s in stack if s != -1]
        if live:
            parent = live[-1]
        else:
            if t.type == "heading_open":
                lvl = int(t.tag[1])
                while sections and sections[-1][0] >= lvl:
                    sections.pop()
            parent = sections[-1][1] if sections else -1
        idx = len(out)
        out.append((parent, t.map[0], t.map[1]))
        if not live and t.type == "heading_open":
            sections.append((int(t.tag[1]), idx))
        if t.nesting == 1:
            stack.append(idx)
    return out


def assign_exchanges(roles):
    """Message roles -> ``[(exchange, speaker)]`` and the number of alternation exceptions.

    An exchange is a user message and the reply to it; speaker is 0 for the user, 1 for the
    assistant (a fixed code, not a sibling count). A new exchange starts at every user message, and
    also when the current exchange already holds a message from this speaker -- two assistant
    messages in a row, or a conversation that opens with the assistant. Those cases, and a user
    message left without a reply before the next one, are counted as exceptions.
    """
    out, ex, filled, exceptions = [], -1, set(), 0
    for i, r in enumerate(roles):
        sp = 0 if r == "user" else 1
        if sp == 0 or ex < 0 or sp in filled:
            if ex >= 0 and sp == 0 and 1 not in filled:
                exceptions += 1                 # previous user message got no reply
            if sp == 1:
                exceptions += 1                 # reply with no user message before it
            ex += 1
            filled = set()
        filled.add(sp)
        out.append((ex, sp))
    return out, exceptions


def render_conversation(conv, enc, md, block: int, levels: int):
    """One conversation -> ``(ids, tok_node, node_path, tok_within, stats)`` or ``(None, reason)``.

    Tree: root (not a level) -> exchange (rank = exchange index) -> speaker (rank 0 = user,
    1 = assistant) -> [heading section ->] Markdown blocks. Each message is its own speaker node;
    its role line ``User:`` / ``Assistant:`` and the blank line after it belong to that node, so a
    user message and its reply never share a node.
    """
    roles = [m["role"] for m in conv]
    if any(r not in ROLE for r in roles):
        return None, "role"
    msgs = [(ROLE[m["role"]], normalise(m["content"])) for m in conv]
    exch, n_exc = assign_exchanges(roles)
    parts, paths, byte_node, msg_of_node, msg_start_byte = [], [], [], [], []
    ex_node = {}
    marker_lines = 0
    acc = 0
    for mi, ((role, content), (ex, sp)) in enumerate(zip(msgs, exch)):
        if ex not in ex_node:
            ex_node[ex] = len(paths)
            paths.append([ex]); msg_of_node.append(-1)
        sp_id = len(paths)
        paths.append([ex, sp]); msg_of_node.append(mi)
        head = f"{role}:\n"
        blocks = block_nodes(content, md)
        child_count, local_to_global = {}, []
        for parent, lo, hi in blocks:
            ppath = paths[sp_id] if parent == -1 else paths[local_to_global[parent]]
            r = child_count.get(parent, 0)
            child_count[parent] = r + 1
            local_to_global.append(len(paths))
            paths.append(ppath + [r]); msg_of_node.append(mi)
        lines = content.split("\n")
        marker_lines += sum(1 for ln in lines if ln.startswith(("User:", "Assistant:")))
        line_node = [sp_id] * len(lines)
        line_depth = [0] * len(lines)
        for li_local, (parent, lo, hi) in enumerate(blocks):
            g = local_to_global[li_local]
            d = len(paths[g])
            for ln in range(lo, min(hi, len(lines))):
                if d >= line_depth[ln]:
                    line_node[ln], line_depth[ln] = g, d
        for ln in range(1, len(lines)):
            if line_depth[ln] == 0:
                line_node[ln] = line_node[ln - 1]
        sep = "\n\n" if mi < len(msgs) - 1 else ""
        parts.append(head + content + sep)
        msg_start_byte.append(acc)
        bn = [sp_id] * len(head.encode("utf-8"))
        for ln, text in enumerate(lines):
            nb = len(text.encode("utf-8")) + (1 if ln < len(lines) - 1 else 0)
            bn.extend([line_node[ln]] * nb)
        bn.extend([sp_id] * len(sep.encode("utf-8")))
        byte_node.extend(bn)
        acc += len(bn)
    text = "".join(parts)
    body = enc.encode_ordinary(text)
    if len(body) < MIN_TEXT_TOKENS:
        return None, "short"
    if len(body) + 2 > block:
        return None, "long"
    max_depth = max(len(p) for p in paths)
    if max_depth > levels:
        return None, "deep"
    if max(r for p in paths for r in p) >= MAX_RANK:
        return None, "alias"
    byte_node = np.asarray(byte_node, dtype=np.int32)
    assert len(byte_node) == len(text.encode("utf-8"))
    firsts = np.empty(len(body), dtype=np.int64)
    tok_start = np.empty(len(body), dtype=np.int64)
    off = 0
    for j, tid in enumerate(body):
        b = enc.decode_single_token_bytes(tid)
        k = 0
        while k < len(b) and b[k] in _WS:
            k += 1
        tok_start[j] = off
        firsts[j] = off + (k if k < len(b) else 0)
        off += len(b)
    assert off == len(byte_node)
    # the role marker must tokenise identically every time: a token boundary at each message's
    # first byte, then exactly the marker's tokens ("User" ":" / "Assistant" ":")
    pos = {int(b): j for j, b in enumerate(tok_start)}
    marker_ok = True
    for (role, _), b0 in zip(msgs, msg_start_byte):
        j = pos.get(b0)
        want = MARKER_IDS[role]
        if j is None or list(body[j:j + len(want)]) != want:
            marker_ok = False
    if not marker_ok:
        return None, "marker"
    tok_node = byte_node[firsts]
    mon = np.asarray(msg_of_node, dtype=np.int64)
    tok_msg = mon[tok_node]
    msg_len = np.bincount(tok_msg[tok_msg >= 0], minlength=len(msgs))
    if msg_len.max() > MAX_WITHIN:
        return None, "msg_long"                 # a message's within-message index would wrap
    order = np.argsort(tok_node, kind="stable")
    grouped = tok_node[order]
    grp_start = np.searchsorted(grouped, grouped, side="left")
    within = np.empty(len(order), dtype=np.int64)
    within[order] = np.arange(len(order)) - grp_start
    if within.max() > MAX_WITHIN:
        return None, "alias"
    node_path = np.full((len(paths), levels), ABSENT, dtype=np.int16)
    for i, p in enumerate(paths):
        node_path[i, :len(p)] = p
    ids = np.concatenate([[EOT], body, [EOT]]).astype(np.uint16)
    tn = np.concatenate([[ABSENT], tok_node, [ABSENT]]).astype(np.int32)
    tw = np.concatenate([[ABSENT], within, [ABSENT]]).astype(np.int16)
    keep = int(tn.max()) + 1
    node_path = node_path[:keep]
    sp_ = sorted(tuple(p) for p in paths[:keep])
    n_interior = sum(1 for i, q in enumerate(sp_[:-1]) if sp_[i + 1][:len(q)] == q and len(sp_[i + 1]) > len(q))
    depth_of_tok = (node_path[tok_node] >= 0).sum(1)
    user_tok = int(sum(msg_len[i] for i, (r, _) in enumerate(msgs) if r == "User"))
    st = dict(n_text_tokens=len(body), nodes=keep, nodes_interior=n_interior, messages=len(msgs),
              exchanges=exch[-1][0] + 1, alternation_exceptions=n_exc, marker_lines_in_text=marker_lines,
              user_tokens=user_tok, assistant_tokens=int(msg_len.sum()) - user_tok,
              msg_len=msg_len.tolist(), max_depth=max_depth,
              depth_hist=np.bincount(depth_of_tok, minlength=levels + 1).tolist(),
              ranks=[int(r) for p in paths[:keep] for r in p[2:]],      # within-message ranks
              decoded_ok=enc.decode(body) == text)
    return (ids, tn, node_path, tw, st), None


def _process(args):
    path, rg, block, levels = args
    import pyarrow.parquet as pq
    enc, md = _tools()
    tb = pq.ParquetFile(path).read_row_group(rg, columns=["conversation_hash", "language", "toxic",
                                                          "conversation"])
    out, drops = [], dict(read=0, lang=0, toxic=0, role=0, short=0, long=0, deep=0, alias=0,
                     msg_long=0, marker=0, error=0)
    for row in tb.to_pylist():
        drops["read"] += 1
        if row["language"] != "English":
            drops["lang"] += 1
            continue
        if row["toxic"]:
            drops["toxic"] += 1
            continue
        try:
            rec, why = render_conversation(row["conversation"], enc, md, block, levels)
        except Exception:
            drops["error"] += 1
            continue
        if rec is None:
            drops[why] += 1
            continue
        ids = rec[0]
        texthash = hashlib.blake2b(ids.tobytes(), digest_size=12).hexdigest()
        out.append((row["conversation_hash"], texthash) + rec)
    return path, rg, out, drops


def split_key(conv_hash: str) -> int:
    return int(hashlib.blake2b(conv_hash.encode(), digest_size=8, key=str(SPLIT_SEED).encode()).hexdigest(), 16)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--levels", type=int, default=LEVELS)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--limit-groups", type=int, default=0, help="smoke: process only this many row groups")
    a = ap.parse_args()
    if "HF_HOME" not in os.environ:
        sys.exit(f"[chat] REFUSED: set HF_HOME={HF_HOME}")
    t0 = time.time()
    import pyarrow.parquet as pq
    files = sorted(glob.glob(SNAPSHOT_GLOB))
    if not files:
        sys.exit("[chat] no WildChat shards; run fetch_wildchat.py")
    tasks = [(f, g, a.block, a.levels) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
    if a.limit_groups:
        tasks = tasks[:a.limit_groups]
    recs, drops = [], {}
    with Pool(a.workers) as pool:
        for i, (f, g, out, d) in enumerate(pool.imap_unordered(_process, tasks), 1):
            recs.extend((os.path.basename(f), g, j, *r) for j, r in enumerate(out))
            for k, v in d.items():
                drops[k] = drops.get(k, 0) + v
            if i % 50 == 0 or i == len(tasks):
                print(f"[chat] {i}/{len(tasks)} groups, {len(recs):,} kept, {time.time()-t0:.0f}s", flush=True)
    # deterministic order, then exact-duplicate renders kept once
    recs.sort(key=lambda r: (r[0], r[1], r[2]))
    seen, uniq, n_dup = set(), [], 0
    for r in recs:
        if r[4] in seen:
            n_dup += 1
            continue
        seen.add(r[4])
        uniq.append(r)
    recs = uniq
    bad_decode = sum(1 for r in recs if not r[9]["decoded_ok"])
    if bad_decode:
        sys.exit(f"[chat] BUG: {bad_decode} records do not decode back to their rendered text")
    if len(recs) < N_VAL + N_TEST + 1:
        sys.exit(f"[chat] REFUSED: only {len(recs)} conversations")
    by_key = sorted(range(len(recs)), key=lambda i: split_key(recs[i][3]))
    val_i, test_i = by_key[:N_VAL], by_key[N_VAL:N_VAL + N_TEST]
    held = set(val_i) | set(test_i)
    train_i = [i for i in range(len(recs)) if i not in held]

    os.makedirs(a.out, exist_ok=True)
    side = a.out + "_dompath"
    os.makedirs(side, exist_ok=True)
    w_tok = StreamNpy(os.path.join(a.out, "tokens.npy"), np.uint16)
    w_tn = StreamNpy(os.path.join(side, "tok_node.npy"), np.int32)
    w_np = StreamNpy(os.path.join(side, "node_path.npy"), np.int16, cols=a.levels)
    w_tw = StreamNpy(os.path.join(side, "tok_within.npy"), np.int16)
    tok_off, node_off, ids_out = [0], [0], []
    depth_hist = np.zeros(a.levels + 1, dtype=np.int64)
    ranks, msgs_n, exch_n, msg_lens, nodes, interior, tokens = [], [], [], [], 0, 0, 0
    role_tok = dict(user=0, assistant=0)
    n_exc_conv = n_exc = n_marker_conv = n_marker_lines = 0
    for i in train_i + val_i + test_i:
        r = recs[i]
        ids, tn, npth, tw, st = r[5], r[6], r[7], r[8], r[9]
        w_tok.append(ids); w_tn.append(tn); w_np.append(npth); w_tw.append(tw)
        tok_off.append(tok_off[-1] + len(ids)); node_off.append(node_off[-1] + len(npth))
        ids_out.append(r[3])
        depth_hist += np.asarray(st["depth_hist"])
        ranks.extend(st["ranks"]); msgs_n.append(st["messages"]); exch_n.append(st["exchanges"])
        msg_lens.extend(st["msg_len"])
        role_tok["user"] += st["user_tokens"]; role_tok["assistant"] += st["assistant_tokens"]
        n_exc += st["alternation_exceptions"]; n_exc_conv += int(st["alternation_exceptions"] > 0)
        n_marker_lines += st["marker_lines_in_text"]; n_marker_conv += int(st["marker_lines_in_text"] > 0)
        nodes += st["nodes"]; interior += st["nodes_interior"]; tokens += len(ids)
    for w in (w_tok, w_tn, w_np, w_tw):
        w.close()
    np.save(os.path.join(a.out, "tok_off.npy"), np.array(tok_off, dtype=np.int64))
    np.save(os.path.join(side, "node_off.npy"), np.array(node_off, dtype=np.int64))
    n = len(ids_out)
    ranks = np.asarray(ranks) if ranks else np.zeros(1)
    train_tokens = int(tok_off[len(train_i)])
    meta = dict(text_pack_version=1, block_size=a.block, windowed=False, vocab_size=VOCAB_PADDED, eot=EOT,
                tokenizer="gpt2", tokenizer_is_pretrained=True, renders_with="chat_markdown",
                render="chat", n_records=n, n_train=len(train_i), n_val=len(val_i), n_test=len(test_i),
                split_seed=SPLIT_SEED,
                split_rule="by conversation: blake2b(conversation_hash) order; first 2,048 val, next 2,048 test; "
                           "records stored train, then val, then test",
                source="allenai/WildChat-1M (ODC-BY)", filter="language == English and not toxic",
                min_text_tokens=MIN_TEXT_TOKENS, levels=a.levels, drop_long=True, drop_deep=True, drop_alias=True,
                max_rank=MAX_RANK, max_within=MAX_WITHIN, conversations_read=drops.get("read", 0),
                dropped=drops, duplicates_dropped=n_dup,
                total_tokens=tokens, train_tokens=train_tokens, tokens_per_record_mean=tokens / n,
                records_cropped_frac=0.0, text_tokens_lost_to_crop_frac=0.0,
                nodes_interior=interior, build_wall_s=round(time.time() - t0, 1))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(a.out, "page_ids.json"), "w") as f:
        json.dump(ids_out, f)
    tok_depth = depth_hist / max(depth_hist.sum(), 1)
    manifest = dict(dompath_version=2, text_pack=a.out, levels=a.levels, absent=ABSENT, render="chat",
                    node_rule="root (not a level) -> exchange (rank = exchange index; a user message and its "
                              "reply) -> speaker (rank 0 = user, 1 = assistant; one node per message) -> "
                              "[heading section ->] Markdown blocks (markdown-it-py CommonMark + tables) -> nested blocks; "
                              "tight-list paragraphs are not nodes; a table row is the finest table node",
                    token_rule="a token belongs to the node owning its first non-whitespace byte; a content line "
                               "belongs to the deepest block covering it, an uncovered line to the previous line's "
                               "node; the role line and the blank line after a message belong to the message's speaker node; framing "
                               "EOTs are ABSENT",
                    within_rule="tok_within = the token's ordinal among its node's tokens; -1 for framing EOTs",
                    exchange_rule=("a new exchange starts at every user message, and when the current exchange "
                                   "already holds a message from this speaker (two replies in a row, or a "
                                   "conversation opening with the assistant); those and unanswered user messages "
                                   "are counted as alternation exceptions"),
                    lookahead=("exchange and block ranks count earlier siblings only; a block's TYPE can depend on "
                               "later lines (setext headings, table delimiter rows), which moves a few tokens "
                               "between nodes but never changes an earlier sibling's rank"),
                    turn_rank_note=("the exchange rank is per-conversation, like a list item's rank; "
                                    "it is NOT a global coordinate the way a spreadsheet row is shared by every "
                                    "cell in it -- two conversations' exchange 3 share nothing but the index. The "
                                    "speaker rank IS global: 0 always means the user"),
                    nodes_total=nodes, nodes_interior=interior,
                    nodes_interior_frac=interior / max(nodes, 1),
                    token_depth_frac={str(d): round(float(v), 5) for d, v in enumerate(tok_depth) if v > 0},
                    within_message_rank_p50=float(np.percentile(ranks, 50)),
                    within_message_rank_p99=float(np.percentile(ranks, 99)),
                    messages_per_conv_p50=float(np.percentile(msgs_n, 50)),
                    messages_per_conv_p99=float(np.percentile(msgs_n, 99)),
                    exchanges_per_conv_p50=float(np.percentile(exch_n, 50)),
                    exchanges_per_conv_p99=float(np.percentile(exch_n, 99)),
                    tokens_per_message_p50=float(np.percentile(msg_lens, 50)),
                    tokens_per_message_p99=float(np.percentile(msg_lens, 99)),
                    user_token_share=role_tok["user"] / max(sum(role_tok.values()), 1),
                    alternation_exceptions=n_exc, conversations_with_alternation_exceptions=n_exc_conv,
                    conversations_with_marker_like_lines_in_text=n_marker_conv,
                    marker_like_lines_in_text=n_marker_lines)
    with open(os.path.join(side, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items()}, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
