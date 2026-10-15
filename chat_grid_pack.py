#!/usr/bin/env python
"""Derive the chat GRID packs from a chat pack: (exchange, speaker, token-in-message) [+ Markdown depth].

    .venv/bin/python chat_grid_pack.py --pack $TOPOS_PACKS/chat_wildchat_b2048

THE USER'S QUESTION (2026-10-04): "depth and width is just grid but one axis would be the turn and
the other would be the tokens within the turn", and "split between the user and the response so that
it is clear who is speaking". So a token's chat coordinate is (exchange, speaker, index in its
message): which user-message-and-reply exchange, who is speaking (0 user, 1 assistant), and how far
into that message -- the chat analogue of a spreadsheet's (row, column), with the speaker explicit
rather than left to turn parity.

HOW, WITHOUT NEW MODEL CODE. ``rope_all`` rotates sequence planes, the per-level tree planes by
``present * theta + rank * step``, and the token-leaf planes by the sidecar's ``tok_within``. So:

- ``<pack>_grid``: every MESSAGE is one node whose path is ``[exchange, speaker]`` with levels
  2..21 ABSENT (an absent level contributes exactly zero rotation, so those planes are idle);
  ``tok_node`` maps every token of a message -- its role line first -- to it; ``tok_within`` is the
  token's index within its message. Under ``rope_all``: sequence on 16 planes, exchange on 2,
  speaker on 2, the within-message index on the 4 token planes (ladder BASE_STEP / 4**p, slowest
  wrapping at 1,272 -- the chat builder drops conversations with a longer message, and this
  script REFUSES a pack where one exceeds it rather than letting the coordinate wrap).
- ``<pack>_griddepth``: as above plus Markdown depth: nodes are the distinct (exchange, speaker,
  depth) triples of a record, path ``[exchange, speaker, depth]`` with depth = levels below the
  speaker node (0 = the role line and text outside any block, 1 = a top-level block, ...).

EVERYTHING EXCEPT THE SIDECAR'S ``tok_node`` / ``node_path`` / ``node_off`` / ``tok_within`` IS A
SYMLINK to the source pack, so the token stream is byte-identical across every chat arm.

Also verifies the SOURCE pack on every record: node tables in prefix order (every node's parent
path appears earlier in its record), ``tok_node`` in range, framing EOTs absent.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from npy_stream import StreamNpy

ABSENT = -1
MAX_WITHIN = 1272


def link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pack", required=True)
    a = ap.parse_args()
    src, side = a.pack.rstrip("/"), a.pack.rstrip("/") + "_dompath"
    tok_off = np.load(os.path.join(src, "tok_off.npy"))
    tn_all = np.load(os.path.join(side, "tok_node.npy"), mmap_mode="r")
    np_all = np.load(os.path.join(side, "node_path.npy"), mmap_mode="r")
    no_all = np.load(os.path.join(side, "node_off.npy"))
    levels = np_all.shape[1]
    outs = {}
    for name in ("grid", "griddepth"):
        out = f"{src}_{name}"
        os.makedirs(out, exist_ok=True)
        os.makedirs(out + "_dompath", exist_ok=True)
        for f in sorted(os.listdir(src)):
            link(os.path.join(src, f), os.path.join(out, f))
        o = out + "_dompath"
        outs[name] = dict(dir=o, tn=StreamNpy(os.path.join(o, "tok_node.npy"), np.int32),
                          np=StreamNpy(os.path.join(o, "node_path.npy"), np.int16, cols=levels),
                          tw=StreamNpy(os.path.join(o, "tok_within.npy"), np.int16), off=[0])
    n_rec = len(tok_off) - 1
    turn_len_max, turn_lens, bad_prefix = 0, [], 0
    for r in range(n_rec):
        tn = np.asarray(tn_all[tok_off[r]:tok_off[r + 1]])
        npth = np.asarray(np_all[no_all[r]:no_all[r + 1]])
        inside = tn >= 0
        assert tn[0] == ABSENT and tn[-1] == ABSENT, r
        assert tn[inside].max() < len(npth), r
        # prefix order: each node's parent path occurs earlier in the table
        seen = set()
        for i, row in enumerate(npth):
            d = int((row >= 0).sum())
            p = tuple(int(x) for x in row[:d])
            if d > 1 and p[:-1] not in seen:
                bad_prefix += 1
            seen.add(p)
        depth = (npth >= 0).sum(1)
        assert (depth[tn[inside]] >= 2).all(), f"record {r}: a token on an exchange node"
        # message key = (exchange, speaker), encoded as one integer for run-finding
        msg_of_node = npth[:, 0].astype(np.int64) * 2 + np.maximum(npth[:, 1], 0)
        t_turn = np.full(len(tn), -1, dtype=np.int64)
        t_turn[inside] = msg_of_node[tn[inside]]
        t_depth = np.zeros(len(tn), dtype=np.int64)
        t_depth[inside] = depth[tn[inside]] - 2
        # within-turn ordinal; turns are contiguous in the stream
        within = np.full(len(tn), ABSENT, dtype=np.int64)
        idx = np.nonzero(inside)[0]
        tt = t_turn[idx]
        starts = np.r_[0, np.nonzero(np.diff(tt))[0] + 1]
        assert len(np.unique(tt)) == len(starts), f"record {r}: a message is not contiguous"
        run = np.zeros(len(idx), dtype=np.int64)
        for s, e in zip(starts, np.r_[starts[1:], len(idx)]):
            run[s:e] = np.arange(e - s)
            turn_lens.append(e - s)
        within[idx] = run
        turn_len_max = max(turn_len_max, int(run.max()) + 1)
        # grid: node = turn
        turns = np.unique(tt)
        g_np = np.full((len(turns), levels), ABSENT, dtype=np.int16)
        g_np[:, 0] = turns // 2
        g_np[:, 1] = turns % 2
        g_tn = np.full(len(tn), ABSENT, dtype=np.int32)
        g_tn[idx] = np.searchsorted(turns, tt)
        # griddepth: node = (turn, depth) in order of first appearance
        pairs = list(dict.fromkeys(zip(tt.tolist(), t_depth[idx].tolist())))
        pid = {p: i for i, p in enumerate(pairs)}
        gd_np = np.full((len(pairs), levels), ABSENT, dtype=np.int16)
        gd_np[:, 0] = [p[0] // 2 for p in pairs]
        gd_np[:, 1] = [p[0] % 2 for p in pairs]
        gd_np[:, 2] = [p[1] for p in pairs]
        gd_tn = np.full(len(tn), ABSENT, dtype=np.int32)
        gd_tn[idx] = [pid[p] for p in zip(tt.tolist(), t_depth[idx].tolist())]
        tw = np.minimum(within, np.iinfo(np.int16).max).astype(np.int16)
        for name, (t_n, n_p) in (("grid", (g_tn, g_np)), ("griddepth", (gd_tn, gd_np))):
            w = outs[name]
            w["tn"].append(t_n); w["np"].append(n_p); w["tw"].append(tw)
            w["off"].append(w["off"][-1] + len(n_p))
    for name, w in outs.items():
        for k in ("tn", "np", "tw"):
            w[k].close()
        np.save(os.path.join(w["dir"], "node_off.npy"), np.array(w["off"], dtype=np.int64))
    tl = np.asarray(turn_lens)
    stats = dict(records=n_rec, nodes_not_in_prefix_order=bad_prefix, messages=len(tl),
                 tokens_per_message_p50=float(np.percentile(tl, 50)),
                 tokens_per_message_p99=float(np.percentile(tl, 99)),
                 tokens_per_message_max=int(tl.max()), messages_over_slow_wrap=int((tl > MAX_WITHIN).sum()),
                 messages_over_318=int((tl > 318).sum()))
    for name, w in outs.items():
        man = dict(derived_from=side, render="chat_" + name, levels=levels, absent=ABSENT,
                   node_rule=("one node per message, path [exchange, speaker]" if name == "grid" else
                              "one node per distinct (exchange, speaker, Markdown depth below the speaker) in a "
                              "record, path [exchange, speaker, depth]"),
                   within_rule="tok_within = the token's index within its MESSAGE (role line first); -1 for framing EOTs",
                   arm="rope_all: sequence 16 planes; level 0 = exchange (2 planes); level 1 = speaker (2 planes)"
                       + ("; level 2 = Markdown depth (2 planes)" if name == "griddepth" else "")
                       + "; token planes (4) = index within the message; other levels ABSENT (zero rotation)",
                   **stats)
        json.dump(man, open(os.path.join(w["dir"], "manifest.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    if bad_prefix:
        raise SystemExit(f"source pack has {bad_prefix} nodes out of prefix order")
    if stats["messages_over_slow_wrap"]:
        raise SystemExit(f"{stats['messages_over_slow_wrap']} messages exceed {MAX_WITHIN} tokens: the "
                         f"within-message coordinate would wrap on the slowest token plane")


if __name__ == "__main__":
    main()
