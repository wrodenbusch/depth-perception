"""Tests for build_chat_pack.render_conversation on SYNTHETIC conversations (never WildChat text).

    .venv/bin/python -m pytest -q test_chat_pack.py
"""
import numpy as np

import build_chat_pack as B

ENC, MD = B._tools()


def conv(*msgs):
    return [dict(role=r, content=c) for r, c in msgs]


def render(c):
    rec, why = B.render_conversation(c, ENC, MD, 2048, 22)
    assert why is None, why
    return rec


LONG = "This sentence is filler so the conversation clears the minimum length. " * 4

C = conv(("user", "How do I sort a list?\n" + LONG),
         ("assistant", "# Sorting\n\nUse:\n\n- `sorted(xs)`\n- `xs.sort()`\n  - in place\n\n## Example\n\n"
                       "```python\nxs = [3, 1]\n```\n"),
         ("user", "Thanks"),
         ("assistant", "1. one\n2. two"))


def node_texts(rec):
    ids, tn, npth, tw, st = rec
    out = {}
    for t, n in zip(ids[1:-1], tn[1:-1]):
        out.setdefault(int(n), []).append(ENC.decode([int(t)]))
    return {tuple(int(x) for x in npth[n] if x >= 0): "".join(v) for n, v in out.items()}


def test_decodes_and_frames():
    ids, tn, npth, tw, st = render(C)
    assert ids[0] == B.EOT and ids[-1] == B.EOT and tn[0] == -1 and tn[-1] == -1
    assert st["decoded_ok"]


def test_exchange_and_speaker_levels():
    by_path = node_texts(render(C))
    # role lines sit on the speaker nodes: exchange 0 user, exchange 0 assistant, exchange 1 ...
    assert by_path[(0, 0)].startswith("User:")
    assert by_path[(0, 1)].startswith("Assistant:")
    assert by_path[(1, 0)].startswith("User:")
    assert by_path[(1, 0, 0)] == "Thanks"               # a paragraph is a block below the speaker
    assert by_path[(1, 1)].startswith("Assistant:")
    # no token sits on an exchange node
    assert not any(len(p) == 1 for p in by_path)


def test_markdown_nesting():
    by_path = node_texts(render(C))
    # the H1 is the first block of the reply; its section holds the paragraph, the list, the H2
    h1 = [p for p, t in by_path.items() if "Sorting" in t][0]
    assert h1 == (0, 1, 0)
    lst_items = sorted(p for p, t in by_path.items() if "sorted" in t or "xs.sort" in t)
    assert all(len(p) > len(h1) and p[:3] == h1 for p in lst_items)
    nested = [p for p, t in by_path.items() if "in place" in t][0]
    assert len(nested) == len(lst_items[1]) + 2       # list -> item
    h2 = [p for p, t in by_path.items() if "Example" in t][0]
    code = [p for p, t in by_path.items() if "xs = [3" in t][0]
    assert code[:len(h2)] == h2 and len(code) == len(h2) + 1


def test_exchanges_for_nonalternating():
    ex, n = B.assign_exchanges(["user", "user", "assistant", "assistant"])
    assert ex == [(0, 0), (1, 0), (1, 1), (2, 1)] and n == 2
    ex, n = B.assign_exchanges(["assistant", "user", "assistant"])
    assert ex == [(0, 1), (1, 0), (1, 1)] and n == 1


def test_within_message_drop():
    long_reply = "word " * 1400
    rec, why = B.render_conversation(conv(("user", LONG), ("assistant", long_reply)), ENC, MD, 2048, 22)
    assert rec is None and why == "msg_long"


def test_prefix_order():
    ids, tn, npth, tw, st = render(C)
    seen = set()
    for row in npth:
        p = tuple(int(x) for x in row if x >= 0)
        assert len(p) == 1 or p[:-1] in seen
        seen.add(p)
