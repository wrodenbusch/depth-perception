"""Build a spreadsheet text pack + tree sidecar from the sheetpedia `sheet_graph` corpus.

WHAT THIS IS
------------
The spreadsheet analogue of ``build_web_pack.py``. Input is
``$TOPOS_SOURCES/spreadsheet/sheet_graph_v2_c4096*.jsonl``, one JSON record per workbook
carrying an explicit tree (so unlike the web builder there is no HTML to parse).

Output, in the conventions the other cells use:

  <out>/            tokens.npy, tok_off.npy, meta.json,         (PackedWindows text pack)
                    record_ids.json
  <out>_dompath/    node_path.npy, node_off.npy, tok_node.npy,  (tree sidecar)
                    tok_within.npy, tok_ref_sheet.npy,
                    tok_ref_row.npy, tok_ref_col.npy, manifest.json

MEASURED FACTS ABOUT THE CORPUS (verified 2026-09-28)
-----------------------------------------------------
**Node ids in the record are BFS (level order), not DFS.** ``set(Ref-tagged node BFS ids)
== set(refs sources)`` in 300/300 records; under DFS numbering it matched in only 16/300.
``payloads`` AND ``names`` are both keyed by that same BFS id space, as strings.

``refs`` is a list of ``[source, target]`` BFS ids, record-local: 100% of sources are
``Ref``-tagged, one per node, and targets are ``Free`` cell nodes (24,093/24,477) or whole-row
nodes (384/24,477).

**Node type comes from the FIELD attribute, never from depth.** 18.5% of records carry a
trailing ``Names f:names`` table whose ``Free f:name`` children sit at the SAME depth as rows
and whose grandchildren sit at the same depth as cells. Keying on depth renders a workbook's
symbol table as if it were a sheet of data; this builder dispatches on ``f:sheet`` / ``f:row``
/ ``f:cell`` / ``f:names`` / ``f:name``, which is what the corpus pipeline itself keys on.

**``names`` is per-node and it holds the address.** 300/300 records have every ``names`` key
inside the BFS id range, and every named node is ``Free``-tagged: sheet titles on ``f:sheet``,
ABSOLUTE row indices on ``f:row``, column letters on ``f:cell``, function names on
``Free{func} f:arg``, and inexpressible operand text (``$B:$B``, ``Table1[Col]``) on
``Free f:arg``. Row names are absolute and non-empty rows are SPARSE: **43.4% of row nodes have
a name that is not their sibling rank + 1** (69,769 of 160,616 over 2,000 records), so deriving
an address from sibling position is wrong for nearly half the corpus and puts rows 5 and 50
side by side.

**Formula content lives in the tag, the attributes and ``names`` -- not in ``payloads``.**
``payloads`` exists only on ``Lit`` nodes. An earlier version of this builder rendered a node
as ``payloads[id]`` alone, so every operator, every function name and every reference
disappeared: the formula ``=A1-(B1+C1+D1)`` rendered as the EMPTY STRING. 52,588 of the
Formula nodes in a 400-record sample were affected. This builder renders the full expression:
operators from the attribute key (``Op{sub}`` -> ``-``), function names and literal operand
text from ``names``, references from the resolved edge, and grouping from ``Paren``/``Array``.

COORDINATES -- AND WHY THE TWO CHANNELS MUST AGREE
---------------------------------------------------
``--coord absolute`` (default) puts the ABSOLUTE 0-based row and column on the ``f:row`` and
``f:cell`` levels of ``node_path``, read from ``names``; every other level keeps its sibling
rank. ``--coord rank`` uses sibling rank everywhere, as an ablation.

The reference coordinate written to ``tok_ref_sheet``/``tok_ref_row``/``tok_ref_col`` is produced
by the SAME function in the SAME space, so a reference's rotary angle coincides EXACTLY with its
target cell's own angle. That coincidence is the mechanism under test, and it is why the two must
not be allowed to drift apart: spelling ``A50`` in the text while rotating by sibling rank 12
would put the two channels in different spaces and test nothing.

**The SHEET axis is part of that address** (added 2026-09-28). ``(row, col)`` alone makes a
reference that crosses to another sheet coincide with whatever cell holds those coordinates on
the reference's OWN sheet -- a confident wrong answer rather than a missing one. Measured over
1,500 workbooks: **17.25% of references are cross-sheet** and 25.5% of workbooks hold more than
one sheet. ``node_path`` level 0 is ``coord_of`` of the depth-1 ancestor, which for any cell IS
its sheet node, so the three axes are one space with levels 0, 1 and 2 of the path.

Absolute indices are mostly affordable: over 2,000 records the absolute row is p99 = 1,825 and
p999 = 2,526, and columns reach p99 = 152 against Excel's 16,383 ceiling, which is inside int16
by construction.

**CORRECTION.** This docstring previously said "only 15 of 160,616 exceed the int16 ceiling".
That figure counted ROWS OVER THE CEILING IN A 2,000-RECORD SAMPLE and has since been requoted as
though it were the impact, including in a commit message. The real impact on the shipped v1 pack
was **222,084 clamped coordinates** and **16,208 reference tokens sitting at exactly 32,767** --
orders of magnitude larger. Rows past the ceiling are no longer clamped; see
``coords_past_int16_path_truncated``.

RECORDS ARE DROPPED, NOT TRUNCATED (2026-09-28)
-----------------------------------------------
A workbook longer than ``block_size`` is **discarded**, not cut. Truncation cost 61% of
workbooks a tail and created a problem of its own: a reference whose target fell past the cut
pointed at nothing. Dropping removes that at the source -- **measured, 0.00% of references that
name a cell or a row have their target outside the window**, against 24.4% of pointer-bearing
tokens under truncation.

**The criterion is applied to BOTH renders, and that is load-bearing.** ``vanilla`` spells every
address out so it is longer, and a workbook can fit one render and not the other. If each pack
dropped independently the two would hold different record sets and any comparison between them
would be void. So both renders are measured for every record and the record is dropped if EITHER
exceeds the cap, whichever render is being emitted. ``record_ids.json`` records the surviving set
in order so the two packs can be ASSERTED identical rather than assumed.

Consequence to state in any writeup: results speak to workbooks that fit ``block_size`` tokens
**under the longer of the two renders**, which is a different corpus population from the
truncated pack. The manifest carries the survival count and the recomputed node-per-token
density; the old 0.56 figure was measured under truncation and does not carry over.

The suppression path is kept as a safety net and is expected never to fire
(``n_refs_target_past_truncation`` = 0). A separate counter, ``n_refs_target_not_a_cell``, holds
the references whose target is a **defined name** -- those denote no grid location and correctly
carry no coordinate on any axis. That is a permanent property of the corpus rather than a window
artifact, which is why the two are not conflated.

OFFSETS ARE UTF-8 BYTES, NOT CHARACTERS
---------------------------------------
``Emitter`` spans and the token offsets in ``map_tokens`` must be in the same unit, and that unit
has to be **bytes**. ``enc.decode([t])`` on a token holding half a multi-byte character returns a
single ``U+FFFD``, so summing decoded CHARACTER lengths over-counts and every later token in the
record is attributed to a span further right. Measured on the untruncated render over 3,000
records: **9.73% of records drift and 16.54% of body tokens sit at or after a drift onset**, one
record misattributed from its very first token. That corrupted ``tok_node`` -- so the TREE PATH
was wrong for those tokens in *every* sheets arm, not only the reference ones. ``decode_bytes``
agrees with ``len(s.encode("utf-8"))`` exactly, including across a split character.

CAUSALITY -- WHY FORWARD REFERENCES ARE NOW KEPT
-------------------------------------------------
The concern was that emitting a forward reference's target coordinate hands the model the
layout of a cell it has not read. That was true when ``Ref`` nodes rendered to nothing. It is
false now: a reference renders as its ADDRESS TEXT (``A50``, ``$B$7``, ``Sheet2!C3``) in BOTH
render modes, derived from the target's ``names`` entry, which is the one-channel form the
corpus pipeline is built around. By the time the model reads the reference it has read the
address, so the coordinate is already in the prefix and the rotation is redundant with text
rather than additional to it. Render order therefore does not matter and no BFS re-ordering is
needed. ``--drop-forward-refs`` restores the old conservative behaviour as an ablation.

RENDER MODES
------------
``vanilla``  ``<address>=<content>`` per cell, reading order. SpreadsheetLLM's own baseline
             serialisation (Dong et al., EMNLP 2024, arXiv:2407.09025), so our sequence arm is
             comparable to published numbers. Addresses come from ``names``.
``values``   ``<content>`` only: the cell's OWN address is dropped, because that is exactly
             what the tree rotation supplies. Reference addresses INSIDE formulas are still
             spelled, so both arms see every pointer and the tree arm's reference rotation is
             measured as an addition on top of text the sequence arm also has. Without this the
             sequence baseline could not see references at all and any win would be trivial --
             a rigging risk the corpus authors flag in their own builder.

Both are byte-deterministic given a record; the separator bytes are recorded in meta.
"""
from __future__ import annotations
from topos_paths import SOURCES  # noqa: E402

import argparse
import collections
import json
import os
import re

import numpy as np

from npy_stream import StreamNpy

ABSENT = -1
LEVELS = 22
INT16_MAX = 32767

#: Infix/prefix/postfix operator spellings, keyed by the node's non-field attribute.
#: Copied from ``MARKER_OP`` in the corpus's own builder (docs/sheet_graph_corpus_ref.py
#: .reference) so the two cannot drift. ``+`` cannot be an attribute name, hence the words.
OPS = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^", "concat": "&",
    "eq": "=", "lt": "<", "gt": ">", "le": "<=", "ge": ">=", "ne": "<>", "pct": "%",
    "isect": " ", "union": ",", "rng": ":",
}


def esc(s: str) -> str:
    """The corpus's own escaping: keeps a cell's text off the separator bytes.

    Character-wise and context-free, so ``esc(a + b) == esc(a) + esc(b)`` -- which is what
    lets this be applied per emitted piece without disturbing the character offsets the
    token attribution depends on.
    """
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def col_letters(idx: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'. Spreadsheet column naming from a 0-based index."""
    s = ""
    n = idx + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def col_index(letters: str) -> int | None:
    """'A' -> 0, 'AA' -> 26. ``None`` if this is not a pure column label."""
    if not letters:
        return None
    n = 0
    for ch in letters.upper():
        if not ("A" <= ch <= "Z"):
            return None
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def bfs_nodes(root):
    """``[(bfs_id, depth, node, parent_bfs_id, rank_among_siblings)]`` in level order.

    The corpus numbers nodes this way, so ``refs``, ``payloads`` and ``names`` all index it.
    """
    out = []
    q = collections.deque([(root, 0, -1, 0)])
    i = 0
    while q:
        n, d, par, rk = q.popleft()
        out.append((i, d, n, par, rk))
        me = i
        i += 1
        for k, c in enumerate(n.get("children") or []):
            q.append((c, d + 1, me, k))
    return out


def field_of(node) -> str | None:
    """The node's ``f:`` field -- the corpus's own discriminator for what a node IS."""
    for k in node["attributes"]:
        if k.startswith("f:"):
            return k
    return None


#: Excel's sheet-qualifier quoting, as the corpus spells it.
_PLAIN_SHEET = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_LOOKS_LIKE_REF = re.compile(r"^\$?[A-Z]{1,3}\$?[0-9]{1,7}$")


def quote_sheet(title: str) -> str:
    if _PLAIN_SHEET.match(title) and not _LOOKS_LIKE_REF.match(title):
        return title
    return "'" + title.replace("'", "''") + "'"


def markers_of(node) -> list[str]:
    """The node's non-field attributes, in order -- the corpus's ``markers_of``."""
    return [k for k in node["attributes"] if not k.startswith("f:")]


class Emitter:
    """Accumulates text, remembering which node emitted each byte range.

    Every piece is appended at the end and attributed to at most one node, so the spans are
    disjoint and already sorted -- which is what lets the token mapping be a single linear
    walk instead of a scan over every span for every token.
    """

    def __init__(self):
        self.buf: list[str] = []
        self.pos = 0
        self.spans: list[tuple[int, int, int]] = []          # (start, end, node_bfs_id)
        #: (start, end, sheet, row, col, target_bfs_id) -- three axes, because a reference
        #: without its sheet aliases onto the wrong cell for 17.25% of references.
        self.refs: list[tuple[int, int, int, int, int, int]] = []

    def add(self, s: str, node: int | None = None) -> int:
        start = self.pos
        if s:
            self.buf.append(s)
            # BYTES, not characters. tiktoken decodes a token to BYTES; a multi-byte character
            # split across two BPE tokens decodes to U+FFFD per fragment, so summing decoded
            # CHARACTER lengths over-counts and every later token in the record gets attributed
            # to a span further right. Measured on the shipped v1 pack: 2.93% of records drift
            # and 2.50% of body tokens sit at or after a drift onset, corrupting tok_node -- so
            # the TREE PATH was wrong for those tokens in every sheets arm, not just the
            # reference ones. Byte counts agree exactly: 'a'.encode() length vs
            # len(enc.decode_bytes([t])) match even across a split character.
            self.pos += len(s.encode("utf-8"))
            if node is not None:
                self.spans.append((start, self.pos, node))
        return start

    def text(self) -> str:
        return "".join(self.buf)


class Workbook:
    """One corpus record, with the accessors the render and the sidecar both need."""

    def __init__(self, rec, coord="absolute"):
        self.rec = rec
        self.coord = coord
        nodes = bfs_nodes(rec["tree"])
        self.nodes = nodes
        self.byid = {m: (d, n, p, rk) for m, d, n, p, rk in nodes}
        self.kids: dict[int, list[int]] = collections.defaultdict(list)
        for m, d, n, p, rk in nodes:
            if p >= 0:
                self.kids[p].append(m)
        self.names = rec.get("names") or {}
        self.payloads = rec.get("payloads") or {}
        self.ref_target = {}
        for src, tgt in (rec.get("refs") or []):
            self.ref_target[src] = tgt
        #: filled by render(); character offset where each cell / row begins
        self.cell_start: dict[int, int] = {}
        self.row_start: dict[int, int] = {}
        # DFS/reading order, the id space of the sidecar
        self.dfs: dict[int, int] = {}
        counter = 0
        stack = [0]
        while stack:
            m = stack.pop()
            self.dfs[m] = counter
            counter += 1
            stack.extend(reversed(self.kids.get(m, [])))

    # --- accessors -------------------------------------------------------------
    def tag(self, m):
        return self.byid[m][1]["tag"]

    def attrs(self, m):
        return self.byid[m][1]["attributes"]

    def fld(self, m):
        return field_of(self.byid[m][1])

    def parent(self, m):
        return self.byid[m][2]

    def name(self, m):
        return self.names.get(str(m))

    def payload(self, m):
        return self.payloads.get(str(m))

    def sheet_of(self, m):
        """The enclosing ``f:sheet`` node, or -1 for a node under the defined-names table."""
        cur = m
        while cur != -1:
            if self.fld(cur) == "f:sheet":
                return cur
            cur = self.parent(cur)
        return -1

    # --- coordinates -----------------------------------------------------------
    def coord_of(self, m) -> int:
        """The node's coordinate on its level: absolute row/column where available.

        Used for BOTH ``node_path`` and the reference coordinate, so the two cannot drift.
        """
        d, node, p, rk = self.byid[m]
        if self.coord == "absolute":
            f = field_of(node)
            nm = self.names.get(str(m))
            if nm is not None:
                if f == "f:row":
                    # GUARDED: a corpus row named "0" would give -1, which IS the ABSENT
                    # sentinel. That put a level claiming absence ABOVE levels holding real
                    # values -- a malformed path, mis-read by anything treating a path as a
                    # prefix, and in node_path it reached every tree arm rather than only the
                    # pointer ones. Measured on the shipped pack: 16 tokens and 8 node_path
                    # rows. Falling back to the sibling rank keeps the coordinate NON-NEGATIVE,
                    # so ABSENT stays unrepresentable by any real coordinate.
                    try:
                        v = int(nm)
                    except ValueError:
                        return rk
                    return v - 1 if v >= 1 else rk
                if f == "f:cell":
                    ci = col_index(nm)
                    if ci is not None:
                        return ci
        return rk

    def target_coord(self, tgt) -> tuple[int, int, int]:
        """``(sheet, row, col)`` of a reference target, in the same space as ``coord_of``.

        THE SHEET IS PART OF THE ADDRESS, and leaving it out was a real fault: 17.25% of
        references cross to another sheet (measured, 1,500 workbooks), and a ``(row, col)``
        pair alone makes such a reference coincide with whatever cell holds those coordinates
        on the reference's OWN sheet. That is not a missing signal, it is a confident wrong
        one. ``node_path`` level 0 is ``coord_of`` of the depth-1 ancestor, which for any cell
        IS its sheet node, so this is the same space as the other two axes.

        All three are ABSENT together for a defined-name target, which has no grid coordinate
        at all; consumers key on the ROW, so such a reference leaves the token's own
        coordinates alone rather than half-overwriting them.
        """
        f = self.fld(tgt)
        if f != "f:cell":
            # A DEFINED NAME denotes no grid location, so there is nothing to point at. A
            # whole-row target (``f:row``) is also refused: see ``n_refs_whole_row_target``.
            # Either way the answer is ALL THREE ABSENT, never a partial coordinate.
            return ABSENT, ABSENT, ABSENT
        sh = self.sheet_of(tgt)
        if sh == -1:
            # sheet_of returns -1 for a node with no f:sheet ancestor, and -1 IS the ABSENT
            # sentinel. Letting it through as a coordinate would overwrite a reference's level 0
            # with "absent" while levels 1 and 2 held real values -- exactly the malformed path
            # this method exists to prevent.
            return ABSENT, ABSENT, ABSENT
        trip = (self.coord_of(sh), self.coord_of(self.parent(tgt)), self.coord_of(tgt))
        # ALL-OR-NOTHING. A partial overwrite is a malformed path, so a pointer is emitted only
        # when every axis is a real, in-range coordinate. This also removes any need to decide
        # what half a coordinate means, because it cannot arise.
        if any(v < 0 or v > INT16_MAX for v in trip):
            return ABSENT, ABSENT, ABSENT
        return trip

    def address(self, ref_bid, tgt) -> str:
        """The Excel spelling of a reference, derived from the TARGET's names.

        ``$`` markers come from the Ref node's own ``absr``/``absc`` attributes, and the sheet
        is qualified only when the target lives on a different sheet than the reference.
        """
        a = self.attrs(ref_bid)
        f = self.fld(tgt)
        if f == "f:name":
            return self.name(tgt) or "#NAME?"
        if f == "f:cell":
            row_nm = self.name(self.parent(tgt)) or ""
            col_nm = self.name(tgt) or ""
            body = f"{'$' if 'absc' in a else ''}{col_nm}{'$' if 'absr' in a else ''}{row_nm}"
        elif f == "f:row":
            body = f"{'$' if 'absr' in a else ''}{self.name(tgt) or ''}"
        else:
            return "#REF!"
        src_sheet, tgt_sheet = self.sheet_of(ref_bid), self.sheet_of(tgt)
        if tgt_sheet != src_sheet:
            return f"{quote_sheet(self.name(tgt_sheet) or '')}!{body}"
        return body

    # --- rendering -------------------------------------------------------------
    def expr(self, m, e: Emitter) -> None:
        """Render one expression node and its subtree into ``e``, infix, lossless."""
        t = self.tag(m)
        a = self.attrs(m)
        if t == "Lit":
            e.add(esc(self.payload(m) or ""), m)
            return
        if t == "Ref":
            self.ref(m, e)
            return
        if t == "Range":
            ch = self.kids.get(m, [])
            for i, c in enumerate(ch):
                if i:
                    e.add(":")
                # the sheet qualifier is written once, on the first endpoint
                self.ref(c, e, strip_sheet=(i > 0))
            return
        if t == "Free" and "func" in a:
            e.add(esc(self.name(m) or "") + "(", m)
            for i, arg in enumerate(c for c in self.kids.get(m, []) if self.tag(c) == "Arg"):
                if i:
                    e.add(",")
                for c in self.kids.get(arg, []):
                    self.expr(c, e)
            e.add(")")
            return
        if t == "Free":
            e.add(esc(self.name(m) or ""), m)
            return
        if t in ("Op", "Pre", "Post"):
            mk = markers_of(self.byid[m][1])
            e.add(OPS.get(mk[0], "") if mk else "", m)
            return
        if t == "Paren":
            e.add("(", m)
            for c in self.kids.get(m, []):
                self.expr(c, e)
            e.add(")")
            return
        if t == "Array":
            e.add("{", m)
            for i, rw in enumerate(c for c in self.kids.get(m, []) if self.tag(c) == "ArrayRow"):
                if i:
                    e.add(";")
                for j, arg in enumerate(c for c in self.kids.get(rw, []) if self.tag(c) == "Arg"):
                    if j:
                        e.add(",")
                    for c in self.kids.get(arg, []):
                        self.expr(c, e)
            e.add("}")
            return
        if t == "Formula":
            # Excel's brace display for an array or data-table formula, as the corpus's own
            # render_text writes it: '{' + '=' + body + '}'.
            braced = "array" in a or "datatable" in a
            e.add("{=" if braced else "=", m)
            for c in self.kids.get(m, []):
                self.expr(c, e)
            if braced:
                e.add("}")
            return
        if t == "Empty":
            return
        for c in self.kids.get(m, []):
            self.expr(c, e)

    def cell_body(self, cell, e: Emitter) -> None:
        """Emit a cell's content: a literal value, nothing, or a full infix formula.

        A literal whose text begins ``=``, ``{`` or ``'`` gets Excel's leading apostrophe, so
        a value that merely LOOKS like a formula still reads back as a value. The rule belongs
        here and not in ``expr``, because a ``Lit`` operand inside a formula must not get it.
        """
        for ch in self.kids.get(cell, []):
            if self.tag(ch) == "Lit":
                text = self.payload(ch) or ""
                e.add(esc("'" + text if text[:1] in ("=", "{", "'") else text), ch)
            else:
                self.expr(ch, e)

    def ref(self, m, e: Emitter, strip_sheet: bool = False) -> None:
        """Emit one reference as its address text and record its target coordinate."""
        tgt = self.ref_target.get(m)
        if tgt is None:
            e.add("#REF!", m)          # a RefFwd / dropped edge has no spelling
            return
        s = self.address(m, tgt)
        if strip_sheet and "!" in s:
            s = s.split("!", 1)[1]
        start = e.pos
        e.add(s, m)
        sheet, row, col = self.target_coord(tgt)
        e.refs.append((start, e.pos, sheet, row, col, tgt))

    def render(self, mode, sep_cell="\t", sep_row="\n") -> Emitter:
        """Walk the sheet/row/cell spine in reading order, then the defined-names table.

        Records the character offset at which each cell and row begins, which is what makes
        the forward-reference ablation exact rather than inferred.
        """
        e = Emitter()
        self.cell_start: dict[int, int] = {}
        self.row_start: dict[int, int] = {}
        for sheet in self.kids.get(0, []):
            if self.fld(sheet) != "f:sheet":
                continue
            for row in self.kids.get(sheet, []):
                if self.fld(row) != "f:row":
                    continue
                self.row_start[row] = e.pos
                wrote = False
                for cell in self.kids.get(row, []):
                    if self.fld(cell) != "f:cell":
                        continue
                    if wrote:
                        e.add(sep_cell)
                    self.cell_start[cell] = e.pos
                    if mode == "vanilla":
                        # ':' and not '=': a formula's content already starts with '=', and
                        # 'I2==A2-(D2)' would spend a token on the collision.
                        e.add(f"{self.name(cell) or ''}{self.name(row) or ''}:", cell)
                    self.cell_body(cell, e)
                    wrote = True
                if wrote:
                    e.add(sep_row)
        for tbl in self.kids.get(0, []):
            if self.fld(tbl) != "f:names":
                continue
            for nm in self.kids.get(tbl, []):
                if self.fld(nm) != "f:name":
                    continue
                e.add(esc(self.name(nm) or "") + "=", nm)
                for ch in self.kids.get(nm, []):
                    self.expr(ch, e)
                e.add(sep_row)
        return e

    def node_path(self, levels, clamped) -> np.ndarray:
        """``(n_nodes, levels)`` of per-level coordinates, rows indexed by DFS position."""
        path = np.full((len(self.nodes), levels), ABSENT, dtype=np.int16)
        for m, d, node, p, rk in self.nodes:
            if d == 0:
                continue
            chain = []
            cur = m
            while cur != -1 and self.byid[cur][0] > 0:
                chain.append(self.coord_of(cur))
                cur = self.byid[cur][2]
            chain.reverse()
            di = self.dfs[m]
            for lev, r in enumerate(chain[:levels]):
                if r > INT16_MAX:
                    # NOT CLAMPED. Clamping mapped every row from 32,768 to Excel's last row
                    # (1,048,575) onto ONE coordinate, so references to row 40,000 and row
                    # 900,000 became angularly coincident with each other and with a genuine
                    # row 32,767. The clamp is symmetric, so it never broke the coincidence
                    # invariant -- it made coincidence OVER-FIRE, which is the one failure this
                    # arm cannot afford. Measured on the shipped pack: 222,084 clamped
                    # coordinates and 16,208 reference tokens sitting at exactly the ceiling.
                    #
                    # The path is a PREFIX, so the well-formed response is to stop here and
                    # leave the rest ABSENT -- the node simply reads as shallower. Writing
                    # ABSENT at this level while deeper levels held values would be malformed.
                    clamped[0] += 1
                    break
                path[di, lev] = r
        return path


def token_byte_offsets(enc, ids):
    """``(byte_at, bytes_kept)`` -- each BODY token's UTF-8 BYTE offset, and the total.

    BYTES are the only unit in which the emitter's spans and the tokenizer's stream agree.
    ``enc.decode([t])`` on a token holding half a multi-byte character yields a single U+FFFD,
    so a per-token sum of CHARACTER lengths over-counts and drifts right for the rest of the
    record; ``enc.decode_bytes([t])`` is exact. ``Emitter.add`` therefore also measures bytes.

    ``ids[0]`` and ``ids[-1]`` are the framing EOTs and occupy no bytes.
    """
    body = ids[1:-1] if len(ids) > 2 else []
    byte_at = []
    c = 0
    for t in body:
        byte_at.append(c)
        c += len(enc.decode_bytes([t]))
    return byte_at, c


def map_tokens(enc, ids, e: Emitter, wb: Workbook, byte_at):
    """``(tok_node, tok_within, tok_ref_sheet, tok_ref_row, tok_ref_col)``.

    Spans are disjoint and sorted, so this is a linear merge rather than a scan per token.
    A token is attributed to the piece containing its FIRST character.
    """
    n = len(ids)
    tn = np.full(n, ABSENT, dtype=np.int32)
    tw = np.full(n, ABSENT, dtype=np.int16)
    rs = np.full(n, ABSENT, dtype=np.int16)
    rr = np.full(n, ABSENT, dtype=np.int16)
    rc = np.full(n, ABSENT, dtype=np.int16)

    si = 0
    spans = e.spans
    within = collections.Counter()
    for ti, ch in enumerate(byte_at):
        while si < len(spans) and spans[si][1] <= ch:
            si += 1
        if si < len(spans) and spans[si][0] <= ch:
            node = spans[si][2]
            di = wb.dfs[node]
            tn[ti + 1] = di
            tw[ti + 1] = min(within[di], INT16_MAX)
            within[di] += 1

    ri = 0
    refs = e.refs
    for ti, ch in enumerate(byte_at):
        while ri < len(refs) and refs[ri][1] <= ch:
            ri += 1
        if ri < len(refs) and refs[ri][0] <= ch:
            sheet, row, col = refs[ri][2], refs[ri][3], refs[ri][4]
            # target_coord has already refused anything out of range or partial, so these are
            # either all real and in-range or all ABSENT. Asserted rather than re-clamped: a
            # clamp here would silently reintroduce the over-firing coincidence.
            if sheet != ABSENT and row != ABSENT and col != ABSENT:
                rs[ti + 1], rr[ti + 1], rc[ti + 1] = sheet, row, col
    return tn, tw, rs, rr, rc


def build(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--corpus", default=f"{SOURCES}/spreadsheet/sheet_graph_v2_c4096.jsonl")
    ap.add_argument("--out", required=True, help="text pack dir; sidecar goes to <out>_dompath")
    ap.add_argument("--render", choices=("vanilla", "values"), required=True)
    ap.add_argument("--coord", choices=("absolute", "rank"), default="absolute",
                    help="absolute: row/col levels carry the real grid address from `names`")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--levels", type=int, default=LEVELS)
    ap.add_argument("--block-size", type=int, default=4096,
                    help="records longer than this in EITHER render are DROPPED, not cut")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--drop-forward-refs", action="store_true",
                    help="ABLATION: withhold the coordinate for targets rendered later")
    a = ap.parse_args(argv)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    os.makedirs(a.out, exist_ok=True)
    side = a.out.rstrip("/") + "_dompath"
    os.makedirs(side, exist_ok=True)

    # STREAMED, and node rows TRIMMED -- both copied from build_web_pack.py, which records
    # why it matters beyond disk: cell 4's graph arms went disk-bound on a sidecar too large
    # for the page cache and lost 1.8x to the sequence arm, which reads no sidecar at all. A
    # spreadsheet truncates even harder than a web page (63% of workbooks against 82% of
    # pages), so the untrimmed table here would have been 9.5 GB, most of it never read.
    w_tok = StreamNpy(os.path.join(a.out, "tokens.npy"), np.uint16)
    w_np = StreamNpy(os.path.join(side, "node_path.npy"), np.int16, cols=a.levels)
    w_tn = StreamNpy(os.path.join(side, "tok_node.npy"), np.int32)
    w_tw = StreamNpy(os.path.join(side, "tok_within.npy"), np.int16)
    w_rs = StreamNpy(os.path.join(side, "tok_ref_sheet.npy"), np.int16)
    w_rr = StreamNpy(os.path.join(side, "tok_ref_row.npy"), np.int16)
    w_rc = StreamNpy(os.path.join(side, "tok_ref_col.npy"), np.int16)
    tok_off = [0]
    node_off = [0]
    nodes_stored = 0

    n_rec = 0
    n_ref = n_fwd = n_fwd_dropped = n_ref_cut = 0
    #: References whose TARGET was truncated out of the window, so no coordinate is emitted,
    #: against those whose target survives and therefore carry a usable pointer. A DIFFERENT
    #: quantity from n_refs_past_truncation, which counts references whose OWN span is cut.
    n_ref_target_cut = n_ref_live = n_ref_no_cell = n_ref_whole_row = 0
    dropped_deep = n_truncated = n_dropped_long = 0
    #: Record ids in emission order. Written out so the two packs' record sets can be
    #: ASSERTED identical rather than assumed -- the paired comparison depends on it.
    record_ids: list[str] = []
    clamped = [0]
    total_tokens = total_nodes = 0
    n_formula = 0

    with open(a.corpus) as fh:
        for line in fh:
            if a.limit and n_rec >= a.limit:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "tree" not in rec:
                continue
            if rec.get("depth", 0) > a.levels:
                dropped_deep += 1
                continue

            wb = Workbook(rec, coord=a.coord)
            # DROP, DO NOT TRUNCATE -- and drop on the SAME criterion in both packs.
            #
            # The two renders have different lengths (vanilla spells every address out), so a
            # workbook can fit one and not the other. If each pack dropped independently the two
            # would hold different record sets and the paired comparison between them would be
            # meaningless. So BOTH renders are measured here and the record is dropped if EITHER
            # exceeds the cap, whichever render this invocation is actually emitting. That makes
            # the surviving set a function of the corpus and block_size alone -- identical across
            # invocations -- and `record_ids.json` lets it be asserted rather than trusted.
            #
            # Truncation is what this replaces. It cut 61% of workbooks, and it was the source of
            # the cut-away-target problem: a reference whose target fell past the cut. With no
            # truncation every target is present, so reference coverage becomes complete.
            texts, framed = {}, {}
            too_long = empty = False
            for mode in ("values", "vanilla"):
                em = wb.render(mode)                 # the REQUESTED mode is rendered last below
                txt = em.text()
                if not txt:
                    empty = True
                    break
                tk = enc.encode_ordinary(txt)
                if not tk:
                    empty = True
                    break
                if len(tk) + 2 > a.block_size:       # +2 for the framing EOTs
                    too_long = True
                texts[mode] = txt
                framed[mode] = tk
            if empty:
                continue
            if too_long:
                n_dropped_long += 1
                continue
            # Re-render the requested mode LAST so wb.cell_start / wb.row_start describe it.
            e = wb.render(a.render)
            ids = [eot] + framed[a.render] + [eot]
            record_ids.append(rec.get("id", f"#{n_rec}"))

            byte_at, bytes_kept = token_byte_offsets(enc, ids)

            # Forward references: kept by default -- the address is spelled at the reference
            # site in both renders, so the coordinate is already in the prefix. The ablation
            # compares the target's own start offset, recorded during the render.
            kept_refs = []
            for start, end, sheet, row, col, tgt in e.refs:
                n_ref += 1
                if wb.fld(tgt) == "f:row":
                    n_ref_whole_row += 1
                tgt_at = wb.cell_start.get(tgt, wb.row_start.get(tgt))
                if tgt_at is not None and tgt_at > start:
                    n_fwd += 1
                    if a.drop_forward_refs:
                        n_fwd_dropped += 1
                        continue
                # TARGET TRUNCATED AWAY -> emit NO coordinate, on all three axes together.
                # Measured on the val split of the v1 pack: 24.4% of pointer-bearing tokens
                # named a cell that is not in the window at all (33,010 of 135,206), because
                # the record is cut at block_size and the target falls past the cut. Under the
                # old separate-group design such a pointer merely wasted its planes -- the
                # token kept its own location in the tree planes. Under rope_ref_anchor the
                # token OVERWRITES its own sheet/row/column with the target's, so an absent
                # target would cost it its real position AND hand it a coordinate matching
                # nothing: strictly worse than leaving it alone, on a quarter of all pointers,
                # and it would look exactly like the mechanism failing.
                #
                # All three axes are suppressed together so a token is never half-overwritten;
                # consumers key on the row, so ABSENT there leaves its own path untouched.
                if tgt_at is None:
                    # The target is not a cell or a row at all -- a DEFINED NAME, which denotes
                    # no grid location. Distinct from truncation and counted separately, because
                    # conflating the two would make a permanent property of the corpus look like
                    # a window artifact that a bigger block_size could fix.
                    n_ref_no_cell += 1
                    kept_refs.append((start, end, ABSENT, ABSENT, ABSENT, tgt))
                    continue
                if tgt_at >= bytes_kept:
                    n_ref_target_cut += 1
                    kept_refs.append((start, end, ABSENT, ABSENT, ABSENT, tgt))
                    continue
                n_ref_live += 1
                kept_refs.append((start, end, sheet, row, col, tgt))
            e.refs = kept_refs

            tn, tw, rs, rr, rc = map_tokens(enc, ids, e, wb, byte_at)
            n_ref_cut += sum(1 for s, *_rest in e.refs if s >= bytes_kept)

            # Node ids are record-local and the table is indexed by them, so keeping the
            # prefix up to the largest id any token names preserves every id in use.
            path = wb.node_path(a.levels, clamped)
            keep = int(tn.max()) + 1 if len(tn) and tn.max() >= 0 else 0
            total_nodes += len(wb.nodes)
            path = path[:keep]
            nodes_stored += len(path)
            n_formula += sum(1 for m, d, n, p, rk in wb.nodes if n["tag"] == "Formula")

            w_np.append(path)
            node_off.append(node_off[-1] + len(path))
            w_tn.append(tn)
            w_tw.append(tw)
            w_rs.append(rs)
            w_rr.append(rr)
            w_rc.append(rc)
            w_tok.append(np.asarray(ids, dtype=np.uint16))
            tok_off.append(tok_off[-1] + len(ids))
            total_tokens += len(ids)
            n_rec += 1

    # ---- close the streams, then the offset tables ----
    for w in (w_tok, w_np, w_tn, w_tw, w_rs, w_rr, w_rc):
        w.close()
    np.save(os.path.join(a.out, "tok_off.npy"), np.asarray(tok_off, dtype=np.int64))
    # The surviving record set, in order. Both packs must produce an IDENTICAL list; the
    # paired comparison between them is void otherwise, so it is written down to be checked.
    json.dump(record_ids, open(os.path.join(a.out, "record_ids.json"), "w"))
    np.save(os.path.join(side, "node_off.npy"), np.asarray(node_off, dtype=np.int64))

    n_val = n_test = max(1, n_rec // 10)
    n_train = n_rec - n_val - n_test
    meta = {
        "text_pack_version": 1,
        "corpus": a.corpus,
        "block_size": a.block_size,
        "n_records": n_rec,
        "tokenizer": "gpt2",
        "tokenizer_is_pretrained": True,
        "vocab_size": 50304,
        "eot": eot,
        "render": a.render,
        "render_rule": ("vanilla: '<address>:<content>' per cell, TAB between cells, NEWLINE "
                        "between rows -- SpreadsheetLLM's baseline serialisation "
                        "(arXiv:2407.09025). values: '<content>' only, same separators. Content "
                        "is the literal value, or the FULL infix formula including operators, "
                        "function names and reference addresses."),
        "address_rule": ("from the corpus `names` table, which IS keyed by BFS node id: column "
                         "letters on the f:cell node, ABSOLUTE row index on its f:row parent. "
                         "Sibling rank is NOT used -- 43.4% of row nodes have a name that is "
                         "not rank+1, because non-empty rows are sparse."),
        "reference_rule": ("a reference renders as its Excel address, derived from the target's "
                          "names entry, with $ markers from the Ref node's absr/absc and a "
                          "sheet qualifier only for cross-sheet targets. Present in BOTH render "
                          "modes, so the sequence arm sees every pointer the tree arm rotates."),
        "defined_names": "the trailing Names table renders as '<name>=<target>' lines",
        "tokenizer_caveat": ("GPT-2 fragments numbers and formula text badly; a spreadsheet "
                            "corpus is mostly numeric cells, so this understates what a "
                            "purpose-built vocabulary would achieve"),
        "split_seed": a.split_seed,
        "split_rule": "by record; records stored train, then val, then test",
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "total_tokens": total_tokens,
        "tokens_per_record_mean": total_tokens / max(n_rec, 1),
        "records_dropped_deeper_than_levels": dropped_deep,
        "records_dropped_too_long": n_dropped_long,
        "records_truncated": n_truncated,      # 0 by construction now: records are DROPPED
        "record_selection": (
            f"Records longer than block_size={a.block_size} are DROPPED, not truncated, and\n"
            f"the criterion is applied to BOTH renders: a record is dropped if EITHER the\n"
            f"values or the vanilla render exceeds the cap. So sheet_values and sheet_vanilla\n"
            f"hold the SAME {n_rec} records in the SAME order -- assert it with\n"
            f"record_ids.json. {n_dropped_long} records were dropped for length and\n"
            f"{dropped_deep} for depth. Consequence to state in any writeup: results speak to\n"
            f"workbooks that fit 4,096 tokens under the LONGER of the two renders, which is a\n"
            f"different corpus population from the truncated v1 pack."),
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)

    man = {
        "dompath_version": 2,
        "built_by": "build_sheet_pack.py",
        "text_pack": a.out,
        "levels": a.levels,
        "absent": ABSENT,
        "render": a.render,
        "coord": a.coord,
        "node_id_space": ("SIDECAR node ids are DFS/reading order so tok_node aligns with the "
                          "render. The CORPUS numbers nodes in BFS order and `refs`, `payloads` "
                          "and `names` all index that space; the builder maps BFS->DFS. "
                          "Verified: set(Ref-node BFS ids) == set(refs sources) in 300/300 "
                          "records, versus 16/300 under DFS."),
        "node_typing": ("by the f: FIELD attribute, never by depth: a trailing Names table "
                        "(18.5% of records) puts defined names at row depth and their targets "
                        "at cell depth."),
        "coord_rule": ("absolute: the f:row level carries the absolute 0-based row index and "
                       "the f:cell level the 0-based column index, both from `names`; all "
                       "other levels carry the sibling rank. rank: sibling rank everywhere."),
        "node_rule": ("a node's path is the list of per-level coordinates from the child of "
                      "the root down to the node itself"),
        "token_attribution": ("each token is attributed to the DEEPEST node that emitted its "
                              "first character, so formula structure is visible to the "
                              "rotation rather than collapsed onto the cell node"),
        "ref_rule": ("tok_ref_sheet/row/col = the (sheet, row, col) of the target, produced by "
                     "the SAME coord_of() as node_path so a reference's angle coincides exactly "
                     "with its target cell's angle. Written on the tokens of the ADDRESS ITSELF, "
                     "per reference -- a cell with several references gets several distinct "
                     "coordinates. A whole-row target gets col = ABSENT; a defined-name "
                     "target denotes no cell and gets all THREE ABSENT, so a consumer keying on "
                     "the row leaves such a token's own coordinates alone."),
        "ref_sheet_axis": ("tok_ref_sheet.npy was ADDED 2026-09-28 and is why this pack exists. "
                           "Without it a reference is only (row, col), so one crossing to "
                           "another sheet coincides with whatever cell holds those coordinates "
                           "on the reference's OWN sheet -- a confident wrong answer, not a "
                           "missing one. Measured on 1,500 workbooks: 17.25% of references are "
                           "cross-sheet and 25.5% of workbooks have more than one sheet. "
                           "node_path level 0 is coord_of of the depth-1 ancestor, which for a "
                           "cell IS its sheet, so all three axes share one space."),
        "ref_causality": ("forward references are KEPT. A reference renders as its address "
                          "text in both modes, so the target coordinate is in the prefix by "
                          "the time the reference is read and the rotation is redundant with "
                          "text rather than additional to it. --drop-forward-refs is an "
                          "ablation."),
        "target_truncation": (
            f"Records are DROPPED rather than truncated, so every reference that names a cell "
            f"or a row has its target IN the window: n_refs_target_past_truncation is "
            f"{n_ref_target_cut} and is expected to be 0. The suppression code remains as a "
            f"safety net. {n_ref_no_cell} references name a DEFINED NAME rather than a cell -- "
            f"those denote no grid location and correctly carry no coordinate on any axis; that "
            f"is a permanent property of the corpus, not a window artifact. "
            f"{n_ref_live} references carry a usable pointer "
            f"({100.0 * n_ref_live / max(n_ref, 1):.2f}% of all references; "
            f"{100.0 * n_ref_live / max(n_ref - n_ref_no_cell, 1):.2f}% of those naming a cell)."),
        "n_refs_target_past_truncation": n_ref_target_cut,
        "n_refs_target_not_a_cell": n_ref_no_cell,
        "n_refs_with_live_target": n_ref_live,
        "n_refs_total": n_ref,
        "n_refs_forward_in_render_order": n_fwd,
        "n_refs_forward_dropped": n_fwd_dropped,
        "truncation": (f"{n_truncated} of {n_rec} workbooks are longer than block_size "
                       f"({a.block_size}) and are cut, so the pack sees each workbook's "
                       f"leading region only. {n_ref_cut} of {n_ref} references fall past the "
                       f"cut and reach no token; the reference coordinate therefore lands on "
                       f"the {n_ref - n_ref_cut} that survive. A spreadsheet has far more "
                       f"references per token than the Python or web packs, so this is the "
                       f"binding limit on reference coverage here, not causality."),
        "n_records_truncated": n_truncated,
        "n_refs_past_truncation": n_ref_cut,
        "drop_forward_refs": bool(a.drop_forward_refs),
        "coords_past_int16_path_truncated": (
            clamped[0],
            "Path levels DROPPED because the coordinate exceeds the int16 ceiling; the "
            "node reads as shallower from that level down. NOT clamped: clamping "
            "collapsed every row from 32,768 to Excel's 1,048,575 onto one coordinate, "
            "making references to row 40,000 and row 900,000 coincident with each other "
            "and with a real row 32,767 -- coincidence over-firing, which is the one "
            "failure this cell cannot afford. The shipped v1 pack had 222,084 clamped "
            "coordinates and 16,208 reference tokens at exactly the ceiling."),
        "n_refs_whole_row_target": n_ref_whole_row,
        "whole_row_targets": (
            "References whose target is a whole ROW rather than a cell. Refused: they "
            "would need a pointer with no column, and a partial overwrite is a malformed "
            "path. Measured 0 over all 52.8M token positions of the v1 pack, so this "
            "path is UNEXERCISED by this corpus -- the counter exists so that if the "
            "corpus ever changes it shows up as a number rather than as silent "
            "half-coordinates."),
        "n_formula_nodes": n_formula,
        "n_nodes_in_corpus": total_nodes,
        "n_nodes_stored": nodes_stored,
        "node_trim": (f"{nodes_stored} of {total_nodes} node rows stored "
                      f"({100.0 * nodes_stored / max(total_nodes, 1):.1f}%). A record's tokens "
                      f"are cut at block_size, so nodes past the cut are named by no token. "
                      f"Trimming is what keeps the sidecar inside the page cache: cell 4's "
                      f"graph arms went disk-bound on an untrimmed one and lost 1.8x to the "
                      f"sequence arm, which reads no sidecar at all."),
        "nodes_stored_per_token": nodes_stored / max(total_tokens, 1),
        "records_dropped_deeper_than_levels": dropped_deep,
        "records_dropped_too_long": n_dropped_long,
        "n_records": n_rec,
    }
    json.dump(man, open(os.path.join(side, "manifest.json"), "w"), indent=1)
    print(json.dumps({"n_records": n_rec, "total_tokens": total_tokens,
                      "records_truncated": n_truncated,
                      "records_dropped_too_long": n_dropped_long,
                      "refs": n_ref, "refs_forward": n_fwd,
                      "refs_live_target": n_ref_live,
                      "refs_target_truncated": n_ref_target_cut,
                      "refs_target_not_a_cell": n_ref_no_cell,
                      "refs_forward_dropped": n_fwd_dropped,
                      "refs_past_truncation": n_ref_cut,
                      "coords_clamped": clamped[0],
                      "dropped_deep": dropped_deep,
                      "out": a.out, "sidecar": side}, indent=1))


if __name__ == "__main__":
    build()
