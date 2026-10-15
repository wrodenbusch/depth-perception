"""The sheet builder, checked against the corpus's OWN decoder as an oracle.

``docs/sheet_graph_corpus_ref.py.reference`` is the builder that produced the corpus, and it
carries the inverse direction too: ``record_to_decoded`` turns a record back into per-cell
``(row, col, kind, text)`` and ``render_text`` spells each cell's body. That is ground truth
for the formula language, so these tests do not re-state the grammar -- they diff our render
against theirs, cell by cell.

Why this file exists: an earlier version of the builder read a node's text from ``payloads``
alone, which exists only on ``Lit`` nodes, so every operator, function name and reference
vanished and ``=A1-(B1+C1)`` rendered as the empty string. Nothing caught it, because nothing
compared the render to anything.
"""
from __future__ import annotations
from topos_paths import SOURCES  # noqa: E402

import importlib.machinery
import importlib.util
import json
import os

import numpy as np
import pytest

import build_sheet_pack as B

CORPUS = f"{SOURCES}/spreadsheet/sheet_graph_v2_c4096.jsonl"
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "docs", "sheet_graph_corpus_ref.py.reference")
N_RECORDS = 120


def _load_oracle():
    # the ``.reference`` suffix is deliberate -- it keeps the file out of the import path, so
    # it needs an explicit source loader rather than extension sniffing
    loader = importlib.machinery.SourceFileLoader("sheet_graph_corpus_ref", REF)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def oracle():
    if not os.path.exists(REF):
        pytest.skip("corpus reference builder not present")
    return _load_oracle()


@pytest.fixture(scope="module")
def records():
    if not os.path.exists(CORPUS):
        pytest.skip("spreadsheet corpus not present")
    out = []
    with open(CORPUS) as fh:
        for line in fh:
            rec = json.loads(line)
            if "tree" in rec:
                out.append(rec)
            if len(out) >= N_RECORDS:
                break
    return out


def _oracle_bodies(oracle, rec):
    """``{(row_int, col_letters): body}`` exactly as the corpus's render_text would write it."""
    dec = oracle.record_to_decoded(rec)
    out = {}
    for _title, cells in dec.sheets:
        for r, col, kind, text in cells:
            if kind == "value":
                body = oracle._esc(("'" + text) if text[:1] in ("=", "{", "'") else text)
            elif kind == "empty":
                body = ""
            elif kind == "formula":
                body = oracle._esc(text)
            else:
                body = "{" + oracle._esc(text) + "}"
            out[(r, col)] = body
    return out


def _our_bodies(rec):
    """``{(row_int, col_letters): body}`` from our renderer, per cell."""
    wb = B.Workbook(rec, coord="absolute")
    out = {}
    for sheet in wb.kids.get(0, []):
        if wb.fld(sheet) != "f:sheet":
            continue
        for row in wb.kids.get(sheet, []):
            if wb.fld(row) != "f:row":
                continue
            for cell in wb.kids.get(row, []):
                if wb.fld(cell) != "f:cell":
                    continue
                e = B.Emitter()
                wb.cell_body(cell, e)
                out[(int(wb.name(row)), wb.name(cell))] = e.text()
    return out


# --------------------------------------------------------------------------- render fidelity


def test_every_cell_matches_the_corpus_own_decoder(oracle, records):
    """Our rendered body equals the corpus's, for every cell of every record."""
    checked = mismatched = 0
    first = []
    for rec in records:
        try:
            want = _oracle_bodies(oracle, rec)
        except Exception:
            continue                      # the oracle refuses some records; skip, do not hide
        got = _our_bodies(rec)
        assert set(want) == set(got), "different cell sets"
        for k in want:
            checked += 1
            if want[k] != got[k]:
                mismatched += 1
                if len(first) < 5:
                    first.append((k, want[k], got[k]))
    assert checked > 5_000, f"only {checked} cells checked -- corpus sample too small"
    assert mismatched == 0, f"{mismatched}/{checked} cells differ; first: {first}"


def test_formulas_are_not_empty(records):
    """The exact regression that motivated the rewrite: formula cells rendered to ''."""
    n_formula = n_empty = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        for m, d, node, p, rk in wb.nodes:
            if node["tag"] != "Formula":
                continue
            n_formula += 1
            e = B.Emitter()
            wb.expr(m, e)
            if e.text().strip() in ("", "="):
                n_empty += 1
    assert n_formula > 500, f"only {n_formula} formulas in the sample"
    assert n_empty == 0, f"{n_empty}/{n_formula} formulas rendered empty"


def test_operator_table_matches_the_corpus(oracle):
    """Every operator marker the corpus can emit has a spelling here."""
    assert B.OPS == oracle.MARKER_OP


# ------------------------------------------------------------------- addresses & coordinates


def test_addresses_come_from_names_not_sibling_rank(records):
    """Row names are absolute and rows are sparse, so rank-derived addresses are wrong.

    This asserts the bug is gone by showing the two rules genuinely disagree on this corpus
    and that we follow the names table.
    """
    disagree = total = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        for m, d, node, p, rk in wb.nodes:
            if wb.fld(m) != "f:row":
                continue
            total += 1
            if int(wb.name(m)) != rk + 1:
                disagree += 1
    assert total > 1_000
    assert disagree > total * 0.2, (
        f"only {disagree}/{total} rows are sparse -- the fixture may have changed")


def test_reference_coordinate_equals_its_targets_own_path(records):
    """The mechanism under test: a reference's angle must coincide with its target's.

    Both sides come from ``coord_of``, so this is really a guard that the two call sites
    cannot drift apart -- which is exactly how a previous version broke.
    """
    ok = bad = 0
    clamped = [0]
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        e = wb.render("values")
        path = wb.node_path(B.LEVELS, clamped)
        for _s, _en, sheet, row, col, tgt in e.refs:
            if col == B.ABSENT:
                continue                  # whole-row target: no column to compare
            di = wb.dfs[tgt]
            if (int(path[di, 0]) == sheet and int(path[di, 1]) == row
                    and int(path[di, 2]) == col):
                ok += 1
            else:
                bad += 1
    assert ok > 5_000, f"only {ok} cell references checked"
    assert bad == 0, f"{bad} references point at a different angle than their target"


def test_rank_coord_mode_also_agrees_with_itself(records):
    """The ablation must keep the same coincidence property, in its own space."""
    bad = 0
    clamped = [0]
    for rec in records[:30]:
        wb = B.Workbook(rec, coord="rank")
        e = wb.render("values")
        path = wb.node_path(B.LEVELS, clamped)
        for _s, _en, sheet, row, col, tgt in e.refs:
            if col == B.ABSENT:
                continue
            di = wb.dfs[tgt]
            if not (int(path[di, 0]) == sheet and int(path[di, 1]) == row
                    and int(path[di, 2]) == col):
                bad += 1
    assert bad == 0, f"{bad} references disagree under --coord rank"


def test_absolute_and_rank_actually_differ(records):
    """If the two coord modes produced the same numbers the ablation would be vacuous."""
    diff = 0
    clamped = [0]
    for rec in records[:30]:
        a = B.Workbook(rec, coord="absolute").node_path(B.LEVELS, clamped)
        r = B.Workbook(rec, coord="rank").node_path(B.LEVELS, clamped)
        diff += int((a != r).sum())
    assert diff > 0, "absolute and rank coords are identical on this sample"


def test_column_letters_round_trip():
    for i in (0, 1, 25, 26, 27, 51, 52, 701, 702, 16383):
        assert B.col_index(B.col_letters(i)) == i
    assert B.col_index("") is None
    assert B.col_index("A1") is None


# ------------------------------------------------------------------------- node typing


def test_defined_names_are_not_rendered_as_data(records):
    """A trailing Names table sits at row/cell DEPTH; typing by depth renders it as a sheet."""
    with_names = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        tables = [m for m in wb.kids.get(0, []) if wb.fld(m) == "f:names"]
        if not tables:
            continue
        with_names += 1
        for tbl in tables:
            # every child of the names table is a defined name, never a row -- and it sits at
            # the SAME depth a row does, which is why typing by depth was wrong
            for nm in wb.kids.get(tbl, []):
                assert wb.fld(nm) == "f:name"
                assert wb.byid[nm][0] == 2
    assert with_names > 0, "no record in the sample has a defined-names table"


# ------------------------------------------------------------------------- pack invariants


def test_pack_arrays_are_consistent(tmp_path, records):
    """End to end on a small slice: every array lines up and references land on addresses."""
    out = str(tmp_path / "pack")
    B.build(["--corpus", CORPUS, "--out", out, "--render", "values", "--limit", "40",
             "--block-size", "4096"])
    side = out + "_dompath"
    tokens = np.load(os.path.join(out, "tokens.npy"))
    tok_off = np.load(os.path.join(out, "tok_off.npy"))
    node_path = np.load(os.path.join(side, "node_path.npy"))
    node_off = np.load(os.path.join(side, "node_off.npy"))
    tok_node = np.load(os.path.join(side, "tok_node.npy"))
    rr = np.load(os.path.join(side, "tok_ref_row.npy"))
    rc = np.load(os.path.join(side, "tok_ref_col.npy"))

    assert len(tokens) == tok_off[-1]
    assert len(tok_node) == len(tokens) == len(rr) == len(rc)
    assert node_path.shape[0] == node_off[-1]
    assert node_path.shape[1] == B.LEVELS
    assert len(tok_off) == len(node_off)

    # a token's node id is record-local and must be in range for its record
    for i in range(len(tok_off) - 1):
        seg = tok_node[tok_off[i]:tok_off[i + 1]]
        n_nodes = node_off[i + 1] - node_off[i]
        assert seg.max() < n_nodes, f"record {i}: node id past its node table"
        assert seg.min() >= B.ABSENT

    assert (rr != B.ABSENT).sum() > 0, "no token carries a reference coordinate"
    # a column coordinate never appears without a row coordinate
    assert not ((rc != B.ABSENT) & (rr == B.ABSENT)).any()


def test_forward_refs_are_kept_by_default(tmp_path, records):
    """Forward references exist in render order and are kept, because the address is text."""
    keep = str(tmp_path / "keep")
    drop = str(tmp_path / "drop")
    B.build(["--corpus", CORPUS, "--out", keep, "--render", "values", "--limit", "40",
             "--block-size", "4096"])
    B.build(["--corpus", CORPUS, "--out", drop, "--render", "values", "--limit", "40",
             "--block-size", "4096", "--drop-forward-refs"])
    mk = json.load(open(os.path.join(keep + "_dompath", "manifest.json")))
    md = json.load(open(os.path.join(drop + "_dompath", "manifest.json")))
    assert mk["n_refs_forward_in_render_order"] > 0, "no forward refs in the sample"
    assert mk["n_refs_forward_dropped"] == 0
    assert md["n_refs_forward_dropped"] == md["n_refs_forward_in_render_order"]
    kept = np.load(os.path.join(keep + "_dompath", "tok_ref_row.npy"))
    dropped = np.load(os.path.join(drop + "_dompath", "tok_ref_row.npy"))
    assert (kept != B.ABSENT).sum() > (dropped != B.ABSENT).sum()


def test_both_renders_spell_references(tmp_path):
    """The fairness requirement: the sequence arm must see every pointer the tree arm rotates."""
    for mode in ("vanilla", "values"):
        out = str(tmp_path / mode)
        B.build(["--corpus", CORPUS, "--out", out, "--render", mode, "--limit", "40",
                 "--block-size", "4096"])
        man = json.load(open(os.path.join(out + "_dompath", "manifest.json")))
        assert man["n_refs_total"] > 0


# ------------------------------------------------------- the SHEET axis (added 2026-09-28)


def test_cross_sheet_references_exist_in_this_corpus(records):
    """If they did not, the sheet axis would be pointless and these tests vacuous."""
    cross = same = multi = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        if (rec["meta"].get("n_sheets") or 1) > 1:
            multi += 1
        for src, tgt in (rec.get("refs") or []):
            if wb.sheet_of(src) == wb.sheet_of(tgt):
                same += 1
            else:
                cross += 1
    assert multi > 0, "no multi-sheet workbook in the sample"
    assert cross > 100, f"only {cross} cross-sheet references; the sheet axis needs exercising"
    assert cross / (cross + same) > 0.05, "cross-sheet references too rare to matter"


def test_three_axes_identify_a_cross_sheet_target_uniquely(records):
    """FAULT 1 of rope_ref_path, on real data.

    With (row, col) only, a cross-sheet reference matches any cell holding those coordinates on
    ANY sheet -- including the reference's own. Asserts that the (sheet, row, col) triple picks
    out exactly one cell where the pair does not, and that the triple names the real target.
    """
    ambiguous_pairs = checked = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        # every cell in the workbook, keyed both ways
        by_triple: dict = {}
        by_pair: dict = {}
        for m, d, node, p, rk in wb.nodes:
            if wb.fld(m) != "f:cell":
                continue
            sh = wb.sheet_of(m)
            trip = (wb.coord_of(sh), wb.coord_of(wb.parent(m)), wb.coord_of(m))
            by_triple.setdefault(trip, []).append(m)
            by_pair.setdefault(trip[1:], []).append(m)
        for src, tgt in (rec.get("refs") or []):
            if wb.fld(tgt) != "f:cell":
                continue
            sheet, row, col = wb.target_coord(tgt)
            trip = (sheet, row, col)
            assert by_triple.get(trip), f"target {trip} names no cell -- coordinate spaces differ"
            assert tgt in by_triple[trip], "the triple does not name the actual target cell"
            assert len(by_triple[trip]) == 1, f"the triple {trip} names {len(by_triple[trip])} cells"
            checked += 1
            if len(by_pair.get(trip[1:], [])) > 1:
                ambiguous_pairs += 1
    assert checked > 5_000, f"only {checked} cell references checked"
    assert ambiguous_pairs > 0, (
        "no reference in the sample had an ambiguous (row, col) -- then this corpus could not "
        "show the aliasing the sheet axis fixes, and the fixture has changed"
    )


def test_a_defined_name_target_reports_no_coordinate(records):
    """A defined name denotes no cell, so all three axes must be ABSENT together.

    Half-overwriting is the danger: a consumer keys on the row, so a partial coordinate would
    let a reference take a target's row while keeping its own column.
    """
    seen = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        for src, tgt in (rec.get("refs") or []):
            if wb.fld(tgt) == "f:name":
                seen += 1
                assert wb.target_coord(tgt) == (B.ABSENT, B.ABSENT, B.ABSENT), (
                    "a defined-name target produced a grid coordinate")
    if seen == 0:
        pytest.skip("no defined-name reference target in this sample")


def test_whole_row_target_has_a_sheet_and_a_row_but_no_column(records):
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        for src, tgt in (rec.get("refs") or []):
            if wb.fld(tgt) != "f:row":
                continue
            sheet, row, col = wb.target_coord(tgt)
            assert col == B.ABSENT
            assert row != B.ABSENT and sheet != B.ABSENT
def test_sheet_array_is_written_and_aligned(tmp_path):
    out = str(tmp_path / "pack")
    B.build(["--corpus", CORPUS, "--out", out, "--render", "values", "--limit", "40",
             "--block-size", "4096"])
    side = out + "_dompath"
    assert os.path.exists(os.path.join(side, "tok_ref_sheet.npy"))
    n = len(np.load(os.path.join(out, "tokens.npy")))
    for f in ("tok_ref_sheet.npy", "tok_ref_row.npy", "tok_ref_col.npy"):
        assert len(np.load(os.path.join(side, f))) == n, f"{f} is not aligned to the tokens"


# -------------------------------------------------------------------- the loader's two widths


def test_loader_serves_three_axes_when_the_pack_has_a_sheet(tmp_path):
    """``rope_ref_anchor`` needs (sheet, row, col); the loader must hand it three axes."""
    import torch
    from topos_data import PackedWindows
    out = str(tmp_path / "pack")
    B.build(["--corpus", CORPUS, "--out", out, "--render", "values", "--limit", "60",
             "--block-size", "4096"])
    pw = PackedWindows(out, need_tree=False, sidecar="path", need_ref=True, need_ref_sheet=True)
    assert pw.has_ref_sheet
    g = torch.Generator().manual_seed(0)
    groups, _ = pw.step_batches("train", 8, 8192, generator=g)
    _x, _y, pos = pw.get_batch_records(groups[0], torch.device("cpu"))
    assert "ref" in pos
    assert pos["ref"].shape[-1] == 3, f"served {pos['ref'].shape[-1]} axes, need 3"
    assert pos["ref"].dtype == torch.long
    # sheet and row are ABSENT together -- never a partial coordinate reaching the model
    r = pos["ref"]
    assert not ((r[..., 0] != B.ABSENT) ^ (r[..., 1] != B.ABSENT)).any()


def test_loader_refuses_a_pack_with_no_sheet_axis(tmp_path):
    """A pack built before 2026-09-28 must be refused, not silently used without the sheet."""
    from topos_data import PackedWindows
    out = str(tmp_path / "pack")
    B.build(["--corpus", CORPUS, "--out", out, "--render", "values", "--limit", "40",
             "--block-size", "4096"])
    os.remove(os.path.join(out + "_dompath", "tok_ref_sheet.npy"))
    # without the sheet the pointer arm still loads, two axes wide
    pw = PackedWindows(out, need_tree=False, sidecar="path", need_ref=True)
    assert not pw.has_ref_sheet
    with pytest.raises(FileNotFoundError, match="tok_ref_sheet"):
        PackedWindows(out, need_tree=False, sidecar="path", need_ref=True, need_ref_sheet=True)


def test_the_anchor_arm_reaches_the_model_from_a_real_pack(tmp_path):
    """End to end on a built pack: a batch's pointer drives rope_ref_anchor without raising,
    and a reference genuinely lands on its target's angle in that batch."""
    import torch
    from topos_data import PackedWindows
    from rope_pos import GraphRoPE, REF_ANCHOR_MODE
    out = str(tmp_path / "pack")
    B.build(["--corpus", CORPUS, "--out", out, "--render", "values", "--limit", "80",
             "--block-size", "4096"])
    pw = PackedWindows(out, need_tree=False, sidecar="path", need_ref=True, need_ref_sheet=True)
    g = torch.Generator().manual_seed(3)
    groups, _ = pw.step_batches("train", 16, 8192, generator=g)
    _x, _y, pos = pw.get_batch_records(groups[0], torch.device("cpu"))
    m = GraphRoPE(128, REF_ANCHOR_MODE, seq_planes=16, levels=22, planes_per_level=2,
                  token_planes=4)
    t = pos["path"].shape[1]
    a = m.angles(t, pos["path"], "cpu", ref=pos["ref"])
    assert a.shape == (pos["path"].shape[0], t, 64)
    HEAD = slice(16, 16 + 6)          # the sheet/row/col planes
    path, ref = pos["path"], pos["ref"]
    checked = 0
    for bi in range(path.shape[0]):
        own = {}
        no_ref = ref[bi, :, 1] < 0
        for ti in torch.nonzero(no_ref).flatten().tolist():
            key = tuple(int(v) for v in path[bi, ti, :3])
            if all(v >= 0 for v in key):
                own.setdefault(key, ti)
        for ti in torch.nonzero(~no_ref).flatten().tolist():
            key = tuple(int(v) for v in ref[bi, ti])
            tgt = own.get(key)
            if tgt is None:
                continue              # target not in this record's retained window
            assert torch.equal(a[bi, ti, HEAD], a[bi, tgt, HEAD]), (
                f"record {bi}: reference at {ti} -> {key} does not match the token at {tgt}"
            )
            checked += 1
    assert checked > 20, f"only {checked} resolvable pairs in the batch -- too few to verify"


# ------------------------------------------------- OFFSETS ARE BYTES, NOT CHARACTERS (bug fix)


def test_token_offsets_are_bytes_and_agree_with_the_emitter():
    """THE DRIFT BUG. Spans were measured in characters, token offsets in decoded characters.

    ``enc.decode([t])`` on a token holding half a multi-byte character yields one U+FFFD, so a
    per-token sum of CHARACTER lengths OVER-counts and every later token in the record is
    attributed to a span further right. Measured on the shipped v1 pack: 2.93% of records drift
    and 2.50% of body tokens sit at or after a drift onset -- corrupting ``tok_node``, so the
    TREE PATH was wrong for those tokens in every sheets arm, not only the reference ones.

    Neither existing test could catch it: the oracle compares rendered TEXT, and the coordinate
    test compares ``coord_of`` outputs. Both are blind to offsets.
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    for s in ("日本語", "αβγδε", "→←", "naïve café", "Ωμέγα 42", "plain ascii"):
        ids = enc.encode_ordinary(s)
        by_char = sum(len(enc.decode([t])) for t in ids)
        by_byte = sum(len(enc.decode_bytes([t])) for t in ids)
        assert by_byte == len(s.encode("utf-8")), f"byte accounting is wrong for {s!r}"
        if by_char != len(s):
            # this string is one the old character accounting got wrong -- the regression case
            assert by_byte != by_char, f"{s!r} should expose the drift"


def test_emitter_measures_bytes():
    e = B.Emitter()
    e.add("日本語", 1)
    assert e.pos == len("日本語".encode("utf-8")) == 9, (
        "Emitter.add is counting characters again; spans will drift against the token stream"
    )
    assert e.spans[0] == (0, 9, 1)


def test_multibyte_record_has_no_offset_drift(tmp_path):
    """End to end on a synthetic multi-byte workbook: every token's byte offset must fall inside
    the span it was attributed to, with no accumulating error."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # a workbook whose cells are deliberately CJK/Greek, which is where the drift appeared
    def cell(col, text):
        return {"tag": "Free", "attributes": {"f:cell": ""},
                "children": [{"tag": "Lit", "attributes": {"str": "", "f:value": ""},
                              "children": []}], "_col": col, "_text": text}
    payloads, names = {}, {}
    cells = []
    words = ["日本語のセル", "αβγδεζη", "→←↑↓", "ascii", "混合 mixed 文字"]
    # bfs ids: 0 root, 1 sheet, 2 row, then cells, then their Lit children
    names["1"] = "Sheet1"; names["2"] = "1"
    kids = []
    for i, w in enumerate(words):
        names[str(3 + i)] = B.col_letters(i)
        kids.append({"tag": "Free", "attributes": {"f:cell": ""},
                     "children": [{"tag": "Lit", "attributes": {"str": "", "f:value": ""},
                                   "children": []}]})
    for i, w in enumerate(words):
        payloads[str(3 + len(words) + i)] = w
    rec = {"tree": {"tag": "Workbook", "attributes": {}, "children": [
               {"tag": "Free", "attributes": {"f:sheet": ""}, "children": [
                   {"tag": "Free", "attributes": {"f:row": ""}, "children": kids}]}]},
           "payloads": payloads, "names": names, "refs": [], "depth": 4, "id": "synthetic"}
    wb = B.Workbook(rec, coord="absolute")
    e = wb.render("values")
    text = e.text()
    assert any(ord(c) > 127 for c in text), "the fixture lost its multi-byte content"
    ids = [enc.eot_token] + enc.encode_ordinary(text) + [enc.eot_token]
    byte_at, bytes_kept = B.token_byte_offsets(enc, ids)
    assert bytes_kept == len(text.encode("utf-8")), (
        f"token byte offsets sum to {bytes_kept} but the text is "
        f"{len(text.encode('utf-8'))} bytes -- the offsets drift"
    )
    # and each attributed span really does contain its token's first byte
    tn, tw, rs, rr, rc = B.map_tokens(enc, ids, e, wb, byte_at)
    raw = text.encode("utf-8")
    for ti, off in enumerate(byte_at):
        node = tn[ti + 1]
        if node == B.ABSENT:
            continue
        hits = [(s0, s1) for s0, s1, nd in e.spans if wb.dfs[nd] == node and s0 <= off < s1]
        assert hits, (
            f"token {ti} at byte {off} was attributed to node {node} but falls in none of that "
            f"node's spans -- offsets and spans are in different units"
        )
        assert raw[off:off + 1], "byte offset past the end of the text"


# ------------------------------------------- DROP, NOT TRUNCATE -- and one shared record set


def test_both_renders_keep_the_same_record_set(tmp_path):
    """THE INVARIANT THE PAIRED COMPARISON RESTS ON.

    ``vanilla`` spells every address out, so it is longer and a workbook can fit one render and
    not the other. If each pack dropped independently the two would diverge and comparing them
    would be meaningless. The drop criterion is therefore evaluated on BOTH renders regardless of
    which one is being emitted.
    """
    outs = {}
    for mode in ("values", "vanilla"):
        out = str(tmp_path / mode)
        B.build(["--corpus", CORPUS, "--out", out, "--render", mode, "--limit", "120",
                 "--block-size", "4096"])
        outs[mode] = out
    ids_v = json.load(open(os.path.join(outs["values"], "record_ids.json")))
    ids_a = json.load(open(os.path.join(outs["vanilla"], "record_ids.json")))
    assert ids_v == ids_a, (
        "the two renders kept DIFFERENT record sets, so no comparison between them is valid"
    )
    assert len(ids_v) > 20, f"only {len(ids_v)} records survived -- fixture too small to test"
    off_v = np.load(os.path.join(outs["values"], "tok_off.npy"))
    off_a = np.load(os.path.join(outs["vanilla"], "tok_off.npy"))
    assert len(off_v) == len(off_a) == len(ids_v) + 1
    # vanilla is the longer render, which is exactly why it drives some of the drops
    assert off_a[-1] > off_v[-1]


def test_no_record_is_truncated(tmp_path):
    """Dropping replaces truncation outright, so the truncation counter must stay at zero."""
    out = str(tmp_path / "pack")
    B.build(["--corpus", CORPUS, "--out", out, "--render", "values", "--limit", "120",
             "--block-size", "4096"])
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["records_truncated"] == 0, "a record was cut; dropping is not in force"
    assert meta["records_dropped_too_long"] > 0, (
        "nothing was dropped for length at block_size 4096 over 200 records -- the fixture or "
        "the cap has changed and this test is no longer exercising the path"
    )
    # every stored record is within the cap, framing tokens included
    off = np.load(os.path.join(out, "tok_off.npy"))
    assert int(np.diff(off).max()) <= 4096


def test_every_cell_reference_has_its_target_in_the_window(tmp_path):
    """With no truncation, coverage must be COMPLETE for references that name a cell.

    The residue is references whose target is a DEFINED NAME, which denotes no grid location at
    all -- a permanent property of the corpus, counted separately so it cannot be mistaken for a
    window artifact.
    """
    out = str(tmp_path / "pack")
    B.build(["--corpus", CORPUS, "--out", out, "--render", "values", "--limit", "120",
             "--block-size", "4096"])
    man = json.load(open(os.path.join(out + "_dompath", "manifest.json")))
    assert man["n_refs_target_past_truncation"] == 0, (
        f"{man['n_refs_target_past_truncation']} references have a target outside the window, "
        "but records are no longer truncated -- something else is dropping cells"
    )
    cell_refs = man["n_refs_total"] - man["n_refs_target_not_a_cell"]
    assert man["n_refs_with_live_target"] == cell_refs, (
        "not every cell-naming reference has a live target"
    )
    assert man["n_refs_target_not_a_cell"] > 0, "no defined-name target in the sample"


# ------------------------------------------ ABSENT MUST BE UNREPRESENTABLE BY A REAL COORDINATE


def test_coord_of_never_returns_a_negative_for_a_row_or_cell():
    """A row named "0" gave ``int("0") - 1 == -1``, which IS the ABSENT sentinel.

    That put a level claiming absence ABOVE levels holding real values -- a malformed path,
    mis-read by anything treating a path as a prefix, and it lived in ``node_path`` so it reached
    every tree arm rather than only the pointer ones. Measured on the shipped pack: 16 tokens and
    8 node_path rows.
    """
    rec = {"tree": {"tag": "Workbook", "attributes": {}, "children": [
               {"tag": "Free", "attributes": {"f:sheet": ""}, "children": [
                   {"tag": "Free", "attributes": {"f:row": ""}, "children": [
                       {"tag": "Free", "attributes": {"f:cell": ""}, "children": [
                           {"tag": "Lit", "attributes": {"str": "", "f:value": ""},
                            "children": []}]}]}]}]},
           "payloads": {"4": "x"},
           "names": {"1": "Sheet1", "2": "0", "3": "A"},   # row named "0" -- the pathological case
           "refs": [], "depth": 4, "id": "row-zero"}
    wb = B.Workbook(rec, coord="absolute")
    for m, d, node, p, rk in wb.nodes:
        if wb.fld(m) in ("f:row", "f:cell"):
            v = wb.coord_of(m)
            assert v >= 0, f"coord_of returned {v} for {wb.fld(m)} -- collides with ABSENT"
    clamped = [0]
    path = wb.node_path(B.LEVELS, clamped)
    # no row may claim absence at a level with a present level beneath it
    for r in range(path.shape[0]):
        seen_absent = False
        for lev in range(path.shape[1]):
            if path[r, lev] == B.ABSENT:
                seen_absent = True
            elif seen_absent:
                raise AssertionError(
                    f"node_path row {r} has a present level {lev} below an ABSENT one: "
                    f"{path[r][:6].tolist()} -- malformed prefix"
                )


def test_no_malformed_prefix_anywhere_in_a_real_pack(records):
    """The same prefix invariant, over real corpus records rather than a fixture."""
    clamped = [0]
    for rec in records[:60]:
        wb = B.Workbook(rec, coord="absolute")
        path = wb.node_path(B.LEVELS, clamped)
        arr = np.asarray(path)
        present = arr >= 0
        # once absent, always absent along a row: cumulative-min of present must equal present
        mono = np.minimum.accumulate(present, axis=1)
        assert (mono == present).all(), "some node_path row has a present level below an absent one"


def test_a_defined_name_target_yields_no_pointer_not_a_minus_one_sheet(records):
    """``sheet_of`` returns -1 for anything under the defined-names table, and -1 IS ABSENT.

    A defined name denotes no grid location, so the chosen behaviour is NO POINTER at all --
    all three axes ABSENT -- rather than a sheet axis of -1 beside real row and column values,
    which would be a malformed path under the overwriting arm. 18.5% of records carry such a
    table, so this is not an edge case.
    """
    seen = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        for src, tgt in (rec.get("refs") or []):
            if wb.fld(tgt) == "f:cell":
                continue
            seen += 1
            trip = wb.target_coord(tgt)
            assert trip == (B.ABSENT, B.ABSENT, B.ABSENT), (
                f"a non-cell target ({wb.fld(tgt)}) produced {trip}; -1 must never flow through "
                "as if it were a coordinate"
            )
    if seen == 0:
        pytest.skip("no non-cell reference target in this sample")


def test_a_pointer_is_all_three_axes_or_none(records):
    """``target_coord`` must never return a partial coordinate."""
    for rec in records[:60]:
        wb = B.Workbook(rec, coord="absolute")
        for src, tgt in (rec.get("refs") or []):
            trip = wb.target_coord(tgt)
            n_absent = sum(1 for v in trip if v == B.ABSENT)
            assert n_absent in (0, 3), f"partial coordinate {trip} for a {wb.fld(tgt)} target"


def test_out_of_range_rows_are_dropped_not_clamped(records):
    """Clamping collapsed every row from 32,768 to Excel's 1,048,575 onto ONE coordinate.

    References to row 40,000 and row 900,000 became coincident with each other and with a real
    row 32,767 -- coincidence OVER-firing, the one failure this cell cannot afford. Nothing may
    now sit at exactly the ceiling.
    """
    clamped = [0]
    ceiling_hits = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        path = wb.node_path(B.LEVELS, clamped)
        ceiling_hits += int((np.asarray(path) == B.INT16_MAX).sum())
        for src, tgt in (rec.get("refs") or []):
            for v in wb.target_coord(tgt):
                assert v != B.INT16_MAX, "a pointer axis sits at the int16 ceiling -- clamped"
    assert ceiling_hits == 0, (
        f"{ceiling_hits} node_path entries sit at exactly {B.INT16_MAX}; clamping is back"
    )


def test_columns_are_inside_int16_by_construction(records):
    """Excel's column limit is 16,383, so only the ROW axis can exceed int16. Asserted, not
    assumed -- the whole no-clamp argument rests on it."""
    worst = 0
    for rec in records:
        wb = B.Workbook(rec, coord="absolute")
        for m, d, node, p, rk in wb.nodes:
            if wb.fld(m) == "f:cell":
                worst = max(worst, wb.coord_of(m))
    assert worst <= 16_383, f"a column coordinate reached {worst}, past Excel's own limit"
    assert worst < B.INT16_MAX
