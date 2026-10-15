#!/usr/bin/env python
"""Assemble the arXiv submission folder: the PDF, the plain-text abstract and the page counts.

    .venv/bin/python make_arxiv.py [--out DIR]

The abstract is lifted from the paper so the two cannot drift, with the markdown emphasis removed
and the typographic characters folded to ASCII -- arXiv's abstract field is plain text and renders
en dashes and curly quotes as mojibake. The page counts in SUBMIT.md come from the built PDF.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

PAPER = "paper/depth-perception.md"
PDF = "paper/depth-perception.pdf"
DEFAULT_OUT = os.path.expanduser("~/arxiv-depth-perception")

FOLD = {"—": "--", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "×": "x", "−": "-",
        "⅓": "1/3", "·": "-", "…": "...", " ": " ", "ₖ": "k", "θ": "theta", "₂": "2"}


def abstract(md: str) -> str:
    m = re.search(r"^## Abstract\s*\n(.*?)\n## ", md, re.S | re.M)
    if not m:
        raise SystemExit("no ## Abstract section in " + PAPER)
    t = m.group(1).strip()
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)          # bold
    t = re.sub(r"\*(.*?)\*", r"\1", t)              # italic
    t = re.sub(r"`(.*?)`", r"\1", t)                # code
    t = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", t)       # links
    t = re.sub(r"\s+", " ", t).strip()
    for a, b in FOLD.items():
        t = t.replace(a, b)
    bad = sorted({c for c in t if ord(c) > 126})
    if bad:
        raise SystemExit(f"abstract still has non-ASCII characters: {bad}")
    if len(t) > 1920:
        raise SystemExit(f"abstract is {len(t)} characters; arXiv's limit is 1920")
    return t


def pages(path: str) -> tuple[int, int]:
    """(total pages, pages before the references section)."""
    import pypdf
    r = pypdf.PdfReader(path)
    n = len(r.pages)
    body = n
    for i, p in enumerate(r.pages, 1):
        if "References" in p.extract_text():
            body = i - 1 if i > 1 else 1
            break
    return n, body


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    txt = abstract(open(PAPER).read())
    open(os.path.join(a.out, "abstract.txt"), "w").write(txt + "\n")
    shutil.copy2(PDF, os.path.join(a.out, "depth-perception.pdf"))
    total, body = pages(PDF)
    figs = len(re.findall(r"^!\[", open(PAPER).read(), re.M))
    sub = os.path.join(a.out, "SUBMIT.md")
    if os.path.exists(sub):
        s = open(sub).read()
        s = re.sub(r"paste `abstract\.txt` \(plain text, [\d,]+ characters;",
                   f"paste `abstract.txt` (plain text, {len(txt):,} characters;", s)
        s = re.sub(r"- \*\*Comments:\*\* `[^`]*`",
                   f"- **Comments:** `{body} pages plus references, {figs} figures. "
                   f"Code, data builders and interactive appendix: "
                   f"https://github.com/wrodenbusch/depth-perception`", s)
        open(sub, "w").write(s)
    print(f"abstract.txt: {len(txt):,} characters")
    print(f"depth-perception.pdf: {total} pages ({body} before the references), {figs} figures")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
