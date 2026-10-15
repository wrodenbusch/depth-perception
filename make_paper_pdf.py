#!/usr/bin/env python
"""Render paper/depth-perception.md to paper/depth-perception.pdf with its figures embedded.

    .venv/bin/python make_paper_pdf.py

Markdown -> one self-contained HTML page (figures inlined as SVG data, so nothing is fetched) ->
headless Chromium's print-to-PDF. No LaTeX or pandoc is installed on this machine, and Chromium
already renders the figures' SVG exactly as the browser does. The preprint banner and page numbers
are CSS page-margin boxes, so they appear on every page.

The snap-packaged Chromium cannot read /tmp or hidden directories, so the intermediate HTML goes
in a visible scratch directory under $HOME, removed afterwards.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile

import markdown

SRC = "paper/depth-perception.md"
OUT = "paper/depth-perception.pdf"
# Read the date from the paper's byline so the banner and the byline cannot drift apart.
_BYLINE = re.search(r"Preprint, (\d{4}-\d{2}-\d{2})\.", open(SRC).read())
BANNER = f"Preprint, {_BYLINE.group(1)}." if _BYLINE else "Preprint."
CHROMIUM = os.environ.get("CHROMIUM", "/snap/bin/chromium")

CSS = f"""
@page {{
  size: Letter; margin: 0.66in 0.72in 0.62in 0.72in;
  @top-center {{ content: "{BANNER}"; font: 8.5pt Georgia, serif; color: #8a1c1c; }}
  @bottom-center {{ content: counter(page) " / " counter(pages); font: 8.5pt Georgia, serif; color: #555; }}
}}
body {{ font: 9.0pt/1.19 Georgia, 'Times New Roman', serif; color: #111; }}
h1 {{ font-size: 17pt; text-align: center; margin: 0 0 4pt; }}
h1 + p {{ text-align: center; color: #555; font-size: 9pt; }}
h2 {{ font-size: 12.5pt; margin: 12pt 0 4pt; break-after: avoid; page-break-after: avoid; }}
h3 {{ font-size: 11pt; margin: 10pt 0 3pt; break-after: avoid; page-break-after: avoid; }}
p, li {{ text-align: justify; hyphens: auto; }}
table {{ border-collapse: collapse; margin: 6pt auto; width: 100%; font-size: 7.8pt; }}
tr {{ break-inside: avoid; }}
th, td {{ border-bottom: 0.5pt solid #bbb; padding: 2pt 5pt; vertical-align: top; }}
th {{ border-bottom: 1pt solid #333; }}
img {{ display: block; width: 100%; margin: 8pt auto 2pt; page-break-inside: avoid; }}
/* the per-setting dot plot is tall; at 78% width it fits beside its section text instead of
   being pushed to the next page and leaving half a page blank */
img[alt^="Gain against the number of extra planes"] {{ width: 78%; }}
/* the headline per-setting figure is short on content per inch; 80% keeps it legible */
img[alt^="Perplexity reduction from sequence + tree + depth"] {{ width: 52%; }}
img[alt^="Which of a head's 64 planes"] {{ width: 60%; }}
/* the attention heatmap reads at 70% width and keeps the PDF within 10 pages */
img[alt^="Where the average head attends"] {{ width: 54%; }}
/* the two compact multi-panel figures read at 92% and keep the paper within its page budget */
img[alt^="Each configuration's gap to sequence only"] {{ width: 82%; }}
img[alt^="Perplexity reduction by kind of token"] {{ width: 82%; }}
/* References: their own page, filled column by column. Chromium balances a multi-column block
   across a page break only when the block is tall enough to need it, so a block that ALMOST fits
   the space left on the body's last page gets pushed whole onto two fresh pages -- shrinking the
   type made the paper LONGER. column-fill: auto with a fixed height and break-before: page makes
   the pagination monotone: the body is ten pages, the references are the eleventh. */
h2:last-of-type ~ p {{ font-size: 8.2pt; margin: 2pt 0; }}
h2:last-of-type {{ break-before: page; page-break-before: always; }}
.refs {{ column-count: 3; column-gap: 10pt; }}
.refs p {{ text-align: left; hyphens: manual; font-size: 6.0pt !important; line-height: 1.10; margin: 0 0 0.8pt !important; break-inside: avoid; }}
p.cap {{ font-size: 8.6pt; color: #333; margin-top: 0; page-break-before: avoid; }}
div.keep {{ break-inside: avoid; page-break-inside: avoid; }}
code {{ font-size: 9pt; }}
"""


def inline_figures(html: str) -> str:
    def repl(m):
        path = os.path.normpath(os.path.join("paper", m.group(1)))
        data = base64.b64encode(open(path, "rb").read()).decode()
        mime = "image/svg+xml" if path.endswith(".svg") else "image/png"
        return f'src="data:{mime};base64,{data}"'
    return re.sub(r'src="(\.\./docs/figures/[^"]+)"', repl, html)


def main() -> None:
    md = open(SRC).read()
    html = markdown.markdown(md, extensions=["tables"])
    # figure captions are italic paragraphs that start "Figure N." -- give them the caption style
    html = re.sub(r"<p><em>((?:Figure|Table) \d+\..*?)</em></p>", r'<p class="cap"><em>\1</em></p>', html, flags=re.S)
    # keep a figure or table together with its caption on one page: a caption split from its figure
    # across a page break was one of the referee's layout complaints (2026-10-03)
    html = re.sub(r'(<p><img [^>]*></p>\s*<p class="cap">.*?</p>)', r'<div class="keep">\1</div>', html, flags=re.S)
    html = re.sub(r'(<table>.*?</table>\s*<p class="cap">.*?</p>)', r'<div class="keep">\1</div>', html, flags=re.S)
    # references: two narrow columns in small type, so ~40 entries stay inside the page budget
    i = html.rfind("<h2>References</h2>")
    if i >= 0:
        j = i + len("<h2>References</h2>")
        html = html[:j] + '<div class="refs">' + html[j:] + "</div>"
    html = inline_figures(html)
    page = f"<!doctype html><meta charset=utf-8><title>paper</title><style>{CSS}</style><body>{html}</body>"
    work = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix="paperpdf-")
    try:
        src = os.path.join(work, "paper.html")
        pdf = os.path.join(work, "paper.pdf")
        open(src, "w").write(page)
        subprocess.run([CHROMIUM, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", f"file://{src}"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        os.replace(pdf, OUT)
    finally:
        for f in os.listdir(work):
            os.remove(os.path.join(work, f))
        os.rmdir(work)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
