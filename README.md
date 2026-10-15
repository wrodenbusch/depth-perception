# Depth Perception for Transformers

**Telling models how deeply each token is nested.**

Almost every language model is told one thing about where a token is: its index in the sequence.
Yet code, web pages, spreadsheets and conversations are structured, and that structure is available
when the data is prepared. This repository tests the smallest change we could find: a **depth
marker**, two rotary (RoPE) planes per attention head that rotate by the token's *nesting depth*
(how many levels of the document's tree enclose it, not the network's layer count), beside
unchanged sequence position. Small decoder-only models trained from scratch on Python, spreadsheets,
web text and chat, compared seed-by-seed, show that the depth marker carries the web-text gain and
two-thirds of the code gain of a full 44-plane tree path at two planes. Where the data is grid-shaped,
one more small coordinate (row and column for tables; speaker and position-in-message for chat) is
needed. Controls show the gain is not structure the baseline lacked: showing the baseline the HTML
tags or the cell addresses as text gives it nothing, shuffling depth removes the gain, and on web
text rotary depth gains about three times what the same depth gains as an input embedding.

**When to use it.** The depth marker needs depth at inference. Where the structure is written in the
text (code, Markdown, JSON, HTML, chat roles), depth follows from the text, including while
generating; with web pages' markup kept in the token stream, depth adds 2.49% (10/10 seeds). Where
the markup is stripped, as in standard pretraining pipelines, depth must be supplied from the source:
it helps reading such documents, but a model asked to write free text with *predicted* depth did
worse than sequence position alone (paper §6, *Generation*).

- **Paper:** [paper/depth-perception.pdf](paper/depth-perception.pdf) ([markdown](paper/depth-perception.md))
- **Interactive appendix** (every comparison, per seed): <https://wrodenbusch.github.io/depth-perception/>
  (source: [docs/index.html](docs/index.html))

## Which coordinate to add

Perplexity reduction against sequence-only RoPE (positive = better), validation unless marked, with
seeds better and significance after a false-discovery-rate correction. Models are 4.7M non-embedding
parameters (the larger web run: 31.5M).

| data | coordinate beside sequence position | extra planes | gain | full tree (44 planes) |
|---|---|---|---|---|
| Web pages, markup in the text (tags kept; visible-text loss) | depth | 2 | 2.49% (10/10) | not run |
| Web pages, markup stripped | depth | 2 | 2.00% (10/10) | 1.20% (10/10) |
| Web pages, markup stripped, long context (one run, test) | depth | 2 | 1.68% | 2.15% |
| Code (Python) | depth | 2 | 0.76% (5/5) | 0.95% (5/5) |
| Code (Python) | depth + statement grid | 8 | 0.92% (5/5) | 0.95% (5/5) |
| Tables (spreadsheets) | depth | 2 | 0.25% (n.s.) | 1.19% (10/10) |
| Tables (spreadsheets) | row + column (+ sheet) | 6 | 1.09% (10/10) | 1.19% (10/10) |
| Chat | depth | 2 | 0.21% (n.s.) | 0.91% (9/10) |
| Chat | depth + speaker, exchange, position in message | 10 | 0.91% (9/10) | 0.91% (9/10) |

The effects are small and untested beyond 31.5M parameters. See the paper's limitations section.

## Repository layout

| path | what it is |
|---|---|
| `model.py`, `rope_pos.py`, `tree_pos.py` | nanoGPT with the partitioned rotary module: sequence, tree-path, depth, grid and reference coordinates on disjoint plane groups |
| `train.py`, `configurator.py`, `config/` | training (nanoGPT's loop) and one config per run arm |
| `topos_data.py`, `npy_stream.py` | packed-record loader that assembles each token's coordinates from a pack's tree sidecar |
| `build_*_pack.py`, `fetch_*.py` | data: fetch raw corpora and build packs (web, chat, spreadsheets, Python) |
| `slice_levels.py`, `depth_rank_pack.py`, `depth_parent_pack.py`, `parity_id_pack.py`, `shuffle_depth_pack.py`, `chat_grid_pack.py`, `py_stmtgrid_pack.py` | derived packs: same token stream, different coordinate (the controls and variants) |
| `eval_test.py`, `eval_visible_text.py`, `understanding_evals.py`, `analysis/` | held-out scoring, per-token-category scoring, attention and test-time-removal analyses |
| `paired.py` | the paired-seed comparison every result in the paper uses (refuses mismatched step counts) |
| `audit_paper_numbers.py` | recomputes every number in the paper from the run logs; `--strict` fails on any mismatch |
| `make_figures.py`, `make_appendix.py`, `make_paper_pdf.py`, `refresh_controls.sh` | figures, the appendix (`docs/index.html`) and the PDF, all built from the logs |
| `runqueue.py`, `add_jobs.py` | a small multi-GPU job queue |
| `test_*.py` | test suite (`pytest`) |

## Reproducing

**Environment.** Python 3.12, a CUDA GPU (results used NVIDIA RTX A4000s, one run per card).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

**Paths.** Nothing is hard-coded; set these (defaults live under `./work/`, see `topos_paths.py`):

```bash
export TOPOS_PACKS=/data/topos/packs      # built packs
export TOPOS_RUNS=/data/topos/runs        # training runs and analyses
export TOPOS_SOURCES=/data/topos/sources  # raw corpora read by the pack builders
export HF_HOME=/data/hf_cache             # Hugging Face cache for the fetch scripts
```

**Data.**
- *Web text* (C4 pages with HTML, `bs-modeling-metadata/c4-en-html-with-metadata`):
  `fetch_web_shards.py`, then `build_web_pack.py` (the short-page setting uses `--block 2048 --drop-long`,
  whole pages only; the tags-kept control is built by `build_web_tags_pack.py`).
- *Chat* (WildChat-1M, English): `fetch_wildchat.py`, then `build_chat_pack.py`. The builder prints
  aggregate statistics only; no conversation text is reproduced anywhere in this repository.
- *Spreadsheets*: `build_sheet_pack.py` reads a graph-record corpus of workbooks; the record format
  is documented in `docs/sheet_graph_corpus_ref.py.reference`.
- *Python*: `convert_treepos.py` converts an AST-rendered Python corpus (APPS + KodCode, 423k
  programs) with per-token tree positions into a pack. The renderer that produced that corpus is not
  part of this release.
- Variants and controls are derived from these packs with the `*_pack.py` scripts above; each
  config's header names the pack it reads.

No data or checkpoints are distributed.

**Train and compare.**

```bash
OMP_NUM_THREADS=8 .venv/bin/python train.py config/webc_depth.py --seed=0
OMP_NUM_THREADS=8 .venv/bin/python train.py config/webc_seq.py --seed=0
.venv/bin/python paired.py --root $TOPOS_RUNS/webc --a depth --b seq --require-step 9375
```

Cap CPU threads (`OMP_NUM_THREADS`): with one PyTorch process per GPU on a many-core machine, the
default of one thread per core makes idle threads spin and starves the GPUs.

**Rebuild the paper's numbers, figures, appendix and PDF** once runs exist:

```bash
.venv/bin/python audit_paper_numbers.py --strict
./refresh_controls.sh
```

## Collaboration

The open question is whether the depth marker's gain holds at billion-parameter scale. If you have the
compute or structured data to test it, I'd be glad to hear from you: will@rodenbusch.org.

## Citation

```bibtex
@misc{rodenbusch2026depthperception,
  title  = {Depth Perception for Transformers: Telling Models How Deeply Each Token Is Nested},
  author = {Rodenbusch, Will},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/wrodenbusch/depth-perception}
}
```

## Acknowledgements

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. The whole method builds
on rotary position embedding (RoPE) by Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen and
Yunfeng Liu, *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Neurocomputing, 2024;
arXiv:2104.09864).

Data, with thanks to their authors:
- Web pages: C4 (Raffel et al., 2020) with HTML, as released by BigScience
  (`bs-modeling-metadata/c4-en-html-with-metadata`).
- Spreadsheets: Sheetpedia (Tian, Han, Wang & Liao, NeurIPS 2025 Datasets and Benchmarks).
- Python: APPS (Hendrycks et al., 2021) and KodCode (Xu et al., 2025).
- Chat: WildChat-1M (Zhao et al., ICLR 2024), released under ODC-BY by the Allen Institute for AI.

No dataset content is redistributed in this repository; the builders download and process it.

## License

MIT; see [LICENSE](LICENSE). nanoGPT portions © Andrej Karpathy.
