#!/usr/bin/env python
"""Fetch N raw C4-HTML shards into the cache path ``build_web_pack.py`` globs.

    HF_HOME=$HF_HOME .venv/bin/python fetch_web_shards.py --shards 83

WHY THIS EXISTS. The raw corpus was deleted after the big pack was built, and the pack cannot
be re-derived into a shorter context without bespoke offset-remapping code -- the class of code
that has caused this project's worst bugs. Re-downloading is cheaper than it looks: a shard is
~458 MB and yields ~24M tokens once long pages are windowed rather than cropped, so a
1-billion-token confirmation corpus is ~42 shards and a 2-billion one ~83.

SORTED ORDER, NOT RANDOM. The original build consumed ``sorted(glob)[:1185]``, so taking the
first N here draws the confirmation corpus from the same region of the same corpus.

Resumable: ``hf_hub_download`` skips a file already complete in the cache, so re-running after
an interruption costs one HEAD request per shard.
"""
from __future__ import annotations
from topos_paths import HF_HOME  # noqa: E402

import argparse
import os
import sys
import time

REPO = "bs-modeling-metadata/c4-en-html-with-metadata"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--shards", type=int, default=83)
    ap.add_argument("--retries", type=int, default=5)
    a = ap.parse_args()
    if "HF_HOME" not in os.environ:
        sys.exit(f"[fetch] REFUSED: set HF_HOME={HF_HOME}, or this lands on /")

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()                     # a log file, not a terminal

    names = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.endswith(".jsonl.gz"))[:a.shards]
    print(f"[fetch] {len(names)} shards -> {os.environ['HF_HOME']}", flush=True)
    t0 = time.time()
    total = 0
    for i, name in enumerate(names, 1):
        for attempt in range(1, a.retries + 1):
            try:
                p = hf_hub_download(REPO, name, repo_type="dataset")
                break
            except Exception as exc:            # a flaky transfer must not lose the batch
                wait = min(60, 2 ** attempt)
                print(f"[fetch] {name} attempt {attempt}/{a.retries} "
                      f"{type(exc).__name__}: {exc} -- retrying in {wait}s", flush=True)
                if attempt == a.retries:
                    sys.exit(f"[fetch] GIVING UP on {name} after {a.retries} attempts")
                time.sleep(wait)
        total += os.path.getsize(p)
        el = time.time() - t0
        print(f"[fetch] {i}/{len(names)} {name}  {total/2**30:.1f} GiB  "
              f"{total/2**20/max(el,1):.0f} MiB/s  eta {el/i*(len(names)-i)/60:.0f} min",
              flush=True)
    print(f"[fetch] done: {total/2**30:.1f} GiB in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
